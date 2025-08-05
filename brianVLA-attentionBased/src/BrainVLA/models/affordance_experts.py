#!/usr/bin/env python

# Copyright 2024 BrainVLA Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
BrainVLA Affordance Experts

This module implements different affordance expert implementations that can be
plugged into the BrainVLA architecture as middle blocks. The experts use attention
mechanisms to obtain information from transformer internals for affordance prediction.
"""

import importlib
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffordanceExpertInterface(ABC):
    """
    Abstract interface for affordance experts in BrainVLA.
    
    This interface ensures that different affordance implementations can be
    easily swapped while maintaining consistent behavior for attention routing
    and transformer integration.
    """
    
    @abstractmethod
    def get_attention_queries(
        self, 
        batch_size: int, 
        device: torch.device,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate affordance attention queries for transformer processing.
        
        Args:
            batch_size: Batch size
            device: Device for tensor creation
            context: Optional context information (e.g., from VLM)
            
        Returns:
            torch.Tensor: Affordance queries (batch_size, n_queries, hidden_dim)
        """
        pass
    
    @abstractmethod
    def get_affordance_embeddings(
        self, 
        transformer_features: torch.Tensor,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate affordance predictions from transformer features.
        
        Args:
            transformer_features: Features from transformer (batch_size, n_queries, hidden_dim)
            images: Input images (batch_size, channels, height, width)
            **kwargs: Additional arguments for specific implementations
            
        Returns:
            torch.Tensor: Affordance predictions/embeddings
        """
        pass
    
    @abstractmethod  
    def compute_affordance_loss(
        self, 
        transformer_features: torch.Tensor,
        images: torch.Tensor, 
        gt_masks: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute affordance prediction losses.
        
        Args:
            transformer_features: Features from transformer
            images: Input images
            gt_masks: Ground truth affordance masks
            **kwargs: Additional arguments for specific implementations
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        pass


class SAMAffordanceExpert(AffordanceExpertInterface, nn.Module):
    """
    SAM-based affordance expert implementation.
    
    This expert uses the Segment Anything Model (SAM) for affordance prediction,
    with transformer features converted to SAM prompt embeddings. The implementation
    is inspired by GLOVER but redesigned for BrainVLA's architecture.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize SAM model
        self._init_sam_model()
        
        # Transformer feature to SAM embedding projection
        self.feature_to_sam_projection = nn.Sequential(
            nn.Linear(
                config.affordance_expert_config["hidden_size"], 
                config.affordance_expert_config["hidden_size"]
            ),
            nn.ReLU(inplace=True),
            nn.Linear(config.affordance_expert_config["hidden_size"], 256),  # SAM embedding dimension
            nn.Dropout(0.0),
        )
        
        # Learnable affordance query embeddings
        self.affordance_queries = nn.Parameter(
            torch.randn(config.n_affordance_queries, config.affordance_expert_config["hidden_size"])
        )
        
        # Optional quality head for IoU prediction
        if getattr(config, 'use_affordance_quality_head', False):
            self.quality_head = nn.Sequential(
                nn.Linear(config.affordance_expert_config["hidden_size"], config.quality_head_dim),
                nn.ReLU(),
                nn.Linear(config.quality_head_dim, 1),
                nn.Sigmoid()
            )
    
    def _init_sam_model(self):
        """Initialize SAM model with proper configuration."""
        try:
            # Try to import SAM - this is a soft dependency
            # Users need to install segment-anything package
            from segment_anything import build_sam_vit_h
            
            self.sam_model = build_sam_vit_h(self.config.sam_checkpoint_path)
            
            # Configure training behavior
            if self.config.freeze_sam_image_encoder:
                self.sam_model.image_encoder.eval()
                for param in self.sam_model.image_encoder.parameters():
                    param.requires_grad = False
            
            if self.config.train_sam_mask_decoder:
                self.sam_model.mask_decoder.train()
                for param in self.sam_model.mask_decoder.parameters():
                    param.requires_grad = True
                    
        except ImportError:
            raise ImportError(
                "SAM dependencies not found. Please install segment-anything: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM model: {e}")
    
    def get_attention_queries(
        self, 
        batch_size: int, 
        device: torch.device,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate affordance attention queries."""
        queries = self.affordance_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        
        # Optional: incorporate context information
        if context is not None and hasattr(self, 'context_projection'):
            context_emb = self.context_projection(context.mean(dim=1, keepdim=True))
            queries = queries + context_emb
        
        return queries
    
    def get_affordance_embeddings(
        self, 
        transformer_features: torch.Tensor,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate affordance masks using SAM from transformer features.
        
        The process follows GLOVER's approach:
        1. Convert transformer features to SAM prompt embeddings
        2. Use SAM image encoder to get image features
        3. Use SAM prompt encoder and mask decoder to generate masks
        """
        batch_size, n_queries, hidden_dim = transformer_features.shape
        
        # Convert transformer features to SAM prompt embeddings
        sam_prompt_embeddings = self.feature_to_sam_projection(transformer_features)
        
        # Get SAM image embeddings (similar to GLOVER's approach)
        with torch.no_grad() if self.config.freeze_sam_image_encoder else torch.enable_grad():
            image_embeddings_list = []
            for i in range(images.shape[0]):
                # Clear cache to manage memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                img_emb = self.sam_model.image_encoder(images[i].unsqueeze(0))
                image_embeddings_list.append(img_emb)
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            image_embeddings = torch.cat(image_embeddings_list, 0)
        
        # Generate affordance masks for each query
        predicted_masks = []
        for i in range(batch_size):
            batch_masks = []
            for j in range(n_queries):
                # Use SAM prompt encoder (following GLOVER's pattern)
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=sam_prompt_embeddings[i, j].unsqueeze(0).unsqueeze(0),
                )
                
                # SAM mask decoder
                sparse_embeddings = sparse_embeddings.to(sam_prompt_embeddings.dtype)
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=self.config.sam_multimask_output,
                )
                
                # Post-process masks to target size (following GLOVER's approach)
                pred_mask = self.sam_model.postprocess_masks(
                    low_res_masks,
                    input_size=images.shape[-2:],
                    original_size=images.shape[-2:],
                )
                
                batch_masks.append(pred_mask[:, 0])  # Take first mask if multimask_output=False
            
            predicted_masks.append(torch.cat(batch_masks, dim=0))
        
        return torch.stack(predicted_masks, dim=0)
    
    def compute_affordance_loss(
        self, 
        transformer_features: torch.Tensor,
        images: torch.Tensor, 
        gt_masks: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute affordance losses using focal and dice loss (following GLOVER)."""
        pred_masks = self.get_affordance_embeddings(transformer_features, images)
        
        batch_size, n_queries = pred_masks.shape[:2]
        
        losses = {}
        total_loss = 0
        
        # Focal loss for affordance prediction
        focal_loss_val = 0
        dice_loss_val = 0
        num_masks = 0
        
        for batch_idx in range(batch_size):
            gt_mask = gt_masks[batch_idx] if len(gt_masks.shape) > 3 else gt_masks
            pred_mask = pred_masks[batch_idx]
            
            # Ensure shapes match
            if gt_mask.shape[0] != pred_mask.shape[0]:
                # Handle shape mismatch by taking minimum
                min_queries = min(gt_mask.shape[0], pred_mask.shape[0])
                gt_mask = gt_mask[:min_queries]
                pred_mask = pred_mask[:min_queries]
            
            # Focal loss (following GLOVER's implementation pattern)
            focal_loss_val += sigmoid_focal_loss(
                pred_mask, gt_mask, num_boxes=gt_mask.shape[0]
            ) * gt_mask.shape[0]
            
            # Dice loss (following GLOVER's implementation pattern)  
            dice_loss_val += dice_loss(
                pred_mask, gt_mask, num_masks=gt_mask.shape[0]
            ) * gt_mask.shape[0]
            
            num_masks += gt_mask.shape[0]
        
        # Normalize and weight losses
        if num_masks > 0:
            losses['affordance_focal_loss'] = (focal_loss_val * 0.1 / num_masks) * self.config.affordance_focal_loss_weight
            losses['affordance_dice_loss'] = (dice_loss_val / num_masks) * self.config.affordance_dice_loss_weight
            total_loss = losses['affordance_focal_loss'] + losses['affordance_dice_loss']
        
        # Optional quality loss
        if hasattr(self, 'quality_head'):
            quality_pred = self.quality_head(transformer_features)
            # Simple IoU-based quality target (can be improved)
            with torch.no_grad():
                iou_targets = self._compute_iou_targets(pred_masks, gt_masks)
            quality_loss = F.mse_loss(quality_pred.squeeze(-1), iou_targets)
            losses['affordance_quality_loss'] = quality_loss * self.config.affordance_quality_loss_weight
            total_loss += losses['affordance_quality_loss']
        
        losses['total_affordance_loss'] = total_loss
        return losses
    
    def _compute_iou_targets(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """Compute IoU targets for quality head training."""
        batch_size, n_queries = pred_masks.shape[:2]
        iou_targets = torch.zeros(batch_size, n_queries, device=pred_masks.device)
        
        pred_binary = (pred_masks.sigmoid() > 0.5).float()
        
        for b in range(batch_size):
            for q in range(n_queries):
                if q < gt_masks.shape[1]:  # Handle shape mismatch
                    intersection = (pred_binary[b, q] * gt_masks[b, q]).sum()
                    union = (pred_binary[b, q] + gt_masks[b, q]).clamp(0, 1).sum()
                    iou_targets[b, q] = intersection / (union + 1e-6)
        
        return iou_targets


class DirectAffordanceExpert(AffordanceExpertInterface, nn.Module):
    """
    Direct affordance expert implementation.
    
    This expert directly predicts affordance maps from transformer features
    without using external models like SAM. Useful for faster inference and
    when SAM is not available.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Direct affordance prediction network
        self.affordance_predictor = nn.Sequential(
            nn.Linear(config.affordance_expert_config["hidden_size"], config.affordance_intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.affordance_intermediate_dim, config.affordance_intermediate_dim // 2),
            nn.ReLU(),
            nn.Linear(config.affordance_intermediate_dim // 2, config.mask_output_size * config.mask_output_size),
        )
        
        # Learnable affordance query embeddings
        self.affordance_queries = nn.Parameter(
            torch.randn(config.n_affordance_queries, config.affordance_expert_config["hidden_size"])
        )
        
        # Optional context integration
        if getattr(config, 'use_context_integration', False):
            self.context_projection = nn.Linear(
                config.paligemma_config["hidden_size"], 
                config.affordance_expert_config["hidden_size"]
            )
    
    def get_attention_queries(
        self, 
        batch_size: int, 
        device: torch.device,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate affordance attention queries."""
        queries = self.affordance_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        
        # Optional: incorporate context information from VLM
        if context is not None and hasattr(self, 'context_projection'):
            context_emb = self.context_projection(context.mean(dim=1, keepdim=True))
            queries = queries + context_emb
        
        return queries
    
    def get_affordance_embeddings(
        self, 
        transformer_features: torch.Tensor,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Generate affordance maps directly from transformer features."""
        batch_size, n_queries, hidden_dim = transformer_features.shape
        
        # Direct prediction from transformer features
        affordance_logits = self.affordance_predictor(transformer_features)
        affordance_maps = affordance_logits.view(
            batch_size, n_queries, 
            self.config.mask_output_size, 
            self.config.mask_output_size
        )
        
        # Upsample to image size if needed
        target_size = images.shape[-2:]
        if target_size != (self.config.mask_output_size, self.config.mask_output_size):
            affordance_maps = F.interpolate(
                affordance_maps, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return affordance_maps
    
    def compute_affordance_loss(
        self, 
        transformer_features: torch.Tensor,
        images: torch.Tensor, 
        gt_masks: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute affordance losses using CE and dice loss."""
        pred_masks = self.get_affordance_embeddings(transformer_features, images)
        
        losses = {}
        
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction='mean')
        losses['affordance_bce_loss'] = bce_loss * self.config.affordance_ce_loss_weight
        
        # Dice loss
        dice_loss_val = dice_loss(pred_masks, gt_masks, num_masks=pred_masks.shape[0] * pred_masks.shape[1])
        losses['affordance_dice_loss'] = dice_loss_val * self.config.affordance_dice_loss_weight
        
        # Total loss
        total_loss = losses['affordance_bce_loss'] + losses['affordance_dice_loss']
        losses['total_affordance_loss'] = total_loss
        
        return losses


# ===== Loss Functions (adapted from GLOVER_plus) =====

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for addressing class imbalance in dense prediction.
    
    Args:
        inputs: Predicted logits
        targets: Ground truth binary masks
        num_boxes: Number of positive examples for normalization
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        
    Returns:
        torch.Tensor: Focal loss value
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
        
    return loss.mean(1).sum() / (num_boxes + 1e-8)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: int,
    scale: float = 1000.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Dice loss for segmentation tasks.
    
    Args:
        inputs: Predicted logits
        targets: Ground truth binary masks  
        num_masks: Number of masks for normalization
        scale: Scaling factor for numerical stability
        eps: Small epsilon for numerical stability
        
    Returns:
        torch.Tensor: Dice loss value
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    
    return loss


# ===== Factory Function =====

def create_affordance_expert(config) -> AffordanceExpertInterface:
    """
    Factory function to create affordance experts based on configuration.
    
    Args:
        config: BrainVLA configuration object
        
    Returns:
        AffordanceExpertInterface: Configured affordance expert instance
    """
    expert_type = config.affordance_expert_type.lower()
    
    if expert_type == "sam":
        return SAMAffordanceExpert(config)
    elif expert_type in ["direct", "query_to_mask"]:
        return DirectAffordanceExpert(config)
    elif expert_type == "custom":
        # Support for custom affordance expert implementations
        if not hasattr(config, 'affordance_expert_class') or not config.affordance_expert_class:
            raise ValueError("Custom affordance expert type requires 'affordance_expert_class' in config")
        
        try:
            module_name, class_name = config.affordance_expert_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            expert_class = getattr(module, class_name)
            
            if not issubclass(expert_class, AffordanceExpertInterface):
                raise TypeError(f"Custom expert class must implement AffordanceExpertInterface")
                
            return expert_class(config)
        except Exception as e:
            raise RuntimeError(f"Failed to create custom affordance expert: {e}")
    else:
        raise ValueError(
            f"Unknown affordance expert type: {expert_type}. "
            f"Supported types: 'sam', 'direct', 'query_to_mask', 'custom'"
        )


# ===== Utility Functions =====

def validate_affordance_expert(expert: AffordanceExpertInterface) -> bool:
    """
    Validate that an affordance expert implements the required interface.
    
    Args:
        expert: Affordance expert instance to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        TypeError: If expert doesn't implement required methods
    """
    required_methods = ['get_attention_queries', 'get_affordance_embeddings', 'compute_affordance_loss']
    
    for method_name in required_methods:
        if not hasattr(expert, method_name):
            raise TypeError(f"Affordance expert missing required method: {method_name}")
        
        method = getattr(expert, method_name)
        if not callable(method):
            raise TypeError(f"Affordance expert method {method_name} is not callable")
    
    return True


def get_affordance_expert_info(expert: AffordanceExpertInterface) -> Dict[str, str]:
    """
    Get information about an affordance expert.
    
    Args:
        expert: Affordance expert instance
        
    Returns:
        Dict[str, str]: Information about the expert
    """
    info = {
        'class_name': expert.__class__.__name__,
        'module': expert.__class__.__module__,
        'type': 'sam' if isinstance(expert, SAMAffordanceExpert) else 'direct',
    }
    
    if hasattr(expert, 'config'):
        info['n_affordance_queries'] = str(getattr(expert.config, 'n_affordance_queries', 'unknown'))
        
    return info