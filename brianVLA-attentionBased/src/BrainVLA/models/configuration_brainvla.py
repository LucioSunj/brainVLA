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
BrainVLA Configuration Classes

This module implements configuration classes for the BrainVLA three-expert 
Vision-Language-Action model with unified attention mechanism.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("brainvla")
@dataclass
class BrainVLAConfig(PreTrainedConfig):
    """
    Configuration class for BrainVLA three-expert Vision-Language-Action model.
    
    This configuration manages the unified attention mechanism between three experts:
    - Prefix Expert (PaliGemma VLM): Processes vision and language inputs
    - Middle Expert (Affordance): Predicts affordance maps 
    - Suffix Expert (Action): Generates robot actions via flow matching
    
    Example:
        ```python
        config = BrainVLAConfig(
            affordance_expert_type="sam",
            n_affordance_queries=8,
            allow_action_see_affordance=True,
            sam_checkpoint_path="sam_vit_h_4b8939.pth",
            affordance_focal_loss_weight=0.1,
            affordance_dice_loss_weight=1.0,
        )
        ```
    """
    
    # Model identification
    model_type: str = "brainvla"
    name: str = "brainvla"
    
    # ===== Basic LeRobot Configuration =====
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    
    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    
    # Input/output dimensions  
    max_state_dim: int = 32
    max_action_dim: int = 32
    
    # Image preprocessing
    resize_imgs_with_padding: Optional[tuple[int, int]] = (224, 224)
    empty_cameras: int = 0
    
    # Tokenizer
    tokenizer_max_length: int = 48
    
    # ===== Three-Expert Architecture Configuration =====
    
    # Core hidden dimensions
    hidden_size: int = 1024  # Shared transformer hidden dimension
    proj_width: int = 1024   # Projection layer width
    
    # Expert interaction controls
    affordance_expert_type: str = "sam"  # "sam" or "query_to_mask"
    allow_action_see_affordance: bool = False  # Action can attend to affordance
    allow_affordance_see_action: bool = False  # Affordance can attend to action
    
    # ===== Affordance Expert Configuration =====
    
    # Query configuration
    n_affordance_queries: int = 8  # Number of affordance query tokens
    max_affordance_dim: int = 256  # Maximum affordance feature dimension
    affordance_query_dim: int = 256  # Affordance query embedding dimension
    
    # Quality estimation
    use_affordance_quality_head: bool = True  # Predict IoU/quality scores
    quality_head_dim: int = 64  # Quality head hidden dimension
    
    # SAM configuration (when affordance_expert_type="sam")
    sam_checkpoint_path: str = "sam_vit_h_4b8939.pth"
    train_sam_mask_decoder: bool = True
    freeze_sam_image_encoder: bool = True
    sam_multimask_output: bool = False
    
    # Query-to-mask configuration (when affordance_expert_type="query_to_mask")
    mask_output_size: int = 224  # Output mask resolution
    affordance_intermediate_dim: int = 512  # Intermediate layer dimension
    
    # ===== Action Expert Configuration =====
    
    # Flow matching
    num_denoising_steps: int = 10  # Number of denoising steps for flow matching
    flow_sigma: float = 1.0  # Noise scale for flow matching
    
    # ===== Loss Weights =====
    
    # Action losses
    action_flow_loss_weight: float = 1.0
    
    # Affordance losses
    affordance_focal_loss_weight: float = 0.1
    affordance_dice_loss_weight: float = 1.0
    affordance_ce_loss_weight: float = 1.0
    affordance_quality_loss_weight: float = 0.2
    
    # Multi-task balancing
    total_affordance_loss_weight: float = 0.5  # Overall affordance vs action balance
    
    # ===== Attention & Efficiency Configuration =====
    
    use_cache: bool = True
    attention_implementation: str = "eager"  # "eager", "fa2", or "flex"
    use_prefix_cache: bool = True  # Enable prefix caching for inference
    gradient_checkpointing: bool = False  # Memory optimization for training
    
    # ===== Training Configuration =====
    
    # Model training behavior
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False  # Train only expert parts, freeze VLM
    train_affordance_expert: bool = True
    train_action_expert: bool = True
    train_state_proj: bool = True
    
    # Optimization
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    
    # Scheduling
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    
    # ===== Expert Model Configurations =====
    
    # VLM Expert (PaliGemma) Configuration
    vlm_config: Dict = field(
        default_factory=lambda: {
            "bos_token_id": 2,
            "eos_token_id": 1,
            "hidden_size": 2048,
            "image_token_index": 257152,
            "model_type": "paligemma",
            "pad_token_id": 0,
            "projection_dim": 2048,
            "text_config": {
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 2048,
                "intermediate_size": 16384,
                "model_type": "gemma",
                "num_attention_heads": 16,  # Higher for VLM
                "num_hidden_layers": 18,
                "num_image_tokens": 256,
                "num_key_value_heads": 1,
                "torch_dtype": "float32",
                "vocab_size": 257152,
            },
            "vision_config": {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "model_type": "siglip_vision_model",
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "num_image_tokens": 256,
                "patch_size": 14,
                "projection_dim": 2048,
                "projector_hidden_act": "gelu_fast",
                "vision_use_head": False,
            },
            "vocab_size": 257152,
        }
    )
    
    # Affordance Expert Configuration (Medium-sized, specialized for spatial reasoning)
    affordance_expert_config: Dict = field(
        default_factory=lambda: {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 128,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 512,  # Medium size for affordance reasoning
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 8192,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 18,  # Full depth for affordance understanding
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "torch_dtype": "float32",
            "use_cache": True,
            "vocab_size": 257152,
        }
    )
    
    # Action Expert Configuration (Full-sized, specialized for action generation)
    action_expert_config: Dict = field(
        default_factory=lambda: {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 1024,  # Full size for action generation
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 8192,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "torch_dtype": "float32",
            "use_cache": True,
            "vocab_size": 257152,
        }
    )
    
    # Expert Training Configuration
    train_affordance_expert: bool = True
    train_action_expert: bool = True
    
    # ===== Experimental Features =====
    
    # Future extensions
    enable_affordance_refinement: bool = False  # Multi-stage affordance refinement
    use_cross_expert_communication: bool = False  # Additional cross-expert layers
    
    # Aloha compatibility (inherited from PI0)
    adapt_to_pi_aloha: bool = False
    use_delta_joint_actions_aloha: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters and set derived values."""
        super().__post_init__()
        
        # Basic validation
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
            
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not supported yet. Got n_obs_steps={self.n_obs_steps}"
            )
        
        # Expert type validation
        if self.affordance_expert_type not in ["sam", "query_to_mask"]:
            raise ValueError(
                f"Unknown affordance_expert_type: {self.affordance_expert_type}. "
                "Must be 'sam' or 'query_to_mask'"
            )
        
        # Expert interaction validation
        if self.allow_action_see_affordance and not self.train_affordance_expert:
            raise ValueError(
                "Cannot allow action to see affordance if affordance expert is not trained"
            )
            
        if self.allow_affordance_see_action and not self.train_action_expert:
            raise ValueError(
                "Cannot allow affordance to see action if action expert is not trained"
            )
        
        # Attention implementation validation
        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"attention_implementation must be one of ['eager', 'fa2', 'flex'], "
                f"got {self.attention_implementation}"
            )
        
        # Dimension validation
        if self.n_affordance_queries <= 0:
            raise ValueError(f"n_affordance_queries must be positive, got {self.n_affordance_queries}")
            
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        
        # SAM-specific validation
        if self.affordance_expert_type == "sam":
            if not self.sam_checkpoint_path:
                raise ValueError("sam_checkpoint_path must be provided when using SAM affordance expert")
        
        # Loss weight validation
        if self.total_affordance_loss_weight < 0:
            raise ValueError("total_affordance_loss_weight must be non-negative")
            
        # Warn about attention head compatibility
        self._validate_attention_heads()
        
        # Aloha compatibility check
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "use_delta_joint_actions_aloha is not implemented yet in BrainVLA"
            )
    
    def _validate_attention_heads(self):
        """Validate attention head configurations for compatibility."""
        for expert_name, expert_config in [
            ("affordance", self.affordance_expert_config),
            ("action", self.action_expert_config)
        ]:
            hidden_size = expert_config["hidden_size"]
            num_heads = expert_config["num_attention_heads"]
            
            if hidden_size % num_heads != 0:
                import warnings
                warnings.warn(
                    f"{expert_name}_expert hidden_size ({hidden_size}) is not divisible by "
                    f"num_attention_heads ({num_heads}). This may cause issues."
                )
    
    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        # Add empty camera features if specified
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera
    
    def get_optimizer_preset(self) -> AdamWConfig:
        """Get optimizer configuration preset."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )
    
    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Get scheduler configuration preset."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
    
    @property
    def observation_delta_indices(self) -> None:
        """Observation delta indices (not used in BrainVLA)."""
        return None
    
    @property
    def action_delta_indices(self) -> List[int]:
        """Action delta indices for temporal modeling."""
        return list(range(self.chunk_size))
    
    @property
    def reward_delta_indices(self) -> None:
        """Reward delta indices (not used in BrainVLA)."""
        return None
    
    def get_experiment_name(self) -> str:
        """Generate experiment name based on key configuration flags."""
        name_parts = [self.model_type]
        
        # Expert type
        name_parts.append(f"aff_{self.affordance_expert_type}")
        
        # Attention routing
        if self.allow_action_see_affordance and self.allow_affordance_see_action:
            name_parts.append("bidirectional")
        elif self.allow_action_see_affordance:
            name_parts.append("act2aff")
        elif self.allow_affordance_see_action:
            name_parts.append("aff2act")
        else:
            name_parts.append("isolated")
        
        # Key parameters
        name_parts.append(f"q{self.n_affordance_queries}")
        name_parts.append(f"h{self.hidden_size}")
        
        return "_".join(name_parts)
    
    def merge_with(self, other_config: Union[Dict, 'BrainVLAConfig']) -> 'BrainVLAConfig':
        """Merge this config with another config, with other_config taking precedence."""
        if isinstance(other_config, dict):
            # Create a copy of this config and update with dict values
            config_dict = self.to_dict()
            config_dict.update(other_config)
            return BrainVLAConfig(**config_dict)
        elif isinstance(other_config, BrainVLAConfig):
            # Merge two BrainVLAConfig objects
            config_dict = self.to_dict()
            other_dict = other_config.to_dict()
            config_dict.update(other_dict)
            return BrainVLAConfig(**config_dict)
        else:
            raise TypeError(f"Cannot merge with type {type(other_config)}")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        # Use dataclass fields to get all attributes
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BrainVLAConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


# ===== Convenience Functions =====

def create_default_brainvla_config(**kwargs) -> BrainVLAConfig:
    """Create a default BrainVLA configuration with optional overrides.
    
    Args:
        **kwargs: Optional configuration overrides
        
    Returns:
        BrainVLAConfig: The configured instance
        
    Example:
        ```python
        config = create_default_brainvla_config(
            affordance_expert_type="query_to_mask",
            allow_action_see_affordance=True,
            n_affordance_queries=16
        )
        ```
    """
    return BrainVLAConfig(**kwargs)


def create_sam_affordance_config(**kwargs) -> BrainVLAConfig:
    """Create a BrainVLA configuration optimized for SAM-based affordance prediction.
    
    Args:
        **kwargs: Optional configuration overrides
        
    Returns:
        BrainVLAConfig: The configured instance with SAM optimizations
    """
    sam_defaults = {
        "affordance_expert_type": "sam",
        "train_sam_mask_decoder": True,
        "freeze_sam_image_encoder": True,
        "affordance_focal_loss_weight": 0.1,
        "affordance_dice_loss_weight": 1.0,
        "use_affordance_quality_head": True,
    }
    sam_defaults.update(kwargs)
    return BrainVLAConfig(**sam_defaults)


def create_query_to_mask_config(**kwargs) -> BrainVLAConfig:
    """Create a BrainVLA configuration for simple query-to-mask affordance prediction.
    
    Args:
        **kwargs: Optional configuration overrides
        
    Returns:
        BrainVLAConfig: The configured instance with query-to-mask optimizations
    """
    q2m_defaults = {
        "affordance_expert_type": "query_to_mask",
        "mask_output_size": 224,
        "affordance_intermediate_dim": 512,
        "affordance_ce_loss_weight": 1.0,
        "affordance_dice_loss_weight": 0.5,
    }
    q2m_defaults.update(kwargs)
    return BrainVLAConfig(**q2m_defaults)