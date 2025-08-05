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
BrainVLA Policy Implementation

This module implements the main BrainVLA policy class that integrates:
- Triple expert architecture (VLM, Affordance, Action)
- Blockwise attention routing
- Flow matching for action generation
- Pluggable affordance experts

The design follows LeRobot's PreTrainedPolicy interface while implementing
BrainVLA's specialized expert architecture.
"""

import math
from collections import deque
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.utils import get_safe_dtype, init_logging

from .configuration_brainvla import BrainVLAConfig
from .affordance_experts import create_affordance_expert, AffordanceExpertInterface
from .attention_utils import build_blockwise_mask
from .paligemma_with_triple_expert import (
    BrainVLAWithTripleExpertConfig,
    BrainVLAWithTripleExpertModel,
)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Create sinusoidal positional embeddings for flow matching timesteps."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Pad vector to new dimension."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1) -> torch.Tensor:
    """Resize image with padding to maintain aspect ratio."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # Pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


class BrainVLAPolicy(PreTrainedPolicy):
    """
    BrainVLA Policy with triple expert architecture.
    
    This policy implements a three-expert Vision-Language-Action model where:
    - VLM Expert: Processes vision and language inputs (PaliGemma)
    - Affordance Expert: Predicts object affordances (pluggable implementation)
    - Action Expert: Generates robot actions via flow matching (Gemma)
    
    All experts share attention computation through blockwise routing while
    maintaining specialized parameters for expert-specific learning.
    """

    config_class = BrainVLAConfig
    name = "brainvla"

    def __init__(
        self,
        config: BrainVLAConfig,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        """
        Initialize BrainVLA policy.
        
        Args:
            config: BrainVLA configuration
            dataset_stats: Dataset statistics for normalization
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        
        # Initialize normalization
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Initialize language tokenizer
        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        
        # Initialize main model
        self.model = BrainVLAFlowMatching(config)
        
        # Reset internal state
        self.reset()

    def reset(self):
        """Reset internal state (e.g., action queues)."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> Dict:
        """Get optimizer parameters."""
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions (not implemented for flow matching)."""
        raise NotImplementedError("Chunk prediction not implemented for BrainVLA flow matching")

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor], noise: Optional[Tensor] = None) -> Tensor:
        """
        Select a single action for environment execution.
        
        Uses action queue to return actions one at a time, refilling the queue
        when empty by sampling from the flow matching model.
        """
        self.eval()

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            # Prepare model inputs
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            # Sample actions from flow matching model
            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )

            # Unpad actions to original dimensions
            original_action_dim = self.config.action_feature.shape[0] if hasattr(self.config, 'action_feature') else actions.shape[-1]
            if actions.shape[-1] > original_action_dim:
                actions = actions[:, :, :original_action_dim]

            # Unnormalize actions
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # Fill action queue (transpose for queue format)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def forward(self, batch: Dict[str, Tensor], noise=None, time=None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Training forward pass.
        
        Args:
            batch: Training batch containing images, language, state, actions, etc.
            noise: Optional noise for flow matching (will be sampled if None)
            time: Optional time for flow matching (will be sampled if None)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Prepare model inputs
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        
        # Get affordance masks if available
        gt_affordance_masks = batch.get("affordance_masks", None)
        
        # Check for action padding
        actions_is_pad = batch.get("action_is_pad")

        # Forward through model
        outputs = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            gt_affordance_masks=gt_affordance_masks,
            noise=noise, time=time
        )

        loss_dict = {}
        total_loss = 0

        # Action loss (flow matching)
        if "action_losses" in outputs:
            action_losses = outputs["action_losses"]
            
            # Apply action padding mask if available
            if actions_is_pad is not None:
                in_episode_bound = ~actions_is_pad
                action_losses = action_losses * in_episode_bound.unsqueeze(-1)

            # Remove padding dimensions
            action_losses = action_losses[:, :, :self.config.max_action_dim]
            
            action_loss = action_losses.mean()
            loss_dict["action_loss"] = action_loss.item()
            total_loss += action_loss

        # Affordance losses
        if "affordance_losses" in outputs and outputs["affordance_losses"]:
            affordance_losses = outputs["affordance_losses"]
            total_affordance_loss = 0
            
            for loss_name, loss_value in affordance_losses.items():
                if loss_name != "total_affordance_loss":
                    loss_dict[f"affordance_{loss_name}"] = loss_value.item()
                    total_affordance_loss += loss_value
            
            # Apply affordance loss weight
            weighted_affordance_loss = total_affordance_loss * self.config.total_affordance_loss_weight
            loss_dict["total_affordance_loss"] = weighted_affordance_loss.item()
            total_loss += weighted_affordance_loss

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def prepare_images(self, batch: Dict[str, Tensor]) -> Tuple[list, list]:
        """Prepare images with BrainVLA preprocessing."""
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.input_features if "image" in key and key in batch]
        missing_img_keys = [key for key in self.config.input_features if "image" in key and key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError("At least one image feature is required")

        # Process present images
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from [0,1] to [-1,1] for SigLIP 
            img = img * 2.0 - 1.0

            batch_size = img.shape[0]
            device = img.device
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Add empty cameras if configured
        for num_empty_cameras in range(min(len(missing_img_keys), self.config.empty_cameras)):
            img = torch.ones_like(images[0]) * -1  # Empty image
            mask = torch.zeros_like(img_masks[0])  # Masked out
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Prepare language tokens."""
        device = next(iter(batch.values())).device
        tasks = batch["task"]

        # PaliGemma prompt must end with newline
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch: Dict[str, Tensor]) -> Tensor:
        """Prepare state vector with padding."""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Prepare action vector with padding."""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


class BrainVLAFlowMatching(nn.Module):
    """
    BrainVLA Flow Matching Model.
    
    This model integrates the triple expert architecture with flow matching
    for action generation and affordance prediction through attention routing.
    """

    def __init__(self, config: BrainVLAConfig):
        super().__init__()
        self.config = config

        # Initialize triple expert model
        triple_expert_config = BrainVLAWithTripleExpertConfig(
            vlm_config=config.vlm_config,
            affordance_expert_config=config.affordance_expert_config,
            action_expert_config=config.action_expert_config,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            train_affordance_expert=config.train_affordance_expert,
            train_action_expert=config.train_action_expert,
            attention_implementation=config.attention_implementation,
        )
        self.brainvla_with_triple_expert = BrainVLAWithTripleExpertModel(triple_expert_config)

        # Initialize pluggable affordance expert
        self.affordance_expert = create_affordance_expert(config)

        # Projection layers for flow matching
        self.state_proj = nn.Linear(config.max_state_dim, config.proj_width)
        self.action_in_proj = nn.Linear(config.max_action_dim, config.proj_width)
        self.action_out_proj = nn.Linear(config.proj_width, config.max_action_dim)

        # Flow matching time embedding MLPs
        self.action_time_mlp_in = nn.Linear(config.proj_width * 2, config.proj_width)
        self.action_time_mlp_out = nn.Linear(config.proj_width, config.proj_width)

        # Set gradient requirements
        self.set_requires_grad()

    def set_requires_grad(self):
        """Configure gradient requirements based on config."""
        for param in self.state_proj.parameters():
            param.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample noise for flow matching."""
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time steps for flow matching."""
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((batch_size,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_vlm(
        self, images: list, img_masks: list, lang_tokens: torch.Tensor, lang_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed VLM inputs (images + language)."""
        # TODO: Avoid list concatenation, use pre-allocation
        embs = []
        pad_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.brainvla_with_triple_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings  
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            batch_size, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(batch_size, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

        # Process language
        lang_emb = self.brainvla_with_triple_expert.embed_language_tokens(lang_tokens)
        
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        vlm_embs = torch.cat(embs, dim=1)
        vlm_pad_masks = torch.cat(pad_masks, dim=1)

        return vlm_embs, vlm_pad_masks

    def embed_affordance(self, batch_size: int, device: torch.device, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed affordance queries."""
        affordance_queries = self.affordance_expert.get_attention_queries(batch_size, device, context)
        affordance_mask = torch.ones(batch_size, affordance_queries.shape[1], dtype=torch.bool, device=device)
        return affordance_queries, affordance_mask

    def embed_action_suffix(self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed action suffix with flow matching."""
        embs = []
        pad_masks = []

        batch_size = state.shape[0]
        device = state.device
        dtype = torch.bfloat16

        # Embed state
        state_emb = self.state_proj(state).to(dtype=dtype)
        embs.append(state_emb[:, None, :])
        
        state_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Embed timestep
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=dtype)

        # Fuse timestep + action with MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)

        action_time_mask = torch.ones(batch_size, action_time_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        action_embs = torch.cat(embs, dim=1)
        action_pad_masks = torch.cat(pad_masks, dim=1)

        return action_embs, action_pad_masks

    def forward(
        self,
        images: list,
        img_masks: list,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        gt_affordance_masks: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Training forward pass through BrainVLA."""
        
        # Sample noise and time for flow matching
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Flow matching interpolation
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed all components
        vlm_embs, vlm_pad_masks = self.embed_vlm(images, img_masks, lang_tokens, lang_masks)
        affordance_embs, affordance_pad_masks = self.embed_affordance(state.shape[0], state.device, vlm_embs)
        action_embs, action_pad_masks = self.embed_action_suffix(state, x_t, time)

        # Get sequence lengths for attention mask
        vlm_len = vlm_embs.shape[1]
        affordance_len = affordance_embs.shape[1]
        action_len = action_embs.shape[1]

        # Build blockwise attention mask
        attention_mask = build_blockwise_mask(
            vlm_len=vlm_len,
            affordance_len=affordance_len,
            action_len=action_len,
            batch_size=state.shape[0],
            device=state.device,
            allow_action_see_affordance=self.config.allow_action_see_affordance,
            allow_affordance_see_action=self.config.allow_affordance_see_action,
        )

        # Apply padding mask
        pad_masks = torch.cat([vlm_pad_masks, affordance_pad_masks, action_pad_masks], dim=1)
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        attention_mask = attention_mask & pad_2d_masks

        # Position IDs
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Forward through triple expert model
        expert_outputs, _ = self.brainvla_with_triple_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            vlm_embeds=vlm_embs,
            affordance_embeds=affordance_embs,
            action_embeds=action_embs,
            use_cache=False,
            fill_kv_cache=False,
        )

        results = {}

        # Action loss (flow matching)
        if expert_outputs[2] is not None:  # Action expert output
            action_output = expert_outputs[2][:, -self.config.n_action_steps:]
            action_output = action_output.to(dtype=torch.float32)
            v_t = self.action_out_proj(action_output)
            action_losses = F.mse_loss(u_t, v_t, reduction="none")
            results["action_losses"] = action_losses

        # Affordance loss
        if expert_outputs[1] is not None and gt_affordance_masks is not None:  # Affordance expert output
            affordance_output = expert_outputs[1]
            affordance_losses = self.affordance_expert.compute_affordance_loss(
                affordance_output, images[0], gt_affordance_masks
            )
            results["affordance_losses"] = affordance_losses

        return results

    @torch.no_grad()
    def sample_actions(
        self,
        images: list,
        img_masks: list,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample actions using flow matching denoising."""
        batch_size = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (batch_size, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Embed VLM (this will be cached)
        vlm_embs, vlm_pad_masks = self.embed_vlm(images, img_masks, lang_tokens, lang_masks)
        affordance_embs, affordance_pad_masks = self.embed_affordance(batch_size, device, vlm_embs)

        # Build prefix attention mask and cache VLM + affordance
        vlm_len = vlm_embs.shape[1]
        affordance_len = affordance_embs.shape[1]
        
        prefix_pad_masks = torch.cat([vlm_pad_masks, affordance_pad_masks], dim=1)
        prefix_attention_mask = build_blockwise_mask(
            vlm_len=vlm_len + affordance_len, affordance_len=0, action_len=0,
            batch_size=batch_size, device=device
        )
        
        pad_2d_masks = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
        prefix_attention_mask = prefix_attention_mask & pad_2d_masks
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Cache prefix computation
        prefix_embs = torch.cat([vlm_embs, affordance_embs], dim=1)
        _, past_key_values = self.brainvla_with_triple_expert.forward(
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            vlm_embeds=prefix_embs,
            affordance_embeds=None,
            action_embeds=None,
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # Iterative denoising
        dt = -1.0 / self.config.num_denoising_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t += dt * v_t
            time += dt

        return x_t

    def denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step for flow matching."""
        # Embed action suffix
        action_embs, action_pad_masks = self.embed_action_suffix(state, x_t, timestep)

        # Build attention mask for suffix
        vlm_affordance_len = prefix_pad_masks.shape[1]
        action_len = action_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]

        # Create mask that allows action to attend to cached prefix
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, action_len, vlm_affordance_len)
        
        action_attention_mask = build_blockwise_mask(
            vlm_len=0, affordance_len=0, action_len=action_len,
            batch_size=batch_size, device=state.device
        )
        
        # Combine with prefix attention
        full_attention_mask = torch.cat([prefix_pad_2d_masks, action_attention_mask], dim=2)

        # Position IDs for action tokens
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(action_pad_masks, dim=1) - 1

        # Forward with cached prefix
        expert_outputs, _ = self.brainvla_with_triple_expert.forward(
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            vlm_embeds=None,
            affordance_embeds=None,
            action_embeds=action_embs,
            past_key_values=past_key_values,
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )

        # Extract action output and predict velocity
        action_output = expert_outputs[2][:, -self.config.n_action_steps:]
        action_output = action_output.to(dtype=torch.float32)
        v_t = self.action_out_proj(action_output)
        
        return v_t