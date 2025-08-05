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
BrainVLA Triple Expert Model

This module implements the specialized transformer wrapper that enables three experts
(VLM, Affordance, Action) to share attention computation while maintaining separate
parameters. Each expert uses its own transformer layers but shares the attention space.

The design follows π0's approach where experts have independent parameters but unified attention.
"""

from typing import List, Optional, Union, Tuple
import math

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000) -> torch.Tensor:
    """
    Apply Rotary Position Embedding (RoPE) to input tensor.
    
    Args:
        x: Input tensor of shape [B, L, H, D]
        positions: Position tensor of shape [B, L]
        max_wavelength: Maximum wavelength for frequency computation
        
    Returns:
        torch.Tensor: Tensor with RoPE applied
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=device
    )
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class BrainVLAWithTripleExpertConfig(PretrainedConfig):
    """Configuration for BrainVLA triple expert model."""
    
    model_type = "BrainVLAWithTripleExpertModel"
    
    def __init__(
        self,
        vlm_config: Optional[dict] = None,
        affordance_expert_config: Optional[dict] = None,
        action_expert_config: Optional[dict] = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = False,
        train_affordance_expert: bool = True,
        train_action_expert: bool = True,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.train_affordance_expert = train_affordance_expert
        self.train_action_expert = train_action_expert
        self.attention_implementation = attention_implementation

        # VLM (PaliGemma) configuration
        if vlm_config is None:
            self.vlm_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
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
            )
        else:
            if isinstance(vlm_config, dict):
                if "model_type" not in vlm_config:
                    vlm_config["model_type"] = "paligemma"
                cfg_cls = CONFIG_MAPPING[vlm_config["model_type"]]
                self.vlm_config = cfg_cls(**vlm_config)
            else:
                self.vlm_config = vlm_config

        # Affordance expert (Gemma) configuration
        if affordance_expert_config is None:
            self.affordance_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=128,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=512,  # Smaller than VLM and action expert
                initializer_range=0.02,
                intermediate_size=2048,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        else:
            if isinstance(affordance_expert_config, dict):
                if "model_type" not in affordance_expert_config:
                    affordance_expert_config["model_type"] = "gemma"
                cfg_cls = CONFIG_MAPPING[affordance_expert_config["model_type"]]
                self.affordance_expert_config = cfg_cls(**affordance_expert_config)
            else:
                self.affordance_expert_config = affordance_expert_config

        # Action expert (Gemma) configuration
        if action_expert_config is None:
            self.action_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,  # Medium size for action expert
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        else:
            if isinstance(action_expert_config, dict):
                if "model_type" not in action_expert_config:
                    action_expert_config["model_type"] = "gemma"
                cfg_cls = CONFIG_MAPPING[action_expert_config["model_type"]]
                self.action_expert_config = cfg_cls(**action_expert_config)
            else:
                self.action_expert_config = action_expert_config

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        
        # Validation
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). "
                f"Expected 'eager', 'fa2' or 'flex'."
            )


class BrainVLAWithTripleExpertModel(PreTrainedModel):
    """
    BrainVLA Triple Expert Model with specialized parameters.
    
    This model implements three independent experts (VLM, Affordance, Action) that
    share attention computation but maintain separate transformer parameters. This
    design enables expert specialization while allowing cross-expert communication
    through shared attention space.
    
    Architecture:
    - VLM Expert: PaliGemma for vision-language understanding
    - Affordance Expert: Gemma for affordance prediction
    - Action Expert: Gemma for action generation
    
    Each expert has its own transformer layers but they share attention computation
    through blockwise routing masks.
    """
    
    config_class = BrainVLAWithTripleExpertConfig

    def __init__(self, config: BrainVLAWithTripleExpertConfig):
        super().__init__(config=config)
        self.config = config
        
        # Initialize three independent expert models
        self.vlm_expert = PaliGemmaForConditionalGeneration(config.vlm_config)
        self.affordance_expert = GemmaForCausalLM(config.affordance_expert_config)
        self.action_expert = GemmaForCausalLM(config.action_expert_config)
        
        # Remove unused embedding layers from expert models (they will use projected embeddings)
        self.affordance_expert.model.embed_tokens = None
        self.affordance_expert.lm_head = None
        self.action_expert.model.embed_tokens = None  
        self.action_expert.lm_head = None
        
        # Apply mixed precision like π0
        self.to_bfloat16_like_pi0()
        
        # Set training behavior
        self.set_requires_grad()

    def set_requires_grad(self):
        """Configure which parameters should be trained based on config."""
        if self.config.freeze_vision_encoder:
            self.vlm_expert.vision_tower.eval()
            for param in self.vlm_expert.vision_tower.parameters():
                param.requires_grad = False

        if self.config.train_expert_only:
            # Freeze VLM, only train experts
            self.vlm_expert.eval()
            for param in self.vlm_expert.parameters():
                param.requires_grad = False
        
        # Configure affordance expert training
        if self.config.train_affordance_expert:
            self.affordance_expert.train()
            for param in self.affordance_expert.parameters():
                param.requires_grad = True
        else:
            self.affordance_expert.eval()
            for param in self.affordance_expert.parameters():
                param.requires_grad = False
                
        # Configure action expert training
        if self.config.train_action_expert:
            self.action_expert.train()
            for param in self.action_expert.parameters():
                param.requires_grad = True
        else:
            self.action_expert.eval()
            for param in self.action_expert.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train to respect frozen components."""
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.vlm_expert.vision_tower.eval()

        if self.config.train_expert_only:
            self.vlm_expert.eval()
            
        if not self.config.train_affordance_expert:
            self.affordance_expert.eval()
            
        if not self.config.train_action_expert:
            self.action_expert.eval()

    def to_bfloat16_like_pi0(self):
        """Apply mixed precision configuration similar to π0."""
        # Convert main models to bfloat16
        self.vlm_expert = self.vlm_expert.to(dtype=torch.bfloat16)

        # Keep specific parameter types in bfloat16
        params_to_change_dtype = [
            "language_model.model.layers",
            "affordance_expert.model.layers", 
            "action_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed images using VLM's vision tower."""
        return self.vlm_expert.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using VLM's language model."""
        return self.vlm_expert.language_model.model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], dict]] = None,
        vlm_embeds: Optional[torch.FloatTensor] = None,
        affordance_embeds: Optional[torch.FloatTensor] = None,
        action_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ) -> Tuple[Tuple[Optional[torch.Tensor], ...], Optional[dict]]:
        """
        Forward pass through the triple expert model.
        
        This method processes inputs through three specialized experts while sharing
        attention computation. Each expert uses its own transformer layers but they
        all participate in unified attention.
        
        Args:
            attention_mask: Blockwise attention mask for expert routing
            position_ids: Position IDs for RoPE
            past_key_values: Cached key-value states for efficiency
            vlm_embeds: VLM expert embeddings
            affordance_embeds: Affordance expert embeddings  
            action_embeds: Action expert embeddings
            use_cache: Whether to use KV caching
            fill_kv_cache: Whether to fill KV cache
            
        Returns:
            Tuple of (expert_outputs, past_key_values)
        """
        # Organize expert models and embeddings
        models = [
            self.vlm_expert.language_model.model,  # VLM expert
            self.affordance_expert.model,          # Affordance expert
            self.action_expert.model               # Action expert
        ]
        
        inputs_embeds = [vlm_embeds, affordance_embeds, action_embeds]
        
        # Get batch size from first available embeddings
        batch_size = None
        for embeds in inputs_embeds:
            if embeds is not None:
                batch_size = embeds.shape[0]
                break
        
        if batch_size is None:
            raise ValueError("At least one expert embedding must be provided")

        # Get model configuration (use VLM config as reference)
        num_layers = self.vlm_expert.config.text_config.num_hidden_layers
        head_dim = self.vlm_expert.config.text_config.head_dim
        
        # Process through transformer layers (following π0's approach)
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            
            # Compute Q, K, V for each expert using their own parameters
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                    
                # Get the specific layer for this expert
                layer = models[i].layers[layer_idx]
                
                # Apply layer normalization with expert's own parameters
                hidden_states = layer.input_layernorm(hidden_states)
                
                # Compute shapes for multi-head attention
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)

                # Compute Q, K, V using expert's own projection parameters
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # Concatenate Q, K, V from all experts for unified attention
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            # Apply RoPE
            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            # Handle KV caching
            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # Concatenate with cached states
                    key_states = torch.cat(
                        [past_key_values[layer_idx]["key_states"], key_states], dim=1
                    )
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

            # Compute unified attention across all experts
            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask,
                batch_size,
                head_dim,
                query_states,
                key_states,
                value_states,
            )
            att_output = att_output.to(dtype=torch.bfloat16)

            # Distribute attention output back to each expert and continue processing
            outputs_embeds = []
            start = 0
            
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is not None:
                    layer = models[i].layers[layer_idx]
                    end = start + hidden_states.shape[1]

                    # Extract this expert's attention output
                    expert_att_output = att_output[:, start:end]
                    
                    # Apply output projection with expert's own parameters
                    if expert_att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        expert_att_output = expert_att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(expert_att_output)

                    # First residual connection
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    # MLP with expert's own parameters
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # Second residual connection
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # Final layer normalization for each expert
        final_outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                final_outputs.append(out_emb)
            else:
                final_outputs.append(None)

        return tuple(final_outputs), past_key_values

    def get_attention_interface(self):
        """Get attention computation interface based on config."""
        if self.config.attention_implementation == "fa2":
            return self.flash_attention_forward
        else:
            return self.eager_attention_forward

    def flash_attention_forward(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Flash attention implementation (placeholder)."""
        raise NotImplementedError("Flash Attention 2 is not implemented yet")

    def eager_attention_forward(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Standard eager attention implementation."""
        num_att_heads = self.vlm_expert.config.text_config.num_attention_heads
        num_key_value_heads = self.vlm_expert.config.text_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        # Shapes: query_states: [batch_size, seq_len, num_att_heads, head_dim]
        #         key_states, value_states: [batch_size, seq_len, num_key_value_heads, head_dim]
        sequence_length = key_states.shape[1]

        # Expand key and value states for grouped query attention
        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim,
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim,
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim,
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim,
        )

        # Compute attention weights
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim ** -0.5
        
        # Apply attention mask
        big_neg = -2.3819763e38  # Large negative value for masking
        masked_att_weights = torch.where(
            attention_mask[:, None, :, :], att_weights, big_neg
        )

        # Softmax and apply to values
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        
        # Reshape output
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )

        return att_output