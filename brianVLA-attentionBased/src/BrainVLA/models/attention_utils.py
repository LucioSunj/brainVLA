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
BrainVLA Attention Utilities

This module implements precise blockwise attention mask construction for the
three-expert unified attention mechanism in BrainVLA.

The attention routing follows these rules:
- VLM ↔ VLM: Complete bidirectional attention
- Affordance → VLM: Can attend to VLM context  
- Action → VLM: Can attend to VLM context
- Affordance ↔ Affordance: Internal bidirectional attention
- Action ↔ Action: Internal bidirectional attention
- Action ↔ Affordance: Configurable cross-expert interaction
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def build_blockwise_mask(
    vlm_len: int,
    affordance_len: int, 
    action_len: int,
    batch_size: int,
    device: torch.device,
    allow_action_see_affordance: bool = False,
    allow_affordance_see_action: bool = False,
) -> torch.Tensor:
    """
    Construct precise blockwise attention mask for three-expert architecture.
    
    This function explicitly constructs a 2D attention mask without using cumulative
    sum tricks, ensuring precise control over expert interactions.
    
    Sequence organization: [VLM tokens | Affordance tokens | Action tokens]
    
    Attention rules:
    - VLM ↔ VLM: Complete bidirectional attention
    - Affordance → VLM: Can attend to VLM (unidirectional)
    - Action → VLM: Can attend to VLM (unidirectional)  
    - Affordance ↔ Affordance: Internal bidirectional attention
    - Action ↔ Action: Internal bidirectional attention
    - Action ↔ Affordance: Configurable bidirectional interaction
    
    Args:
        vlm_len: Number of VLM tokens
        affordance_len: Number of affordance tokens
        action_len: Number of action tokens
        batch_size: Batch size
        device: Device for tensor creation
        allow_action_see_affordance: Whether action tokens can attend to affordance tokens
        allow_affordance_see_action: Whether affordance tokens can attend to action tokens
        
    Returns:
        torch.Tensor: Boolean attention mask of shape (batch_size, total_len, total_len)
                     where True indicates allowed attention and False indicates blocked.
                     
    Example:
        ```python
        mask = build_blockwise_mask(
            vlm_len=10, affordance_len=5, action_len=8,
            batch_size=2, device=torch.device("cpu"),
            allow_action_see_affordance=True
        )
        # mask.shape: (2, 23, 23)
        ```
    """
    if vlm_len < 0 or affordance_len < 0 or action_len < 0:
        raise ValueError("All sequence lengths must be non-negative")
        
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    total_len = vlm_len + affordance_len + action_len
    
    if total_len == 0:
        raise ValueError("Total sequence length cannot be zero")
    
    # Initialize mask as False (no attention allowed by default)
    mask = torch.zeros(batch_size, total_len, total_len, dtype=torch.bool, device=device)
    
    # Define block boundaries
    vlm_start, vlm_end = 0, vlm_len
    aff_start, aff_end = vlm_len, vlm_len + affordance_len
    act_start, act_end = vlm_len + affordance_len, total_len
    
    # 1. VLM ↔ VLM: Complete bidirectional attention
    if vlm_len > 0:
        mask[:, vlm_start:vlm_end, vlm_start:vlm_end] = True
    
    # 2. Affordance → VLM: Unidirectional (affordance can see VLM)
    if affordance_len > 0 and vlm_len > 0:
        mask[:, aff_start:aff_end, vlm_start:vlm_end] = True
    
    # 3. Action → VLM: Unidirectional (action can see VLM)
    if action_len > 0 and vlm_len > 0:
        mask[:, act_start:act_end, vlm_start:vlm_end] = True
    
    # 4. Affordance ↔ Affordance: Internal bidirectional attention
    if affordance_len > 0:
        mask[:, aff_start:aff_end, aff_start:aff_end] = True
    
    # 5. Action ↔ Action: Internal bidirectional attention
    if action_len > 0:
        mask[:, act_start:act_end, act_start:act_end] = True
    
    # 6. Configurable Action ↔ Affordance interaction
    if action_len > 0 and affordance_len > 0:
        if allow_action_see_affordance:
            # Action → Affordance
            mask[:, act_start:act_end, aff_start:aff_end] = True
            
        if allow_affordance_see_action:
            # Affordance → Action  
            mask[:, aff_start:aff_end, act_start:act_end] = True
    
    return mask


def build_prefix_middle_suffix_mask(
    prefix_len: int,
    middle_len: int,
    suffix_len: int, 
    batch_size: int,
    device: torch.device,
    allow_suffix_see_middle: bool = False,
    allow_middle_see_suffix: bool = False,
) -> torch.Tensor:
    """
    Alternative naming for build_blockwise_mask with prefix-middle-suffix terminology.
    
    This is a more intuitive interface for the three-stage architecture.
    
    Args:
        prefix_len: Length of prefix block (VLM)
        middle_len: Length of middle block (Affordance)  
        suffix_len: Length of suffix block (Action)
        batch_size: Batch size
        device: Device for tensor creation
        allow_suffix_see_middle: Whether suffix can attend to middle
        allow_middle_see_suffix: Whether middle can attend to suffix
        
    Returns:
        torch.Tensor: Boolean attention mask
    """
    return build_blockwise_mask(
        vlm_len=prefix_len,
        affordance_len=middle_len,
        action_len=suffix_len,
        batch_size=batch_size,
        device=device,
        allow_action_see_affordance=allow_suffix_see_middle,
        allow_affordance_see_action=allow_middle_see_suffix,
    )


def apply_padding_mask(
    attention_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply padding mask to blockwise attention mask.
    
    Args:
        attention_mask: Blockwise attention mask (batch_size, seq_len, seq_len)
        padding_mask: Padding mask (batch_size, seq_len) where True indicates valid tokens
        
    Returns:
        torch.Tensor: Combined attention mask accounting for padding
    """
    if attention_mask.shape[:2] != padding_mask.shape:
        raise ValueError(
            f"Shape mismatch: attention_mask {attention_mask.shape[:2]} vs "
            f"padding_mask {padding_mask.shape}"
        )
    
    # Create 2D padding mask: (batch_size, seq_len, seq_len)
    # Only allow attention between valid (non-padded) tokens
    padding_2d = padding_mask[:, None, :] & padding_mask[:, :, None]
    
    # Combine with blockwise attention mask
    combined_mask = attention_mask & padding_2d
    
    return combined_mask


def get_attention_block_info(
    vlm_len: int,
    affordance_len: int,
    action_len: int,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Get block boundary information for the three experts.
    
    Args:
        vlm_len: VLM sequence length
        affordance_len: Affordance sequence length  
        action_len: Action sequence length
        
    Returns:
        Tuple of (start, end) indices for (vlm, affordance, action) blocks
    """
    vlm_block = (0, vlm_len)
    affordance_block = (vlm_len, vlm_len + affordance_len)
    action_block = (vlm_len + affordance_len, vlm_len + affordance_len + action_len)
    
    return vlm_block, affordance_block, action_block


def visualize_attention_mask(
    attention_mask: torch.Tensor,
    vlm_len: int,
    affordance_len: int,
    action_len: int,
    batch_idx: int = 0,
) -> str:
    """
    Create a text visualization of the attention mask for debugging.
    
    Args:
        attention_mask: Attention mask tensor
        vlm_len: VLM sequence length
        affordance_len: Affordance sequence length
        action_len: Action sequence length  
        batch_idx: Which batch to visualize
        
    Returns:
        str: Text representation of the attention mask
    """
    if batch_idx >= attention_mask.shape[0]:
        raise ValueError(f"batch_idx {batch_idx} >= batch_size {attention_mask.shape[0]}")
    
    mask_2d = attention_mask[batch_idx].cpu().numpy()
    
    # Create block labels
    vlm_block, aff_block, act_block = get_attention_block_info(vlm_len, affordance_len, action_len)
    
    lines = []
    lines.append("Attention Mask Visualization:")
    lines.append(f"VLM: [{vlm_block[0]}:{vlm_block[1]}], "
                f"Affordance: [{aff_block[0]}:{aff_block[1]}], "  
                f"Action: [{act_block[0]}:{act_block[1]}]")
    lines.append("")
    
    # Create header
    header = "    "
    for i in range(mask_2d.shape[1]):
        if i < vlm_len:
            header += "V"
        elif i < vlm_len + affordance_len:
            header += "A" 
        else:
            header += "C"
    lines.append(header)
    
    # Create rows
    for i in range(mask_2d.shape[0]):
        if i < vlm_len:
            row_label = "V"
        elif i < vlm_len + affordance_len:
            row_label = "A"
        else:
            row_label = "C"
        
        row = f"{row_label:2d} "
        for j in range(mask_2d.shape[1]):
            row += "1" if mask_2d[i, j] else "0"
        lines.append(row)
    
    lines.append("")
    lines.append("Legend: V=VLM, A=Affordance, C=Action (1=allowed, 0=blocked)")
    
    return "\n".join(lines)


def create_causal_mask(
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a standard causal (lower triangular) attention mask.
    
    Args:
        seq_len: Sequence length
        batch_size: Batch size
        device: Device for tensor creation
        
    Returns:
        torch.Tensor: Causal attention mask (batch_size, seq_len, seq_len)
    """
    # Create lower triangular mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    
    # Expand to batch dimension
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    return mask


def combine_masks(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    operation: str = "and",
) -> torch.Tensor:
    """
    Combine two attention masks with specified operation.
    
    Args:
        mask1: First attention mask
        mask2: Second attention mask  
        operation: Combination operation ("and", "or", "xor")
        
    Returns:
        torch.Tensor: Combined attention mask
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"Shape mismatch: {mask1.shape} vs {mask2.shape}")
    
    if operation == "and":
        return mask1 & mask2
    elif operation == "or":
        return mask1 | mask2
    elif operation == "xor":
        return mask1 ^ mask2
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'and', 'or', or 'xor'")


def validate_attention_mask(
    attention_mask: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """
    Validate attention mask properties.
    
    Args:
        attention_mask: Attention mask to validate
        expected_shape: Expected shape tuple (optional)
        
    Returns:
        bool: True if mask is valid
        
    Raises:
        ValueError: If mask is invalid
    """
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError("attention_mask must be a torch.Tensor")
    
    if attention_mask.dtype != torch.bool:
        raise ValueError(f"attention_mask must be bool type, got {attention_mask.dtype}")
    
    if attention_mask.ndim != 3:
        raise ValueError(f"attention_mask must be 3D (batch, seq, seq), got {attention_mask.ndim}D")
    
    batch_size, seq_len1, seq_len2 = attention_mask.shape
    if seq_len1 != seq_len2:
        raise ValueError(f"attention_mask must be square in last 2 dims, got {seq_len1}x{seq_len2}")
    
    if expected_shape is not None and attention_mask.shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {attention_mask.shape}")
    
    return True


# ===== Convenience Functions =====

def create_isolated_experts_mask(
    vlm_len: int,
    affordance_len: int,
    action_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create mask where experts cannot see each other (baseline configuration)."""
    return build_blockwise_mask(
        vlm_len=vlm_len,
        affordance_len=affordance_len,
        action_len=action_len,
        batch_size=batch_size,
        device=device,
        allow_action_see_affordance=False,
        allow_affordance_see_action=False,
    )


def create_action_to_affordance_mask(
    vlm_len: int,
    affordance_len: int,
    action_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create mask where action can see affordance (one-way interaction).""" 
    return build_blockwise_mask(
        vlm_len=vlm_len,
        affordance_len=affordance_len,
        action_len=action_len,
        batch_size=batch_size,
        device=device,
        allow_action_see_affordance=True,
        allow_affordance_see_action=False,
    )


def create_bidirectional_experts_mask(
    vlm_len: int,
    affordance_len: int,
    action_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create mask with full bidirectional expert interaction."""
    return build_blockwise_mask(
        vlm_len=vlm_len,
        affordance_len=affordance_len,
        action_len=action_len,
        batch_size=batch_size,
        device=device,
        allow_action_see_affordance=True,
        allow_affordance_see_action=True,
    )