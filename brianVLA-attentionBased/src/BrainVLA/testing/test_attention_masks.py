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
Unit tests for BrainVLA attention mask construction utilities.

This module tests the precise blockwise attention mask construction
that controls information flow between the three experts.
"""

import pytest
import torch
import sys
import os

# Add the models directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from attention_utils import (
    build_blockwise_mask,
    build_prefix_middle_suffix_mask,
    apply_padding_mask,
    get_attention_block_info,
    visualize_attention_mask,
    create_causal_mask,
    combine_masks,
    validate_attention_mask,
    create_isolated_experts_mask,
    create_action_to_affordance_mask,
    create_bidirectional_experts_mask,
)


class TestBuildBlockwiseMask:
    """Test suite for build_blockwise_mask function."""
    
    def test_basic_mask_construction(self):
        """Test basic mask construction with default settings."""
        batch_size, device = 2, torch.device("cpu")
        vlm_len, aff_len, act_len = 10, 5, 8
        
        mask = build_blockwise_mask(vlm_len, aff_len, act_len, batch_size, device)
        
        # Verify shape
        expected_shape = (batch_size, vlm_len + aff_len + act_len, vlm_len + aff_len + act_len)
        assert mask.shape == expected_shape
        assert mask.dtype == torch.bool
        assert mask.device == device
    
    def test_vlm_bidirectional_attention(self):
        """Test that VLM tokens can attend to each other bidirectionally."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        mask = build_blockwise_mask(vlm_len, aff_len, act_len, batch_size, device)
        
        # VLM block should have complete bidirectional attention
        vlm_block = mask[0, :vlm_len, :vlm_len]
        assert vlm_block.all(), "VLM tokens should attend to all other VLM tokens"
    
    def test_affordance_to_vlm_attention(self):
        """Test that affordance tokens can attend to VLM tokens."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        mask = build_blockwise_mask(vlm_len, aff_len, act_len, batch_size, device)
        
        # Affordance → VLM should be allowed
        aff_to_vlm = mask[0, vlm_len:vlm_len+aff_len, :vlm_len]
        assert aff_to_vlm.all(), "Affordance tokens should attend to VLM tokens"
        
        # VLM → Affordance should NOT be allowed (by default)
        vlm_to_aff = mask[0, :vlm_len, vlm_len:vlm_len+aff_len]
        assert not vlm_to_aff.any(), "VLM tokens should not attend to affordance tokens"
    
    def test_action_to_vlm_attention(self):
        """Test that action tokens can attend to VLM tokens."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        mask = build_blockwise_mask(vlm_len, aff_len, act_len, batch_size, device)
        
        # Action → VLM should be allowed
        act_to_vlm = mask[0, vlm_len+aff_len:, :vlm_len]
        assert act_to_vlm.all(), "Action tokens should attend to VLM tokens"
        
        # VLM → Action should NOT be allowed (by default)
        vlm_to_act = mask[0, :vlm_len, vlm_len+aff_len:]
        assert not vlm_to_act.any(), "VLM tokens should not attend to action tokens"
    
    def test_internal_expert_attention(self):
        """Test internal bidirectional attention within each expert."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        mask = build_blockwise_mask(vlm_len, aff_len, act_len, batch_size, device)
        
        # Affordance internal attention
        aff_internal = mask[0, vlm_len:vlm_len+aff_len, vlm_len:vlm_len+aff_len]
        assert aff_internal.all(), "Affordance tokens should attend to each other"
        
        # Action internal attention
        act_internal = mask[0, vlm_len+aff_len:, vlm_len+aff_len:]
        assert act_internal.all(), "Action tokens should attend to each other"
    
    def test_configurable_action_to_affordance(self):
        """Test configurable action → affordance attention."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        # Test with action → affordance enabled
        mask_enabled = build_blockwise_mask(
            vlm_len, aff_len, act_len, batch_size, device,
            allow_action_see_affordance=True
        )
        act_to_aff_enabled = mask_enabled[0, vlm_len+aff_len:, vlm_len:vlm_len+aff_len]
        assert act_to_aff_enabled.all(), "Action should attend to affordance when enabled"
        
        # Test with action → affordance disabled (default)
        mask_disabled = build_blockwise_mask(
            vlm_len, aff_len, act_len, batch_size, device,
            allow_action_see_affordance=False
        )
        act_to_aff_disabled = mask_disabled[0, vlm_len+aff_len:, vlm_len:vlm_len+aff_len]
        assert not act_to_aff_disabled.any(), "Action should not attend to affordance when disabled"
    
    def test_configurable_affordance_to_action(self):
        """Test configurable affordance → action attention."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        # Test with affordance → action enabled
        mask_enabled = build_blockwise_mask(
            vlm_len, aff_len, act_len, batch_size, device,
            allow_affordance_see_action=True
        )
        aff_to_act_enabled = mask_enabled[0, vlm_len:vlm_len+aff_len, vlm_len+aff_len:]
        assert aff_to_act_enabled.all(), "Affordance should attend to action when enabled"
        
        # Test with affordance → action disabled (default)
        mask_disabled = build_blockwise_mask(
            vlm_len, aff_len, act_len, batch_size, device,
            allow_affordance_see_action=False
        )
        aff_to_act_disabled = mask_disabled[0, vlm_len:vlm_len+aff_len, vlm_len+aff_len:]
        assert not aff_to_act_disabled.any(), "Affordance should not attend to action when disabled"
    
    def test_bidirectional_expert_interaction(self):
        """Test bidirectional expert interaction."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        mask = build_blockwise_mask(
            vlm_len, aff_len, act_len, batch_size, device,
            allow_action_see_affordance=True,
            allow_affordance_see_action=True
        )
        
        # Both directions should be enabled
        act_to_aff = mask[0, vlm_len+aff_len:, vlm_len:vlm_len+aff_len]
        aff_to_act = mask[0, vlm_len:vlm_len+aff_len, vlm_len+aff_len:]
        
        assert act_to_aff.all(), "Action → Affordance should be enabled"
        assert aff_to_act.all(), "Affordance → Action should be enabled"
    
    def test_zero_length_sequences(self):
        """Test handling of zero-length sequences."""
        batch_size, device = 1, torch.device("cpu")
        
        # Test with zero affordance length
        mask = build_blockwise_mask(5, 0, 4, batch_size, device)
        assert mask.shape == (1, 9, 9)
        
        # Test with zero action length
        mask = build_blockwise_mask(5, 3, 0, batch_size, device)
        assert mask.shape == (1, 8, 8)
        
        # VLM and affordance should still work
        vlm_block = mask[0, :5, :5]
        assert vlm_block.all()
        
        aff_to_vlm = mask[0, 5:8, :5]
        assert aff_to_vlm.all()
    
    def test_single_token_sequences(self):
        """Test with single token sequences."""
        batch_size, device = 1, torch.device("cpu")
        
        mask = build_blockwise_mask(1, 1, 1, batch_size, device)
        assert mask.shape == (1, 3, 3)
        
        # Each single token should attend to itself
        assert mask[0, 0, 0]  # VLM
        assert mask[0, 1, 1]  # Affordance
        assert mask[0, 2, 2]  # Action
        
        # Cross-attention rules should still apply
        assert mask[0, 1, 0]  # Affordance → VLM
        assert mask[0, 2, 0]  # Action → VLM
        assert not mask[0, 0, 1]  # VLM ↛ Affordance
        assert not mask[0, 0, 2]  # VLM ↛ Action
    
    def test_device_consistency(self):
        """Test that output mask is on correct device."""
        if torch.cuda.is_available():
            devices = [torch.device("cpu"), torch.device("cuda")]
        else:
            devices = [torch.device("cpu")]
        
        for device in devices:
            mask = build_blockwise_mask(5, 3, 4, 2, device)
            assert mask.device == device
    
    def test_batch_size_consistency(self):
        """Test that mask is consistent across batch dimension."""
        batch_size, device = 3, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        mask = build_blockwise_mask(vlm_len, aff_len, act_len, batch_size, device)
        
        # All batches should have identical masks
        for i in range(1, batch_size):
            assert torch.equal(mask[0], mask[i]), f"Batch {i} differs from batch 0"


class TestErrorHandling:
    """Test error handling in attention mask construction."""
    
    def test_negative_lengths(self):
        """Test that negative lengths raise errors."""
        with pytest.raises(ValueError, match="non-negative"):
            build_blockwise_mask(-1, 5, 4, 2, torch.device("cpu"))
        
        with pytest.raises(ValueError, match="non-negative"):
            build_blockwise_mask(5, -1, 4, 2, torch.device("cpu"))
        
        with pytest.raises(ValueError, match="non-negative"):
            build_blockwise_mask(5, 3, -1, 2, torch.device("cpu"))
    
    def test_zero_batch_size(self):
        """Test that zero batch size raises error."""
        with pytest.raises(ValueError, match="positive"):
            build_blockwise_mask(5, 3, 4, 0, torch.device("cpu"))
    
    def test_negative_batch_size(self):
        """Test that negative batch size raises error."""
        with pytest.raises(ValueError, match="positive"):
            build_blockwise_mask(5, 3, 4, -1, torch.device("cpu"))
    
    def test_all_zero_lengths(self):
        """Test that all zero lengths raise error."""
        with pytest.raises(ValueError, match="Total sequence length cannot be zero"):
            build_blockwise_mask(0, 0, 0, 2, torch.device("cpu"))


class TestAlternativeInterfaces:
    """Test alternative interfaces and convenience functions."""
    
    def test_prefix_middle_suffix_interface(self):
        """Test prefix-middle-suffix naming interface."""
        batch_size, device = 2, torch.device("cpu")
        prefix_len, middle_len, suffix_len = 10, 5, 8
        
        # Both interfaces should produce identical results
        mask1 = build_blockwise_mask(
            prefix_len, middle_len, suffix_len, batch_size, device,
            allow_action_see_affordance=True,
            allow_affordance_see_action=False
        )
        
        mask2 = build_prefix_middle_suffix_mask(
            prefix_len, middle_len, suffix_len, batch_size, device,
            allow_suffix_see_middle=True,
            allow_middle_see_suffix=False
        )
        
        assert torch.equal(mask1, mask2), "Both interfaces should produce identical results"
    
    def test_convenience_functions(self):
        """Test convenience functions for common configurations."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 5, 3, 4
        
        # Test isolated experts
        isolated_mask = create_isolated_experts_mask(vlm_len, aff_len, act_len, batch_size, device)
        act_to_aff = isolated_mask[0, vlm_len+aff_len:, vlm_len:vlm_len+aff_len]
        aff_to_act = isolated_mask[0, vlm_len:vlm_len+aff_len, vlm_len+aff_len:]
        assert not act_to_aff.any() and not aff_to_act.any()
        
        # Test action-to-affordance
        act2aff_mask = create_action_to_affordance_mask(vlm_len, aff_len, act_len, batch_size, device)
        act_to_aff = act2aff_mask[0, vlm_len+aff_len:, vlm_len:vlm_len+aff_len]
        aff_to_act = act2aff_mask[0, vlm_len:vlm_len+aff_len, vlm_len+aff_len:]
        assert act_to_aff.all() and not aff_to_act.any()
        
        # Test bidirectional
        bidirectional_mask = create_bidirectional_experts_mask(vlm_len, aff_len, act_len, batch_size, device)
        act_to_aff = bidirectional_mask[0, vlm_len+aff_len:, vlm_len:vlm_len+aff_len]
        aff_to_act = bidirectional_mask[0, vlm_len:vlm_len+aff_len, vlm_len+aff_len:]
        assert act_to_aff.all() and aff_to_act.all()


class TestPaddingMaskApplication:
    """Test padding mask application functionality."""
    
    def test_apply_padding_mask(self):
        """Test applying padding mask to attention mask."""
        batch_size, device = 2, torch.device("cpu")
        seq_len = 10
        
        # Create base attention mask (all True)
        attention_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
        
        # Create padding mask (last 3 tokens are padded)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        padding_mask[:, -3:] = False
        
        # Apply padding mask
        combined_mask = apply_padding_mask(attention_mask, padding_mask)
        
        # Check that padded positions cannot attend or be attended to
        assert not combined_mask[:, -3:, :].any(), "Padded tokens should not attend to anything"
        assert not combined_mask[:, :, -3:].any(), "Nothing should attend to padded tokens"
        
        # Check that valid positions still work
        valid_mask = combined_mask[:, :-3, :-3]
        assert valid_mask.all(), "Valid tokens should still attend to each other"
    
    def test_padding_mask_shape_mismatch(self):
        """Test error handling for shape mismatches."""
        attention_mask = torch.ones(2, 10, 10, dtype=torch.bool)
        padding_mask = torch.ones(2, 5, dtype=torch.bool)  # Wrong size
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_padding_mask(attention_mask, padding_mask)


class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_get_attention_block_info(self):
        """Test getting block boundary information."""
        vlm_len, aff_len, act_len = 5, 3, 4
        
        vlm_block, aff_block, act_block = get_attention_block_info(vlm_len, aff_len, act_len)
        
        assert vlm_block == (0, 5)
        assert aff_block == (5, 8)
        assert act_block == (8, 12)
    
    def test_visualize_attention_mask(self):
        """Test attention mask visualization."""
        batch_size, device = 1, torch.device("cpu")
        vlm_len, aff_len, act_len = 3, 2, 2
        
        mask = build_blockwise_mask(vlm_len, aff_len, act_len, batch_size, device)
        
        visualization = visualize_attention_mask(mask, vlm_len, aff_len, act_len, batch_idx=0)
        
        # Check that visualization contains expected elements
        assert "Attention Mask Visualization" in visualization
        assert "VLM:" in visualization
        assert "Affordance:" in visualization
        assert "Action:" in visualization
        assert "Legend:" in visualization
    
    def test_create_causal_mask(self):
        """Test causal mask creation."""
        seq_len, batch_size, device = 5, 2, torch.device("cpu")
        
        causal_mask = create_causal_mask(seq_len, batch_size, device)
        
        assert causal_mask.shape == (batch_size, seq_len, seq_len)
        
        # Check lower triangular property
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert causal_mask[0, i, j], f"Position ({i},{j}) should be True"
                else:
                    assert not causal_mask[0, i, j], f"Position ({i},{j}) should be False"
    
    def test_combine_masks(self):
        """Test mask combination operations."""
        shape = (2, 4, 4)
        mask1 = torch.zeros(*shape, dtype=torch.bool)
        mask2 = torch.ones(*shape, dtype=torch.bool)
        
        # Test AND operation
        and_result = combine_masks(mask1, mask2, "and")
        assert not and_result.any(), "AND with all-False should be all-False"
        
        # Test OR operation
        or_result = combine_masks(mask1, mask2, "or")
        assert or_result.all(), "OR with all-True should be all-True"
        
        # Test XOR operation
        xor_result = combine_masks(mask1, mask2, "xor")
        assert xor_result.all(), "XOR of all-False and all-True should be all-True"
    
    def test_validate_attention_mask(self):
        """Test attention mask validation."""
        # Valid mask
        valid_mask = torch.ones(2, 5, 5, dtype=torch.bool)
        assert validate_attention_mask(valid_mask)
        
        # Invalid dtype
        with pytest.raises(ValueError, match="bool type"):
            validate_attention_mask(torch.ones(2, 5, 5, dtype=torch.float32))
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="3D"):
            validate_attention_mask(torch.ones(5, 5, dtype=torch.bool))
        
        # Non-square
        with pytest.raises(ValueError, match="square"):
            validate_attention_mask(torch.ones(2, 5, 4, dtype=torch.bool))
        
        # Shape mismatch
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_attention_mask(valid_mask, expected_shape=(2, 4, 4))


if __name__ == "__main__":
    # Run basic tests to verify functionality
    print("Running basic attention mask tests...")
    
    # Test basic mask construction
    device = torch.device("cpu")
    mask = build_blockwise_mask(5, 3, 4, 2, device, allow_action_see_affordance=True)
    print(f"✓ Basic mask construction: {mask.shape}")
    
    # Test visualization
    viz = visualize_attention_mask(mask, 5, 3, 4, 0)
    print("✓ Mask visualization:")
    print(viz)
    
    print("\nAll basic tests passed! Run with pytest for comprehensive testing.")