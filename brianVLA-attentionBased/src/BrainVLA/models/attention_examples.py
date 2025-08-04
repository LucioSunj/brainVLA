#!/usr/bin/env python

"""
BrainVLA Attention Mask Examples

This module demonstrates the usage of blockwise attention mask construction
for different expert interaction configurations.
"""

import torch
from attention_utils import (
    build_blockwise_mask,
    visualize_attention_mask,
    create_isolated_experts_mask,
    create_action_to_affordance_mask,
    create_bidirectional_experts_mask,
    apply_padding_mask,
)


def demonstrate_attention_masks():
    """Demonstrate different attention mask configurations."""
    
    print("=" * 70)
    print("BrainVLA Attention Mask Configuration Examples")
    print("=" * 70)
    
    # Common parameters
    device = torch.device("cpu")
    batch_size = 1
    vlm_len, affordance_len, action_len = 6, 4, 5
    total_len = vlm_len + affordance_len + action_len
    
    print(f"\nSequence configuration:")
    print(f"  VLM tokens: {vlm_len} (positions 0-{vlm_len-1})")
    print(f"  Affordance tokens: {affordance_len} (positions {vlm_len}-{vlm_len+affordance_len-1})")
    print(f"  Action tokens: {action_len} (positions {vlm_len+affordance_len}-{total_len-1})")
    print(f"  Total length: {total_len}")
    
    # Configuration 1: Isolated experts (baseline)
    print("\n" + "="*50)
    print("Configuration 1: Isolated Experts (Baseline)")
    print("="*50)
    print("- VLM ↔ VLM: ✓ (bidirectional)")
    print("- Affordance → VLM: ✓")  
    print("- Action → VLM: ✓")
    print("- Affordance ↔ Affordance: ✓ (bidirectional)")
    print("- Action ↔ Action: ✓ (bidirectional)")
    print("- Action ↔ Affordance: ✗ (isolated)")
    
    isolated_mask = create_isolated_experts_mask(
        vlm_len, affordance_len, action_len, batch_size, device
    )
    
    print("\nVisualization:")
    print(visualize_attention_mask(isolated_mask, vlm_len, affordance_len, action_len, 0))
    
    # Configuration 2: Action can see affordance
    print("\n" + "="*50)
    print("Configuration 2: Action → Affordance")
    print("="*50)
    print("- Same as baseline, plus:")
    print("- Action → Affordance: ✓ (action can see affordance predictions)")
    
    action_to_aff_mask = create_action_to_affordance_mask(
        vlm_len, affordance_len, action_len, batch_size, device
    )
    
    print("\nVisualization:")
    print(visualize_attention_mask(action_to_aff_mask, vlm_len, affordance_len, action_len, 0))
    
    # Configuration 3: Bidirectional expert interaction
    print("\n" + "="*50)
    print("Configuration 3: Bidirectional Expert Interaction")
    print("="*50)
    print("- Same as baseline, plus:")
    print("- Action → Affordance: ✓")
    print("- Affordance → Action: ✓ (full communication)")
    
    bidirectional_mask = create_bidirectional_experts_mask(
        vlm_len, affordance_len, action_len, batch_size, device
    )
    
    print("\nVisualization:")
    print(visualize_attention_mask(bidirectional_mask, vlm_len, affordance_len, action_len, 0))
    
    # Configuration 4: Custom configuration with padding
    print("\n" + "="*50)
    print("Configuration 4: Custom with Padding")
    print("="*50)
    print("- Custom expert interaction")
    print("- With sequence padding demonstration")
    
    # Create custom mask with affordance → action only
    custom_mask = build_blockwise_mask(
        vlm_len, affordance_len, action_len, batch_size, device,
        allow_action_see_affordance=False,  # Action cannot see affordance
        allow_affordance_see_action=True,   # But affordance can see action
    )
    
    # Apply padding (last 2 tokens are padded)
    padding_mask = torch.ones(batch_size, total_len, dtype=torch.bool, device=device)
    padding_mask[:, -2:] = False  # Pad last 2 tokens
    
    padded_mask = apply_padding_mask(custom_mask, padding_mask)
    
    print(f"\nPadding applied: last 2 tokens are masked")
    print("Custom attention rules: Affordance → Action (but not reverse)")
    print("\nVisualization (with padding):")
    print(visualize_attention_mask(padded_mask, vlm_len, affordance_len, action_len-2, 0))
    print("Note: Last 2 positions show all zeros due to padding")


def demonstrate_mask_properties():
    """Demonstrate attention mask properties and analysis."""
    
    print("\n" + "="*70)
    print("Attention Mask Properties Analysis")
    print("="*70)
    
    device = torch.device("cpu")
    vlm_len, affordance_len, action_len = 4, 3, 3
    
    # Create different configurations
    configs = [
        ("Isolated", False, False),
        ("Action→Affordance", True, False),
        ("Affordance→Action", False, True),
        ("Bidirectional", True, True),
    ]
    
    for config_name, act_to_aff, aff_to_act in configs:
        mask = build_blockwise_mask(
            vlm_len, affordance_len, action_len, 1, device,
            allow_action_see_affordance=act_to_aff,
            allow_affordance_see_action=aff_to_act
        )
        
        # Analyze mask properties
        total_positions = mask.shape[-1] ** 2
        allowed_connections = mask[0].sum().item()
        density = allowed_connections / total_positions * 100
        
        print(f"\n{config_name} Configuration:")
        print(f"  Total possible connections: {total_positions}")
        print(f"  Allowed connections: {allowed_connections}")
        print(f"  Attention density: {density:.1f}%")
        
        # Expert-specific analysis
        vlm_block = mask[0, :vlm_len, :vlm_len]
        aff_block = mask[0, vlm_len:vlm_len+affordance_len, vlm_len:vlm_len+affordance_len]
        act_block = mask[0, vlm_len+affordance_len:, vlm_len+affordance_len:]
        
        print(f"  VLM internal connections: {vlm_block.sum().item()}/{vlm_len**2}")
        print(f"  Affordance internal connections: {aff_block.sum().item()}/{affordance_len**2}")
        print(f"  Action internal connections: {act_block.sum().item()}/{action_len**2}")
        
        # Cross-expert analysis
        act_to_aff_conn = mask[0, vlm_len+affordance_len:, vlm_len:vlm_len+affordance_len].sum().item()
        aff_to_act_conn = mask[0, vlm_len:vlm_len+affordance_len, vlm_len+affordance_len:].sum().item()
        
        print(f"  Action→Affordance connections: {act_to_aff_conn}/{action_len*affordance_len}")
        print(f"  Affordance→Action connections: {aff_to_act_conn}/{affordance_len*action_len}")


def demonstrate_edge_cases():
    """Demonstrate edge cases and special configurations."""
    
    print("\n" + "="*70)
    print("Edge Cases and Special Configurations")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Edge case 1: Single token per expert
    print("\nEdge Case 1: Single Token Per Expert")
    print("-" * 40)
    single_mask = build_blockwise_mask(1, 1, 1, 1, device, True, True)
    print("Sequence: [V][A][C] with bidirectional expert interaction")
    print(visualize_attention_mask(single_mask, 1, 1, 1, 0))
    
    # Edge case 2: No affordance expert
    print("\nEdge Case 2: No Affordance Expert (VLM + Action only)")
    print("-" * 40)
    no_aff_mask = build_blockwise_mask(5, 0, 3, 1, device)
    print("Sequence: [V V V V V][C C C]")
    print(visualize_attention_mask(no_aff_mask, 5, 0, 3, 0))
    
    # Edge case 3: Large sequence with sparse interaction
    print("\nEdge Case 3: Large Sequence Analysis")
    print("-" * 40)
    large_mask = build_blockwise_mask(10, 8, 12, 1, device, False, False)
    total_possible = 30 * 30
    total_allowed = large_mask[0].sum().item()
    sparsity = (1 - total_allowed / total_possible) * 100
    
    print(f"Large sequence (VLM:10, Affordance:8, Action:12)")
    print(f"Total positions: {total_possible}")
    print(f"Allowed connections: {total_allowed}")
    print(f"Sparsity: {sparsity:.1f}% (attention is sparse)")
    
    # Compare computation savings
    dense_flops = total_possible  # All-to-all attention
    sparse_flops = total_allowed  # Sparse attention
    savings = (1 - sparse_flops / dense_flops) * 100
    
    print(f"Computational savings: {savings:.1f}% reduction in attention operations")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_attention_masks()
    demonstrate_mask_properties() 
    demonstrate_edge_cases()
    
    print("\n" + "="*70)
    print("All demonstrations completed successfully!")
    print("="*70)