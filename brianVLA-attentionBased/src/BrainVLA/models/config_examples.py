#!/usr/bin/env python

"""
BrainVLA Configuration Usage Examples

This module demonstrates how to use BrainVLAConfig for different scenarios.
"""

from configuration_brainvla import (
    BrainVLAConfig,
    create_default_brainvla_config,
    create_sam_affordance_config,
    create_query_to_mask_config,
)


def main():
    """Demonstrate BrainVLA configuration usage."""
    
    print("=" * 60)
    print("BrainVLA Configuration Examples")
    print("=" * 60)
    
    # Example 1: Default configuration
    print("\n1. Default Configuration:")
    print("-" * 30)
    default_config = create_default_brainvla_config()
    print(f"Model type: {default_config.model_type}")
    print(f"Affordance expert: {default_config.affordance_expert_type}")
    print(f"Expert interaction: action→affordance={default_config.allow_action_see_affordance}, "
          f"affordance→action={default_config.allow_affordance_see_action}")
    print(f"Experiment name: {default_config.get_experiment_name()}")
    
    # Example 2: SAM-based affordance configuration
    print("\n2. SAM-based Affordance Configuration:")
    print("-" * 30)
    sam_config = create_sam_affordance_config(
        n_affordance_queries=16,
        allow_action_see_affordance=True,
        sam_checkpoint_path="custom_sam.pth"
    )
    print(f"Affordance queries: {sam_config.n_affordance_queries}")
    print(f"SAM checkpoint: {sam_config.sam_checkpoint_path}")
    print(f"Train SAM decoder: {sam_config.train_sam_mask_decoder}")
    print(f"Experiment name: {sam_config.get_experiment_name()}")
    
    # Example 3: Query-to-mask configuration
    print("\n3. Query-to-Mask Configuration:")
    print("-" * 30)
    q2m_config = create_query_to_mask_config(
        mask_output_size=256,
        allow_affordance_see_action=True,
        affordance_intermediate_dim=1024
    )
    print(f"Mask output size: {q2m_config.mask_output_size}")
    print(f"Intermediate dim: {q2m_config.affordance_intermediate_dim}")
    print(f"Experiment name: {q2m_config.get_experiment_name()}")
    
    # Example 4: Bidirectional expert interaction
    print("\n4. Bidirectional Expert Interaction:")
    print("-" * 30)
    bidirectional_config = BrainVLAConfig(
        affordance_expert_type="sam",
        allow_action_see_affordance=True,
        allow_affordance_see_action=True,
        n_affordance_queries=12,
        hidden_size=1536,
        total_affordance_loss_weight=0.3
    )
    print(f"Hidden size: {bidirectional_config.hidden_size}")
    print(f"Bidirectional: {bidirectional_config.allow_action_see_affordance and bidirectional_config.allow_affordance_see_action}")
    print(f"Loss weight: {bidirectional_config.total_affordance_loss_weight}")
    print(f"Experiment name: {bidirectional_config.get_experiment_name()}")
    
    # Example 5: Configuration merging
    print("\n5. Configuration Merging:")
    print("-" * 30)
    base_config = create_default_brainvla_config()
    override_dict = {
        "n_affordance_queries": 32,
        "optimizer_lr": 1e-4,
        "allow_action_see_affordance": True,
    }
    merged_config = base_config.merge_with(override_dict)
    print(f"Original queries: {base_config.n_affordance_queries}")
    print(f"Merged queries: {merged_config.n_affordance_queries}")
    print(f"Merged LR: {merged_config.optimizer_lr}")
    print(f"Merged experiment name: {merged_config.get_experiment_name()}")
    
    # Example 6: Configuration serialization
    print("\n6. Configuration Serialization:")
    print("-" * 30)
    config_dict = merged_config.to_dict()
    print(f"Config keys: {len(config_dict.keys())}")
    print("First 5 config entries:")
    for i, (key, value) in enumerate(config_dict.items()):
        if i >= 5:
            break
        print(f"  {key}: {value}")
    
    # Reconstruct from dict
    reconstructed_config = BrainVLAConfig.from_dict(config_dict)
    print(f"Reconstructed successfully: {reconstructed_config.model_type == merged_config.model_type}")
    
    print("\n" + "=" * 60)
    print("Configuration examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()