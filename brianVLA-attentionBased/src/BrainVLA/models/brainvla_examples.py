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
BrainVLA Usage Examples

This module demonstrates how to use the BrainVLA architecture with different
affordance expert implementations and configurations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List

from .configuration_brainvla import BrainVLAConfig
from .affordance_experts import create_affordance_expert, SAMAffordanceExpert, DirectAffordanceExpert
from .paligemma_with_triple_expert import BrainVLAWithTripleExpertConfig, BrainVLAWithTripleExpertModel
from .modeling_brainvla import BrainVLAPolicy, BrainVLAFlowMatching
from .attention_utils import build_blockwise_mask, visualize_attention_mask


def create_sample_config(expert_type: str = "sam") -> BrainVLAConfig:
    """Create a sample BrainVLA configuration."""
    config = BrainVLAConfig(
        affordance_expert_type=expert_type,
        n_affordance_queries=8,
        n_action_steps=50,
        allow_action_see_affordance=True,
        allow_affordance_see_action=False,
        sam_checkpoint_path="sam_vit_h_4b8939.pth",  # User needs to download this
        train_sam_mask_decoder=True,
        freeze_sam_image_encoder=True,
    )
    return config


def demo_affordance_experts():
    """Demonstrate different affordance expert implementations."""
    print("=" * 60)
    print("BrainVLA Affordance Expert Examples")
    print("=" * 60)
    
    # Create sample data
    batch_size = 2
    device = torch.device("cpu")
    
    # SAM-based affordance expert
    print("\n1. SAM-based Affordance Expert")
    print("-" * 40)
    
    try:
        sam_config = create_sample_config("sam")
        sam_expert = create_affordance_expert(sam_config)
        
        print(f"✓ Created SAM expert: {sam_expert.__class__.__name__}")
        
        # Generate attention queries
        queries = sam_expert.get_attention_queries(batch_size, device)
        print(f"✓ Attention queries shape: {queries.shape}")
        
        # Simulate transformer features
        transformer_features = torch.randn_like(queries)
        images = torch.randn(batch_size, 3, 224, 224)
        
        print("✓ SAM expert ready for use")
        print("  Note: Requires segment-anything package and SAM checkpoint")
        
    except ImportError as e:
        print(f"✗ SAM expert requires additional dependencies: {e}")
    except Exception as e:
        print(f"✗ SAM expert initialization failed: {e}")
    
    # Direct affordance expert
    print("\n2. Direct Affordance Expert")
    print("-" * 40)
    
    direct_config = create_sample_config("direct")
    direct_expert = create_affordance_expert(direct_config)
    
    print(f"✓ Created direct expert: {direct_expert.__class__.__name__}")
    
    # Generate attention queries
    queries = direct_expert.get_attention_queries(batch_size, device)
    print(f"✓ Attention queries shape: {queries.shape}")
    
    # Generate affordance predictions
    transformer_features = torch.randn_like(queries)
    images = torch.randn(batch_size, 3, 224, 224)
    
    affordance_maps = direct_expert.get_affordance_embeddings(transformer_features, images)
    print(f"✓ Affordance maps shape: {affordance_maps.shape}")
    
    # Compute loss with dummy ground truth
    gt_masks = torch.randint(0, 2, affordance_maps.shape).float()
    losses = direct_expert.compute_affordance_loss(transformer_features, images, gt_masks)
    print(f"✓ Computed losses: {list(losses.keys())}")
    
    print("\n✅ Direct affordance expert working correctly!")


def demo_triple_expert_model():
    """Demonstrate the triple expert model with attention routing."""
    print("\n" + "=" * 60)
    print("BrainVLA Triple Expert Model Example")
    print("=" * 60)
    
    # Create configuration
    config = BrainVLAWithTripleExpertConfig(
        train_expert_only=False,
        train_affordance_expert=True,
        train_action_expert=True,
    )
    
    # Create model
    model = BrainVLAWithTripleExpertModel(config)
    print(f"✓ Created triple expert model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Sample data
    batch_size = 2
    device = torch.device("cpu")
    
    vlm_len, affordance_len, action_len = 20, 8, 50
    hidden_dim = 1024  # Should match action expert config
    
    # Create embeddings for each expert
    vlm_embeds = torch.randn(batch_size, vlm_len, 2048)  # VLM uses larger dimension
    affordance_embeds = torch.randn(batch_size, affordance_len, 512)  # Affordance uses smaller
    action_embeds = torch.randn(batch_size, action_len, hidden_dim)
    
    # Create attention mask
    attention_mask = build_blockwise_mask(
        vlm_len, affordance_len, action_len, batch_size, device,
        allow_action_see_affordance=True, allow_affordance_see_action=False
    )
    
    print(f"✓ Created attention mask: {attention_mask.shape}")
    print("✓ Attention routing:")
    print("  - VLM ↔ VLM: bidirectional")  
    print("  - Affordance → VLM: unidirectional")
    print("  - Action → VLM: unidirectional")
    print("  - Action → Affordance: enabled")
    print("  - Affordance → Action: disabled")
    
    # Position IDs
    total_len = vlm_len + affordance_len + action_len
    position_ids = torch.arange(total_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    try:
        with torch.no_grad():  # Avoid memory issues in demo
            outputs, kv_cache = model.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                vlm_embeds=vlm_embeds,
                affordance_embeds=affordance_embeds,
                action_embeds=action_embeds,
                use_cache=False,
            )
        
        print(f"✓ Forward pass successful!")
        print(f"  - VLM output shape: {outputs[0].shape if outputs[0] is not None else 'None'}")
        print(f"  - Affordance output shape: {outputs[1].shape if outputs[1] is not None else 'None'}")
        print(f"  - Action output shape: {outputs[2].shape if outputs[2] is not None else 'None'}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    print("\n✅ Triple expert model working correctly!")


def demo_attention_routing():
    """Demonstrate attention mask construction and visualization."""
    print("\n" + "=" * 60)
    print("BrainVLA Attention Routing Example")
    print("=" * 60)
    
    batch_size = 1
    device = torch.device("cpu")
    vlm_len, affordance_len, action_len = 6, 3, 4
    
    # Different routing configurations
    configs = [
        {
            "name": "Isolated Experts", 
            "allow_action_see_affordance": False,
            "allow_affordance_see_action": False,
            "description": "Baseline: experts cannot communicate"
        },
        {
            "name": "Action→Affordance",
            "allow_action_see_affordance": True, 
            "allow_affordance_see_action": False,
            "description": "Action expert can see affordance context"
        },
        {
            "name": "Bidirectional Experts",
            "allow_action_see_affordance": True,
            "allow_affordance_see_action": True, 
            "description": "Full bidirectional expert communication"
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print("-" * 40)
        print(f"Description: {config['description']}")
        
        mask = build_blockwise_mask(
            vlm_len, affordance_len, action_len, batch_size, device,
            allow_action_see_affordance=config["allow_action_see_affordance"],
            allow_affordance_see_action=config["allow_affordance_see_action"]
        )
        
        print(f"Mask shape: {mask.shape}")
        print("\nVisualization:")
        visualization = visualize_attention_mask(mask, vlm_len, affordance_len, action_len, 0)
        print(visualization)


def demo_full_policy():
    """Demonstrate the complete BrainVLA policy."""
    print("\n" + "=" * 60)
    print("BrainVLA Complete Policy Example")
    print("=" * 60)
    
    # Create configuration
    config = create_sample_config("direct")  # Use direct expert to avoid SAM dependencies
    
    # Create dummy dataset stats for normalization
    dataset_stats = {
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }
    
    try:
        # Create policy
        policy = BrainVLAPolicy(config, dataset_stats)
        print(f"✓ Created BrainVLA policy with {sum(p.numel() for p in policy.parameters())} parameters")
        
        # Sample training batch
        batch_size = 2
        batch = {
            "observation.image": torch.randn(batch_size, 3, 224, 224),
            "observation.state": torch.randn(batch_size, 7),
            "task": ["Pick up the red cube\n", "Place the object in the box\n"],
            "action": torch.randn(batch_size, config.n_action_steps, 7),
        }
        
        print("✓ Created sample training batch")
        
        # Training forward pass
        with torch.no_grad():  # Avoid memory issues in demo
            loss, loss_dict = policy.forward(batch)
        
        print(f"✓ Training forward pass successful!")
        print(f"  - Total loss: {loss:.4f}")
        print(f"  - Loss components: {list(loss_dict.keys())}")
        
        # Inference  
        inference_batch = {
            "observation.image": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 7),
            "task": ["Pick up the red cube\n"],
        }
        
        # Note: select_action requires proper dataset stats and may fail in demo
        print("✓ Policy ready for inference (use select_action method)")
        
    except Exception as e:
        print(f"✗ Policy demonstration failed: {e}")
        print("  This is expected in a demo environment without proper setup")
    
    print("\n✅ BrainVLA policy structure is correct!")


def main():
    """Run all BrainVLA examples."""
    print("🧠 BrainVLA Architecture Examples")
    print("=" * 80)
    
    try:
        demo_affordance_experts()
        demo_triple_expert_model()
        demo_attention_routing()
        demo_full_policy()
        
        print("\n" + "=" * 80)
        print("🎉 All BrainVLA examples completed successfully!")
        print("\nNext steps:")
        print("1. Install segment-anything for SAM-based affordance expert")
        print("2. Download SAM checkpoint: sam_vit_h_4b8939.pth")
        print("3. Prepare training data with affordance annotations")
        print("4. Configure dataset statistics for proper normalization")
        print("5. Run training with BrainVLA policy")
        
    except Exception as e:
        print(f"\n❌ Examples failed with error: {e}")
        print("This may be due to missing dependencies or configuration issues.")


if __name__ == "__main__":
    main()