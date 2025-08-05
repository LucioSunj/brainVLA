# BrainVLA - Brain Vision-Language-Action Model

BrainVLA is a three-expert Vision-Language-Action model that implements specialized transformer architectures with shared attention for robotic control tasks. It extends the π0 architecture with affordance prediction capabilities through a unified attention mechanism.

## 🏗️ Architecture Overview

BrainVLA consists of three specialized experts that share attention computation while maintaining independent parameters:

### 1. **VLM Expert (Prefix)**
- **Model**: PaliGemma for vision-language understanding
- **Role**: Processes visual observations and language instructions
- **Parameters**: ~3B parameters with vision encoder and language model
- **Specialization**: Multimodal understanding and grounding

### 2. **Affordance Expert (Middle)**  
- **Model**: Configurable (SAM-based or Direct)
- **Role**: Predicts object affordances from visual scenes
- **Parameters**: 512M parameters (medium-sized for spatial reasoning)
- **Specialization**: Spatial understanding and affordance detection

### 3. **Action Expert (Suffix)**
- **Model**: Gemma for action generation via flow matching
- **Role**: Generates robot actions from multimodal context
- **Parameters**: 1B parameters (full-sized for complex action generation)
- **Specialization**: Action planning and execution

## 🎯 Key Features

### ✅ Expert Specialization
- **Independent Parameters**: Each expert uses separate transformer layers
- **Shared Attention**: Unified attention computation across all experts
- **Blockwise Routing**: Precise control over information flow between experts

### ✅ Pluggable Affordance Experts
- **SAM-based**: Uses Segment Anything Model for high-quality masks
- **Direct**: Simple query-to-mask for fast inference
- **Custom**: Easy to add new affordance implementations

### ✅ Advanced Attention Routing
```python
# Attention Rules:
VLM ↔ VLM: Complete bidirectional attention
Affordance → VLM: Can attend to VLM context  
Action → VLM: Can attend to VLM context
Affordance ↔ Affordance: Internal bidirectional attention
Action ↔ Action: Internal bidirectional attention
Action ↔ Affordance: Configurable cross-expert interaction
```

### ✅ LeRobot Compatibility
- Full compliance with LeRobot's `PreTrainedPolicy` interface
- Seamless integration with LeRobot training and evaluation pipelines
- Support for dataset normalization and statistics

## 📁 Module Structure

```
BrainVLA/models/
├── configuration_brainvla.py      # Main configuration class
├── modeling_brainvla.py           # BrainVLA policy implementation
├── paligemma_with_triple_expert.py # Triple expert shared attention model
├── affordance_experts.py          # Pluggable affordance expert implementations
├── attention_utils.py             # Blockwise attention mask utilities
├── brainvla_examples.py          # Usage examples and demos
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Basic Configuration

```python
from BrainVLA.models import BrainVLAConfig, BrainVLAPolicy

# Create configuration
config = BrainVLAConfig(
    affordance_expert_type="direct",  # or "sam" for SAM-based
    n_affordance_queries=8,
    n_action_steps=50,
    allow_action_see_affordance=True,
    allow_affordance_see_action=False,
)

# Create policy
policy = BrainVLAPolicy(config, dataset_stats)
```

### 2. Affordance Expert Usage

```python
from BrainVLA.models import create_affordance_expert

# Create SAM-based affordance expert
config.affordance_expert_type = "sam"
config.sam_checkpoint_path = "sam_vit_h_4b8939.pth"
sam_expert = create_affordance_expert(config)

# Create direct affordance expert  
config.affordance_expert_type = "direct"
direct_expert = create_affordance_expert(config)

# Get attention queries
queries = expert.get_attention_queries(batch_size, device)

# Generate affordance predictions
affordance_maps = expert.get_affordance_embeddings(transformer_features, images)
```

### 3. Attention Routing

```python
from BrainVLA.models import build_blockwise_mask, visualize_attention_mask

# Create attention mask
mask = build_blockwise_mask(
    vlm_len=20, affordance_len=8, action_len=50,
    batch_size=2, device=device,
    allow_action_see_affordance=True,
    allow_affordance_see_action=False,
)

# Visualize attention pattern
print(visualize_attention_mask(mask, 20, 8, 50, batch_idx=0))
```

### 4. Training

```python
# Training batch
batch = {
    "observation.image": torch.randn(2, 3, 224, 224),
    "observation.state": torch.randn(2, 7),
    "task": ["Pick up red cube\n", "Place in box\n"],
    "action": torch.randn(2, 50, 7),
    "affordance_masks": torch.randn(2, 8, 224, 224),  # Optional
}

# Forward pass
loss, loss_dict = policy.forward(batch)

# Loss components:
# - action_loss: Flow matching loss for action prediction
# - affordance_focal_loss: Focal loss for affordance segmentation  
# - affordance_dice_loss: Dice loss for affordance segmentation
# - total_affordance_loss: Weighted combination of affordance losses
```

### 5. Inference

```python
# Inference batch
batch = {
    "observation.image": torch.randn(1, 3, 224, 224),
    "observation.state": torch.randn(1, 7),
    "task": ["Pick up the red cube\n"],
}

# Select action
action = policy.select_action(batch)
```

## 🔧 Configuration Options

### Expert Configurations
```python
# VLM Expert (PaliGemma)
vlm_config = {
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_hidden_layers": 18,
    # ... vision and text configs
}

# Affordance Expert (Medium-sized)
affordance_expert_config = {
    "hidden_size": 512,
    "num_attention_heads": 8, 
    "num_hidden_layers": 18,
}

# Action Expert (Full-sized)
action_expert_config = {
    "hidden_size": 1024,
    "num_attention_heads": 8,
    "num_hidden_layers": 18,
}
```

### Affordance Expert Types
```python
# SAM-based affordance expert
config.affordance_expert_type = "sam"
config.sam_checkpoint_path = "sam_vit_h_4b8939.pth"
config.train_sam_mask_decoder = True
config.freeze_sam_image_encoder = True

# Direct affordance expert  
config.affordance_expert_type = "direct"
config.mask_output_size = 224
config.affordance_intermediate_dim = 512
```

### Attention Routing
```python
# Control expert interactions
config.allow_action_see_affordance = True   # Action can attend to affordance
config.allow_affordance_see_action = False  # Affordance cannot attend to action
```

### Loss Weights
```python
# Multi-task loss balancing
config.action_flow_loss_weight = 1.0
config.affordance_focal_loss_weight = 0.1
config.affordance_dice_loss_weight = 1.0
config.total_affordance_loss_weight = 0.5
```

## 🧪 Examples and Testing

Run the comprehensive examples:

```bash
cd src/BrainVLA/models
python brainvla_examples.py
```

This will demonstrate:
- Different affordance expert implementations
- Triple expert model with attention routing
- Complete policy usage
- Attention mask visualization

## 📋 Requirements

### Core Dependencies
- `torch >= 2.0`
- `transformers >= 4.40`
- `lerobot` (LeRobot framework)

### Optional Dependencies
- `segment-anything` (for SAM-based affordance expert)
- `matplotlib` (for attention visualization)

### SAM Setup (for SAM-based affordance expert)
```bash
# Install segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download SAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## 🔍 Implementation Details

### Attention Mechanism
BrainVLA uses blockwise attention routing instead of standard causal attention:
- **Precise Control**: Explicit 2D mask construction
- **Expert Isolation**: Configurable expert interactions
- **Efficient Computation**: Shared attention across all experts

### Flow Matching
Action generation uses flow matching instead of diffusion:
- **Faster Inference**: Fewer denoising steps required
- **Stable Training**: More stable than diffusion objectives
- **Flexible Sampling**: Support for different noise schedules

### Multi-task Learning
Simultaneous training on action prediction and affordance segmentation:
- **Shared Representations**: VLM features benefit both tasks
- **Task-specific Experts**: Specialized parameters for each task
- **Balanced Learning**: Configurable loss weights

## 🎯 Use Cases

### Robotics Applications
- **Manipulation**: Object grasping and placement
- **Navigation**: Spatial affordance understanding
- **Assembly**: Complex multi-step tasks

### Research Applications  
- **Expert Specialization**: Study of multi-expert architectures
- **Cross-modal Learning**: Vision-language-action integration
- **Attention Mechanisms**: Blockwise routing research

## 🤝 Contributing

BrainVLA is designed for extensibility:

### Adding New Affordance Experts
```python
from BrainVLA.models import AffordanceExpertInterface

class CustomAffordanceExpert(AffordanceExpertInterface):
    def get_attention_queries(self, batch_size, device, context=None):
        # Your implementation
        pass
    
    def get_affordance_embeddings(self, transformer_features, images):
        # Your implementation  
        pass
    
    def compute_affordance_loss(self, transformer_features, images, gt_masks):
        # Your implementation
        pass
```

### Custom Attention Routing
```python
from BrainVLA.models import build_blockwise_mask

# Create custom attention patterns
custom_mask = build_blockwise_mask(
    vlm_len, affordance_len, action_len, batch_size, device,
    allow_action_see_affordance=your_logic,
    allow_affordance_see_action=your_logic,
)
```

## 📚 References

- [π0: Physical Intelligence's VLA Model](https://physicalintelligence.company/blog/pi0)
- [PaliGemma: Vision-Language Model](https://huggingface.co/google/paligemma-3b-pt-224)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [LeRobot: Robotics Toolkit](https://github.com/huggingface/lerobot)
- [Flow Matching for Action Generation](https://arxiv.org/abs/2310.02710)

## 📄 License

Licensed under the Apache License, Version 2.0. See LICENSE for details.

---

**BrainVLA**: Bringing specialized expert intelligence to robotic control through unified attention mechanisms. 🧠🤖