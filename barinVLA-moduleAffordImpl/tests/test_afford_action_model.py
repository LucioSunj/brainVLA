import torch

# Import the model and config we want to test
from hume.models.paligemma_with_afford_action_2experts import (
    PaliGemmaWithAffordAction2ExpertsConfig,
    PaliGemmaWithAffordAction2ExpertsModel,
)


def smoke_test_forward(encoder_type: str = "sam"):
    """Runs a single forward pass to ensure the model is wired correctly.

    Args:
        encoder_type: Which latent encoder to test (default: "sam").
    """
    # Build a minimal config. For CI / local testing we keep sizes small.
    config = PaliGemmaWithAffordAction2ExpertsConfig(
        encoder_type=encoder_type,
        debug_mode=True,  # Print tensor shapes for visual confirmation
        action_dim=14,    # Example robot DOF
        fusion_dim=64,    # Reduce dims to keep the model light for testing
        decoder_hidden_dim=32,
        mask_output_size=64,  # 8x8 mask for speed
        paligemma_config={
            "model_type": "paligemma",
            "hidden_size": 128,
            "text_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
            },
            "vision_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 14,
                "projection_dim": 128,
            },
            "projection_dim": 128,
        },
        gemma_expert_config={
            "model_type": "gemma",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "head_dim": 32,
        },
    )

    # Instantiate model
    model = PaliGemmaWithAffordAction2ExpertsModel(config)
    model.eval()

    # Create dummy inputs
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 10, (batch_size, 8))  # Small vocab for speed

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, input_ids=input_ids)

    # Simple assertions
    assert "affordance" in outputs and "action" in outputs, "Missing keys in output"
    assert outputs["action"].shape == (batch_size, config.action_dim), "Incorrect action shape"

    # Print shapes for manual inspection
    print("Action output shape:", outputs["action"].shape)
    for name, result in outputs["affordance"].items():
        print(f"Decoder '{name}' mask shape:", result["masks"].shape)
        print(f"Decoder '{name}' iou shape:", result["iou_scores"].shape)


if __name__ == "__main__":
    # Run smoke test with default SAM encoder
    smoke_test_forward("sam") 