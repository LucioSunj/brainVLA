from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoConfig,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import GemmaForCausalLM

class LatentEncoder(ABC):
    """
    Abstract base class for latent encoders.
    It receives latent features from PaliGemma and outputs a unified
    hidden_states tensor for the decoders.
    """

    @abstractmethod
    def encode(self, latent_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encodes a dictionary of latent features into a single tensor.
        
        Args:
            latent_features: A dictionary containing feature tensors, e.g.,
                             {"image_features": ..., "instruction_features": ...}.
                             
        Returns:
            A tensor of hidden states.
        """
        ...


class SAMDecoder(Protocol):
    """Protocol for SAM decoders.
    
    Any decoder that follows this protocol can be registered with
    PaliGemmaWithAffordAction2ExpertsModel.
    """
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        image_embeddings: torch.Tensor,
        original_size: Tuple[int, int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            image_embeddings: Original image embeddings from SAM encoder
            original_size: Original image size for mask rescaling (optional)
            
        Returns:
            Dict containing predicted masks and other outputs
        """
        ...


class SamEncoder(LatentEncoder, nn.Module):
    """Encoder that combines image and instruction features for SAM."""
    
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        # Projection layers for feature fusion
        self.image_projection = nn.Linear(
            config.paligemma_config.vision_config.hidden_size,
            config.fusion_dim
        )
        self.instruction_projection = nn.Linear(
            config.paligemma_config.text_config.hidden_size,
            config.fusion_dim
        )
        
        # Feature fusion method
        self.fusion_method = config.feature_fusion
        
        if self.fusion_method == "weighted_sum":
            self.fusion_weights = nn.Parameter(torch.ones(2))
            
        elif self.fusion_method == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.fusion_dim,
                num_heads=8,
                batch_first=True
            )
    
    def encode(
        self,
        latent_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            latent_features: A dictionary containing:
                - "image_features": Tensor of shape (batch_size, img_seq_len, img_hidden_size)
                - "instruction_features": Tensor of shape (batch_size, instr_seq_len, instr_hidden_size)
            
        Returns:
            Fused features of shape (batch_size, total_seq_len, fusion_dim)
        """
        image_features = latent_features["image_features"]
        instruction_features = latent_features["instruction_features"]

        # Project features to the same dimension
        image_projected = self.image_projection(image_features)
        instr_projected = self.instruction_projection(instruction_features)
        
        # Debug info for shapes
        if self.config.debug_mode:
            print(f"Image features shape: {image_features.shape}")
            print(f"Instruction features shape: {instruction_features.shape}")
            print(f"Projected image features shape: {image_projected.shape}")
            print(f"Projected instruction features shape: {instr_projected.shape}")
        
        # Feature fusion based on configured method
        if self.fusion_method == "concat":
            # Simple concatenation along sequence dimension
            fused_features = torch.cat([image_projected, instr_projected], dim=1)
            
        elif self.fusion_method == "weighted_sum":
            # First ensure sequences are the same length
            if image_projected.shape[1] != instr_projected.shape[1]:
                raise ValueError(
                    f"For weighted_sum fusion, sequences must have the same length, "
                    f"got {image_projected.shape[1]} and {instr_projected.shape[1]}"
                )
            
            # Normalize weights with softmax
            norm_weights = F.softmax(self.fusion_weights, dim=0)
            fused_features = (
                norm_weights[0] * image_projected + 
                norm_weights[1] * instr_projected
            )
            
        elif self.fusion_method == "cross_attention":
            # Use cross-attention to fuse features
            attn_output, _ = self.cross_attention(
                query=instr_projected,
                key=image_projected,
                value=image_projected
            )
            fused_features = torch.cat([image_projected, attn_output], dim=1)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
        return fused_features


ENCODER_REGISTRY = {
    "sam": SamEncoder,
    # Future encoders can be registered here
}


def build_encoder(config: "PaliGemmaWithAffordAction2ExpertsConfig") -> "LatentEncoder":
    """Factory function to build an encoder based on the config."""
    name = config.encoder_type
    EncoderCls = ENCODER_REGISTRY.get(name.lower())
    if EncoderCls is None:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODER_REGISTRY.keys())}")
    
    return EncoderCls(config=config, **config.encoder_args)


class DefaultSAMDecoder(nn.Module):
    """Basic implementation of a SAM decoder for affordance prediction."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Simple MLP for mask prediction
        self.mask_decoder = nn.Sequential(
            nn.Linear(config.fusion_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.mask_output_size)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor, 
        image_embeddings: torch.Tensor,
        original_size: Tuple[int, int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            image_embeddings: Original image embeddings from SAM encoder
            original_size: Original image size for mask rescaling
            
        Returns:
            Dict containing predicted masks
        """
        # For now, we'll just use a pooled representation of hidden states
        pooled_features = hidden_states.mean(dim=1)
        
        # Predict raw mask logits
        mask_logits = self.mask_decoder(pooled_features)
        
        # Reshape to expected mask dimensions (batch, 1, H, W)
        batch_size = hidden_states.shape[0]
        mask_size = int(self.config.mask_output_size ** 0.5)
        masks = mask_logits.reshape(batch_size, 1, mask_size, mask_size)
        
        # Note: In a real implementation, we would apply the proper SAM mask decoder
        # and rescale to original image size
        
        return {
            "masks": masks,
            "iou_scores": torch.ones(batch_size, 1, device=masks.device)  # Placeholder
        }


class PaliGemmaWithAffordAction2ExpertsConfig(PretrainedConfig):
    """Configuration class for PaliGemmaWithAffordAction2ExpertsModel."""
    
    model_type = "PaliGemmaWithAffordAction2ExpertsModel"
    sub_configs = {"paligemma_config": AutoConfig}
    
    def __init__(
        self,
        paligemma_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        affordance_layer_index: int = -1,  # -1 means use the last layer
        feature_fusion: str = "concat",  # Options: concat, weighted_sum, cross_attention
        fusion_dim: int = 768,
        decoder_hidden_dim: int = 512,
        mask_output_size: int = 256,  # 16x16 masks
        debug_mode: bool = False,
        encoder_type: str = "sam",
        encoder_args: Dict = None,
        action_dim: int = 14,            # ===== 新增: 机器人动作维度 =====
        gemma_expert_config: dict | None = None,  # ===== 新增: Gemma Expert Config =====
        train_expert_only: bool = True,  # ===== 新增: 是否仅训练Expert层 =====
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.affordance_layer_index = affordance_layer_index
        self.feature_fusion = feature_fusion
        self.fusion_dim = fusion_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.mask_output_size = mask_output_size
        self.debug_mode = debug_mode
        self.encoder_type = encoder_type
        self.encoder_args = encoder_args if encoder_args is not None else {}
        self.action_dim = action_dim
        self.train_expert_only = train_expert_only
        
        if paligemma_config is None:
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
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
                    "num_attention_heads": 8,
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
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(paligemma_config, dict):
            if "model_type" not in paligemma_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        # ===== 新增: 处理 Gemma Expert Config =====
        if gemma_expert_config is None:
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
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
        elif isinstance(gemma_expert_config, dict):
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"
            cfg_cls_expert = CONFIG_MAPPING[gemma_expert_config["model_type"]]
            self.gemma_expert_config = cfg_cls_expert(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        
        if self.feature_fusion not in ["concat", "weighted_sum", "cross_attention"]:
            raise ValueError(
                f"Invalid feature fusion method: {self.feature_fusion}. "
                f"Expected one of: concat, weighted_sum, cross_attention."
            )
        
        # ===== 新增: 校验 action_dim =====
        if self.action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")


class PaliGemmaWithAffordAction2ExpertsModel(PreTrainedModel):
    """
    Model that integrates PaliGemma with SAM-style affordance prediction using
    a pluggable latent encoder mechanism.
    
    Usage Example:
    
    # To use the default SamEncoder
    config = PaliGemmaWithAffordAction2ExpertsConfig(encoder_type="sam")
    model = PaliGemmaWithAffordAction2ExpertsModel(config)
    
    # To use a future 'NewEncoder', you would first register it:
    # ENCODER_REGISTRY['new_encoder'] = NewEncoder
    #
    # Then configure the model to use it:
    # config.encoder_type = "new_encoder"
    # config.encoder_args = {"arg1": "value1"}
    # model = PaliGemmaWithAffordAction2ExpertsModel(config)
    """
    
    config_class = PaliGemmaWithAffordAction2ExpertsConfig
    
    def __init__(self, config: PaliGemmaWithAffordAction2ExpertsConfig):
        super().__init__(config=config)
        self.config = config
        
        # Initialize PaliGemma
        self.paligemma = PaliGemmaForConditionalGeneration(
            config=config.paligemma_config
        )
        
        # 这里使用Gemma Experts作为动作预测模型
        # ===== 新增: 初始化 Gemma Expert =====
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # 移除不必要的组件
        self.gemma_expert.model.embed_tokens = None
        self.gemma_expert.lm_head = None
        
        # Action prediction head
        self.action_head = nn.Linear(
            config.gemma_expert_config.hidden_size,
            config.action_dim,
            bias=True,
        )
        
        # Initialize the latent encoder using the factory
        self.encoder = build_encoder(config)
        
        # Initialize decoders registry
        self.decoders = nn.ModuleDict({
            "default": DefaultSAMDecoder(config)
        })
        
        # Convert relevant components to bfloat16
        self.to_bfloat16()
        
        # Set gradients based on config
        self.set_requires_grad()
    
    def set_requires_grad(self):
        """Configure which parts of the model should be trained."""
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False
        # ===== 新增: expert 训练策略 =====
        if self.config.train_expert_only:
            self.paligemma.eval()
            for p in self.paligemma.parameters():
                p.requires_grad = False
    
    def train(self, mode: bool = True):
        """Override train mode to keep vision encoder frozen if configured."""
        super().train(mode)
        
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
        # ===== 新增: 如果只训练 expert，保持 paligemma 冻结 =====
        if self.config.train_expert_only:
            self.paligemma.eval()
    
    def to_bfloat16(self):
        """Convert model components to bfloat16 for efficiency."""
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)
        
        params_to_change_dtype = [
            "language_model.model.layers",
            "vision_tower",
            "multi_modal",
            "gemma_expert.model.layers",   # ===== 新增 dtype 处理 =====
        ]
        
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)
    
    def embed_image(self, image: torch.Tensor):
        """Extract image features using PaliGemma's vision tower."""
        return self.paligemma.get_image_features(image)
    
    def embed_language_tokens(self, tokens: torch.Tensor):
        """Extract language features from tokens using PaliGemma's embedding layer."""
        return self.paligemma.language_model.model.embed_tokens(tokens)
    
    def register_decoder(self, name: str, decoder: nn.Module):
        """Register a new decoder module."""
        if name in self.decoders:
            raise ValueError(f"Decoder with name '{name}' already exists.")
        
        self.decoders[name] = decoder
        return self
    
    def extract_layer_features(self, tokens, images=None, layer_idx=None):
        """Extract features from a specific layer of PaliGemma.
        
        Args:
            tokens: Input token IDs
            images: Optional input images
            layer_idx: Which layer to extract features from (-1 means last layer)
            
        Returns:
            Hidden states from the specified layer
        """
        idx = self.config.affordance_layer_index if layer_idx is None else layer_idx
        
        # Get outputs from all layers
        outputs = self.paligemma(
            input_ids=tokens, 
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract the specified layer's hidden states
        # Note: Adjust index handling based on the actual structure of outputs
        if idx < 0:
            idx = len(outputs.hidden_states) + idx
        
        hidden_states = outputs.hidden_states[idx]
        return hidden_states
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_names: Optional[List[str]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Input images
            input_ids: Input token IDs
            attention_mask: Attention mask for input tokens
            decoder_names: Which decoders to run (None means all)
            original_size: Original image size for mask rescaling
            
        Returns:
            Dict mapping decoder names to their outputs
        """
        if self.config.debug_mode:
            print(f"Forward input shapes - pixel_values: {pixel_values.shape if pixel_values is not None else None}, "
                  f"input_ids: {input_ids.shape if input_ids is not None else None}")
        
        # 1. Extract features from PaliGemma
        image_features = self.embed_image(pixel_values)
        instruction_features = self.embed_language_tokens(input_ids)
        
        latent_features = {
            "image_features": image_features,
            "instruction_features": instruction_features,
        }
        
        # 2. Process through the configured latent encoder
        hidden_states = self.encoder.encode(latent_features)
        
        # ===== 新增: Gemma Expert 推理得到动作 =====
        bs, seq_len, _ = hidden_states.shape
        attn_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=hidden_states.device)
        expert_out = self.gemma_expert.model(
            inputs_embeds=hidden_states,
            attention_mask=attn_mask,
            return_dict=True,
        ).last_hidden_state  # (bs, seq_len, hidden)
        pooled_expert = expert_out.mean(dim=1)  # simple mean pooling
        action_pred = self.action_head(pooled_expert)  # (bs, action_dim)
        
        # 3. Run each decoder on the hidden states
        results_aff = {}
        decoders_to_run = decoder_names or self.decoders.keys()
        
        for name in decoders_to_run:
            if name not in self.decoders:
                raise ValueError(f"Decoder '{name}' not found. Available decoders: {list(self.decoders.keys())}")
            
            decoder = self.decoders[name]
            results_aff[name] = decoder(
                hidden_states=hidden_states,
                image_embeddings=image_features,
                original_size=original_size,
                **kwargs
            )
            
        return {
            "affordance": results_aff,
            "action": action_pred,
        } 

def test():
    """测试PaliGemmaWithAffordAction2ExpertsModel模型"""
    import os
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 模型配置
    config = PaliGemmaWithAffordAction2ExpertsConfig(
        debug_mode=True,  # 启用调试模式查看特征形状
        feature_fusion="cross_attention",  # 使用交叉注意力进行特征融合
        fusion_dim=768,  # 融合维度
        decoder_hidden_dim=512,  # 解码器隐藏维度
        mask_output_size=256,  # 输出掩码大小
        action_dim=14,  # 动作维度
    )
    
    # 初始化模型
    model = PaliGemmaWithAffordAction2ExpertsModel(config)
    model.to(device)
    model.eval()
    
    # 加载分词器 - 使用PaliGemma的分词器
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-mix")
    
    # 读取测试图片 
    # 假设图片在当前工作目录或指定路径
    image_path = "test_image.jpg"  # 请替换为实际图片路径
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片未找到: {image_path}")
        
    image = Image.open(image_path).convert("RGB")
    
    # 图片处理 - 调整为模型所需尺寸
    from torchvision import transforms
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pixel_values = preprocess(image).unsqueeze(0).to(device)  # 添加batch维度
    
    # 文本指令
    instruction = "请识别图片中可以抓取的物体"
    
    # 对文本指令进行编码
    inputs = tokenizer(instruction, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # 模型推理
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values, 
            input_ids=input_ids, 
            original_size=image.size,  # 提供原始图像尺寸用于掩码缩放
        )
    
    # 提取结果
    affordance_masks = outputs["affordance"]["default"]["masks"]  # 形状为(batch, 1, H, W)
    action_pred = outputs["action"]  # 形状为(batch, action_dim)
    
    # 将掩码从tensor转为numpy以便可视化
    mask = affordance_masks[0, 0].cpu().numpy()  # 第一个batch的第一个掩码
    
    # 输出动作预测
    print("预测动作向量:", action_pred[0].cpu().numpy())
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image))
    plt.title("原始图像")
    plt.axis('off')
    
    # 显示掩码预测结果
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='viridis')
    plt.title("可供性掩码预测")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.show()
    
    print("测试完成，结果已保存为prediction_result.png")

if __name__ == "__main__":
    test()