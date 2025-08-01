目标：
在 π0 风格的 unified transformer 里插入一个 SAM-style affordance 预测 expert。这个 affordance expert 与 Paligemma（VLM）共享 self-attention，但通过 blockwise attention mask 只在允许的方向上读取上游信息（即它可以 attend 到 VLM / state，但 VLM 不能反向 attend 它）。affordance expert 的输出经过一个类似 SAM mask decoder 的 head 生成 spatial affordance masks。

一、核心思想（沿用 π0 的结构风格）
1. 把输入组织成多个 block：
   - VLM block：Paligemma（或其它 vision-language encoder）输出的 token-level 表征（包括 segmentation token / affordance hint token）。
   - State block（可选）：机器人 proprioceptive state / 历史 context。
   - Affordance expert queries block：一组 learnable query tokens（类似 DETR 的 queries / π0 中 expert suffix chunk），代表要预测的 affordance latent。

2. Attention routing（blockwise attention mask）：
   - VLM block 之间全双向 attention（如预训练时一样）。
   - State block 可以 attend 到 VLM block（也可选是否反向）。
   - Affordance queries 可以 attend 到 VLM block 和 state block（获取上下文），但前面的 blocks **不能** attend affordance queries（保持因果/模块分离）。即 affordance expert 读取上游信息，但不反向污染。

3. Unified transformer：所有 block 串成一个 sequence 送入共享 self-attention transformer。可以用 expert-style adapters/gating 在 transformer 某些层对 affordance queries 用专门的小投影/参数，但信息流仍是 shared attention。

4. Mask decoding（SAM-style）：
   - 从 transformer 输出中抽取 affordance query 对应的 hidden states（affordance_hidden）。
   - 用一个 affordance mask decoder 把这些 latent 映射为 spatial mask。这个 decoder 可以是：
     a. 轻量 SAM-like prompt-to-mask：把 affordance_hidden 当作 prompt embeddings，结合一个 image spatial backbone（可保留 ViT-style spatial feature）通过类似 SAM 的 mask decoder 生成 mask。
     b. 自定义 spatial decoder：让 affordance_hidden cross-attend image feature map，然后 upsample 得到 mask（例如用 cross-attention + 逐步上采样的 U-Net-ish head）。

二、模块接口与结构（建议类 / 组件）

class UnifiedPi0SamAffordanceModel(nn.Module):
    def __init__(self, config):
        # 1. VLM encoder（Paligemma-like），输出 vlm_tokens: (B, N_vlm, D)
        # 2. Optional state encoder -> state_tokens: (B, N_state, D)
        # 3. Learnable affordance queries: (num_queries, D)
        # 4. Transformer with blockwise attention mask logic
        #    - Shared layers; can include per-block adapters if desired
        # 5. Affordance mask decoder: takes affordance_hidden + image spatial features -> masks
        # 6. Optional IoU/quality head per mask

    def build_blockwise_attention_mask(self, vlm_len, state_len, num_affordance):
        # 返回 (total_len, total_len) 的 boolean mask or float mask for transformer
        # 覆盖规则：
        #   - vlm <-> vlm: allowed
        #   - state attends to vlm: allowed (configurable)
        #   - affordance_queries attends to vlm and state: allowed
        #   - vlm / state attending to affordance_queries: disallowed
        #   - affordance_queries mutual attention: allowed or controlled (可选)
        pass

    def forward(self, image, instruction, state=None, gt_affordance_masks=None):
        # 1. vlm_tokens = self.vlm_encoder(image, instruction)  # (B, N_vlm, D)
        # 2. state_tokens = self.state_encoder(state) if exists  # (B, N_state, D)
        # 3. Affordance queries expand to batch: affordance_queries -> (B, num_queries, D)
        # 4. Concat sequence = [vlm_tokens, state_tokens?, affordance_queries] -> (B, total_len, D)
        # 5. Compute attention mask via build_blockwise_attention_mask
        # 6. Pass through unified transformer with that mask -> hidden: (B, total_len, D)
        # 7. Split out affordance_hidden = hidden[..., affordance_slice]  # (B, num_queries, D)
        # 8. pred_masks, quality = self.mask_decoder(affordance_hidden, image_spatial_feats)
        # 9. Compute losses if training: mask loss (BCE/focal/dice), matching if multiple queries, optionally distillation
        # 10. Return dict with masks, scores, loss components

三、注意/细节

1. Affordance queries 的设计可以是 learnable vectors，初始化后每个 forward 复用（类似 DETR）；也可以在 multi-stage 里用上一轮 mask 生成新的 queries 做 refinement。

2. Mask decoder 选项：
   - 保留一个小的 image spatial backbone（可以是 Paligemma 中的某层 spatial feature 或额外 ViT）来给 decoder 提供空间参考。
   - Affordance_hidden 作为 query 去 cross-attend 到 spatial feature，后接 upsample（例如 linear + pixel shuffle 或 transpose conv），输出每个 query 的 mask（B, num_queries, H, W）。

3. Loss 设计：
   - 对每个 query-mask pair 做 bipartite matching（如果 ground truth 有多 mask），然后 mask supervision loss：BCE + dice + focal 之类。
   - 可选 quality head 预测 IoU，用 L2 loss 监督与实际 IoU 之间差距（参考 SAM）。
   - 如果有 pretrained SAM 作为 teacher，可以对 affordance_hidden 做 feature distillation（如 L2 / cosine 逼近 teacher prompt embedding distribution），缓解 domain shift。
   - 可选 attention regularization：鼓励 affordance queries 在 attention 上更多关注高置信图像区域（通过 auxiliary loss 监督其 attention weights 聚焦）。

4. 训练/推理加速（借鉴 π0）：
   - 前端 block（vlm + state）输出的 key/value 可以缓存（prefix caching），每次只更新 affordance queries part 实现高效重用。
   - Multi-stage refinement：第一轮产生粗 mask，再将其编码为新的 prompt（或更新 affordance queries）进入第二轮细化。

四、伪代码 sketch（简化）

```python
# assume vlm_tokens: (B, N1, D), state_tokens: (B, N2, D), affordance_queries: (num_q, D)
aff_q = self.affordance_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_q, D)
sequence = torch.cat([vlm_tokens, state_tokens, aff_q], dim=1)  # (B, L, D)
attn_mask = self.build_blockwise_attention_mask(N1, N2, num_q)  # (L, L)

hidden = self.transformer(sequence, attention_mask=attn_mask)  # shared self-attention
affordance_hidden = hidden[:, N1 + N2 :, :]  # (B, num_q, D)

pred_masks, quality = self.mask_decoder(affordance_hidden, image_spatial_feats)