"""
增强型Wav2Vec2方言模型

整合层权重聚合方言适配器和语言学驱动发音变异感知层到Wav2Vec2架构中。
支持多任务学习：CTC语音识别 + 方言类型分类。
支持消融实验的四种配置：Full / Adapter-only / Variation-only / Baseline
"""

import torch
import torch.nn as nn
import os
import json
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import warnings

from dialect_modules import DialectFeatureEnhancer


@dataclass
class DialectModelOutput(CausalLMOutput):
    """模型输出，增加dialect_logits字段"""
    dialect_logits: Optional[torch.FloatTensor] = None


class EnhancedWav2Vec2ForDialect(Wav2Vec2ForCTC):
    """
    增强型Wav2Vec2方言识别模型

    架构:
        输入音频 → CNN特征提取器 → Transformer编码器
        → 层权重聚合方言适配器(新增) → 发音变异感知层(新增)
        → CTC输出层 + 方言分类头(新增)

    多任务学习:
        - 主任务: CTC 语音识别
        - 辅助任务: 方言类型分类（共享编码器被迫学习方言判别性特征）

    Args:
        config: Wav2Vec2Config配置对象
        dialect_config: 方言增强模块的配置字典
    """

    def __init__(self, config: Wav2Vec2Config, dialect_config=None):
        super().__init__(config)

        # 默认配置
        default_dialect_config = {
            'use_adapter': True,
            'use_variation_layer': True,
            'bottleneck_size': 64,
            'num_layers': 25,
            'kernel_sizes': [3, 11, 21],
            'num_attention_heads': 8,
            'dropout': 0.1,
            'num_dialects': 7,
            'dialect_loss_weight': 0.1,
            'ctc_temperature': 1.0,
        }

        if dialect_config is not None:
            default_dialect_config.update(dialect_config)

        self.dialect_config = default_dialect_config

        # 方言特征增强模块
        self.dialect_enhancer = DialectFeatureEnhancer(
            hidden_size=config.hidden_size,
            config=self.dialect_config
        )

        # 方言分类头（多任务辅助任务）
        num_dialects = self.dialect_config.get('num_dialects', 7)
        if num_dialects > 0:
            self.dialect_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.GELU(),
                nn.Dropout(self.dialect_config.get('dropout', 0.1)),
                nn.Linear(256, num_dialects)
            )
        else:
            self.dialect_classifier = None

        self.dialect_loss_weight = self.dialect_config.get('dialect_loss_weight', 0.1)
        self.ctc_temperature = self.dialect_config.get('ctc_temperature', 1.0)

        self._init_dialect_weights()

    def _init_dialect_weights(self):
        """初始化方言增强模块的权重（跳过DialectAdapter，因其有专门的近零初始化）"""
        for name, module in self.dialect_enhancer.named_modules():
            # 跳过 DialectAdapter 内部的 Linear 层，保留其 std=1e-3 的近零初始化
            if 'dialect_adapter' in name:
                continue
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def freeze_feature_extractor(self):
        """冻结CNN特征提取器"""
        self.wav2vec2.feature_extractor._freeze_parameters()
        print("✓ 特征提取器已冻结")

    def freeze_base_model(self):
        """冻结基础模型（wav2vec2编码器），保留CTC头、方言增强模块和分类头可训练"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        # 注意：lm_head（CTC输出层）不冻结，否则模型无法学习输出正确的文字
        for param in self.lm_head.parameters():
            param.requires_grad = True
        for param in self.dialect_enhancer.parameters():
            param.requires_grad = True
        if self.dialect_classifier is not None:
            for param in self.dialect_classifier.parameters():
                param.requires_grad = True
        print("✓ 基础模型已冻结，CTC头+方言增强模块+分类头可训练")

    def unfreeze_encoder_layers(self, num_layers=4):
        """解冻最后几层Transformer编码器"""
        total_layers = len(self.wav2vec2.encoder.layers)
        start_layer = max(0, total_layers - num_layers)

        for i in range(start_layer, total_layers):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = True

        print(f"✓ 已解冻最后 {num_layers} 层编码器（第{start_layer}-{total_layers-1}层）")

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        dialect_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        """前向传播（支持多任务：CTC + 方言分类）"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Wav2Vec2编码器（请求所有层输出用于层权重聚合）
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 始终请求各层输出
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # 节省显存：detach 冻结层的 hidden_states，仅保留最后一层在计算图中
        # layer_weights 仍通过 softmax 加权求和获得梯度（对 detach 的常量做加权求和，
        # 梯度 d_loss/d_w_i = detached_h_i，数值不变），不影响层权重学习。
        # 解冻层通过 outputs[0]（最后一层 hidden_states）正常回传梯度。
        if return_dict and outputs.hidden_states is not None:
            hs = outputs.hidden_states
            all_hidden_states = tuple(
                h.detach() if i < len(hs) - 1 else h
                for i, h in enumerate(hs)
            )
        else:
            all_hidden_states = None

        # 2. 方言特征增强（传入全部层用于层权重聚合）
        if attention_mask is not None:
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).long()
            attention_mask_enhanced = torch.zeros(
                hidden_states.shape[:2],
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            for i, length in enumerate(output_lengths):
                attention_mask_enhanced[i, :length] = 1
        else:
            attention_mask_enhanced = None

        hidden_states = self.dialect_enhancer(
            hidden_states, attention_mask_enhanced, all_hidden_states
        )
        # 层权重聚合完成后立即释放 25 层中间输出，回收显存
        del all_hidden_states

        # 3. Dropout + CTC输出
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        # 4. 方言分类辅助任务（时间维度 mean pooling → 句子级表示）
        #    ★ 使用 detach() 切断梯度回传：分类器已稳定收敛(>99%)，
        #      继续回传梯度只会推动encoder学方言区分特征而非ASR特征
        dialect_logits = None
        dialect_loss = None
        if self.dialect_classifier is not None:
            if attention_mask_enhanced is not None:
                mask = attention_mask_enhanced.unsqueeze(-1)  # [B, T, 1]
                pooled = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                pooled = hidden_states.mean(dim=1)  # [B, D]
            # detach: 分类器只用于监控方言识别能力，梯度不回传到encoder
            dialect_logits = self.dialect_classifier(pooled.detach())  # [B, num_dialects]

            if dialect_labels is not None:
                dialect_loss = nn.functional.cross_entropy(dialect_logits, dialect_labels)

        # 5. 计算CTC损失
        ctc_loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            attention_mask_ctc = (
                attention_mask if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask_ctc.sum(-1)).long()

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # Temperature scaling: T>1 使 softmax 分布更平滑（等效于 label smoothing，但零额外反向传播开销）
            if self.ctc_temperature != 1.0:
                log_probs = nn.functional.log_softmax(
                    logits / self.ctc_temperature, dim=-1, dtype=torch.float32
                ).transpose(0, 1)
            else:
                log_probs = nn.functional.log_softmax(
                    logits, dim=-1, dtype=torch.float32
                ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        # 6. 多任务总损失 = CTC + λ * 方言分类
        loss = None
        if ctc_loss is not None:
            loss = ctc_loss
            if dialect_loss is not None:
                loss = loss + self.dialect_loss_weight * dialect_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DialectModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # 不返回中间层输出，节省显存
            attentions=outputs.attentions if output_attentions else None,
            dialect_logits=dialect_logits,
        )

    def get_trainable_parameters(self):
        """获取可训练参数统计"""
        stats = {
            'total': 0, 'trainable': 0, 'frozen': 0,
            'dialect_enhancer': 0, 'dialect_classifier': 0,
            'wav2vec2': 0, 'lm_head': 0
        }

        for name, param in self.named_parameters():
            stats['total'] += param.numel()
            if param.requires_grad:
                stats['trainable'] += param.numel()
            else:
                stats['frozen'] += param.numel()

            if 'dialect_enhancer' in name:
                stats['dialect_enhancer'] += param.numel()
            elif 'dialect_classifier' in name:
                stats['dialect_classifier'] += param.numel()
            elif 'lm_head' in name:
                stats['lm_head'] += param.numel()
            else:
                stats['wav2vec2'] += param.numel()

        return stats

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        stats = self.get_trainable_parameters()

        print(f"\n{'='*60}")
        print("模型参数统计:")
        print(f"{'='*60}")
        print(f"总参数量:              {stats['total']:,}")
        print(f"可训练参数:            {stats['trainable']:,} ({stats['trainable']/stats['total']*100:.2f}%)")
        print(f"冻结参数:              {stats['frozen']:,} ({stats['frozen']/stats['total']*100:.2f}%)")
        print(f"{'-'*60}")
        print(f"Wav2Vec2主体:          {stats['wav2vec2']:,}")
        print(f"方言增强模块:          {stats['dialect_enhancer']:,}")
        print(f"方言分类头:            {stats['dialect_classifier']:,}")
        print(f"CTC输出层:             {stats['lm_head']:,}")
        print(f"{'='*60}\n")


def create_dialect_model_from_pretrained(
    pretrained_model_path,
    dialect_config=None,
    freeze_feature_extractor=True,
    freeze_base_model=False,
    unfreeze_last_n_layers=0
):
    """
    从预训练模型创建方言增强模型
    
    支持两种加载模式：
    1. 从原始预训练模型（origin_model/）：加载基础权重，新建增强模块
    2. 从训练checkpoint：加载完整模型权重（含增强模块），用于断点续训

    Args:
        pretrained_model_path: 预训练模型路径 或 checkpoint路径
        dialect_config: 方言增强配置（支持消融实验开关）
        freeze_feature_extractor: 是否冻结特征提取器
        freeze_base_model: 是否冻结基础模型
        unfreeze_last_n_layers: 解冻最后几层编码器

    Returns:
        EnhancedWav2Vec2ForDialect
    """
    # 检测是否为训练checkpoint（包含 dialect_config.json）
    is_checkpoint = os.path.exists(os.path.join(pretrained_model_path, "dialect_config.json"))
    
    if is_checkpoint:
        print(f"从训练checkpoint加载完整模型: {pretrained_model_path}")
        
        # 读取保存的方言配置
        with open(os.path.join(pretrained_model_path, "dialect_config.json"), "r") as f:
            saved_dialect_config = json.load(f)
        # 如果调用方提供了 dialect_config，优先使用；否则用checkpoint中保存的
        use_config = dialect_config if dialect_config is not None else saved_dialect_config
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = Wav2Vec2Config.from_pretrained(pretrained_model_path)
        
        model = EnhancedWav2Vec2ForDialect(config, use_config)
        
        # 加载完整权重（含 dialect_enhancer / dialect_classifier）
        import safetensors.torch
        weight_path = os.path.join(pretrained_model_path, "model.safetensors")
        if os.path.exists(weight_path):
            state_dict = safetensors.torch.load_file(weight_path)
        else:
            # 兼容 pytorch_model.bin
            state_dict = torch.load(
                os.path.join(pretrained_model_path, "pytorch_model.bin"),
                map_location="cpu"
            )
        model.load_state_dict(state_dict, strict=False)
        print("✓ Checkpoint完整权重加载完成")
    else:
        print(f"从预训练模型加载: {pretrained_model_path}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_path)

        model = EnhancedWav2Vec2ForDialect(base_model.config, dialect_config)

        # 复制预训练权重
        model.wav2vec2.load_state_dict(base_model.wav2vec2.state_dict())
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        model.dropout.load_state_dict(base_model.dropout.state_dict())
        print("✓ 预训练权重加载完成")
        
        del base_model

    # 应用冻结策略
    if freeze_feature_extractor:
        model.freeze_feature_extractor()
    if freeze_base_model:
        model.freeze_base_model()
    if unfreeze_last_n_layers > 0:
        model.unfreeze_encoder_layers(unfreeze_last_n_layers)

    model.print_trainable_parameters()
    return model


if __name__ == "__main__":
    print("测试增强型Wav2Vec2方言模型...")

    config = Wav2Vec2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=5169
    )

    # 测试 Full 配置
    print("\n1. 创建 Full 模型 (Adapter + Variation)...")
    model = EnhancedWav2Vec2ForDialect(config)
    model.freeze_feature_extractor()
    model.freeze_base_model()
    model.unfreeze_encoder_layers(4)
    model.print_trainable_parameters()

    # 测试消融：Adapter-only
    print("\n2. 创建 Adapter-only 模型...")
    model_adapter = EnhancedWav2Vec2ForDialect(
        config, {'use_adapter': True, 'use_variation_layer': False}
    )
    print(f"   增强模块参数: {sum(p.numel() for p in model_adapter.dialect_enhancer.parameters()):,}")

    # 测试前向传播
    print("\n3. 测试前向传播...")
    batch_size = 2
    seq_len = 16000
    input_values = torch.randn(batch_size, seq_len)
    labels = torch.randint(0, config.vocab_size, (batch_size, 100))

    outputs = model(input_values=input_values, labels=labels)
    print(f"   损失: {outputs.loss.item():.4f}")
    print(f"   Logits形状: {outputs.logits.shape}")

    # 测试方言分类多任务
    print("\n4. 测试多任务前向传播 (CTC + 方言分类)...")
    dialect_labels = torch.randint(0, 7, (batch_size,))
    outputs_mt = model(input_values=input_values, labels=labels, dialect_labels=dialect_labels)
    print(f"   多任务总损失: {outputs_mt.loss.item():.4f}")

    print("\n✓ 所有测试通过！")
