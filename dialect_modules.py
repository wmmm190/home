"""方言特征增强模块

基于消融实验设计，保留两个正交模块，各解决一个明确问题：
1. DialectAdapter - 层权重聚合方言适配器：学习各层对方言适配的贡献 + 瓶颈域适配
2. PronunciationVariationLayer - 发音变异感知层：语言学驱动的多尺度变异捕捉

创新点：
- DialectAdapter 引入可学习层权重，聚合 wav2vec2 全部 25 层（1 CNN + 24 Transformer）
  的输出，而非仅用最后一层。方言信息在不同层级分布不均（Pasad et al. 2021），层权重
  训练后可可视化，揭示方言信息主要编码在哪些层，兼具可解释性。
- PronunciationVariationLayer 采用语言学驱动的卷积核尺度 [3, 11, 21]，分别对应
  声母级(~60ms)、韵母/音节级(~220ms)、双音节词级(~420ms) 的时间粒度
  （wav2vec2 每帧 ≈ 20ms），替代无语言学依据的 [3, 5, 7]。

设计取舍说明：
- 移除 ToneAwareAttention：Feng et al. 2024 (Wav2f0) 证明 wav2vec2 中间层
  已编码丰富的 F0/声调信息，Adapter 微调这些层时声调适配隐式完成，
  单独声调模块在消融实验中与 Adapter 效果重叠。
- 两模块解决不重叠的问题：Adapter→特征空间适配，变异层→局部时序模式。

参考文献：
[1] Wang et al., "How to Learn a New Language?", arXiv:2411.18217, 2024.
[2] Liu et al., "PELE", Interspeech 2024.  (Adapter优于LoRA)
[3] Gulati et al., "Conformer", Interspeech 2020.  (CNN+Attention互补)
[4] Feng et al., "Wav2f0", NCMMSC 2024.  (wav2vec2隐式编码声调)
[5] Pasad et al., "Layer-wise Analysis of a Self-supervised Speech Representation Model", IEEE ASRU 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DialectAdapter(nn.Module):
    """
    层权重聚合方言适配器

    解决问题：预训练模型的特征空间偏向普通话，方言数据稀缺导致全量微调过拟合。

    创新点：学习 wav2vec2 各层（1 CNN + 24 Transformer = 25 层）对方言适配的
    贡献权重，对加权聚合后的表示做瓶颈变换。训练后可视化 softmax(layer_weights)
    揭示方言信息在预训练模型中的层级分布——这本身就是一个可解释性分析图表。

    设计要点：
    - 可学习层权重 softmax 归一化后加权求和各层输出
    - 瓶颈结构 (D→B→D) 大幅减少可训练参数，适合小数据集
    - 近零初始化确保训练初期不破坏预训练表示
    - 残差连接保留模型通用能力
    - 可学习缩放因子控制适配强度

    参考：
    - Wang et al. 2024 (adapter+预热)
    - Liu et al. 2024 (PELE, Adapter优于LoRA)
    - Pasad et al. 2021 (wav2vec2各层编码不同级别的语音信息)

    Args:
        hidden_size: 输入/输出维度 (默认1024)
        bottleneck_size: 瓶颈维度 (默认64，降低过拟合风险)
        num_layers: wav2vec2输出层数 (默认25 = 1 CNN + 24 Transformer)
        dropout: Dropout比率 (默认0.15，增强正则化)
    """

    def __init__(self, hidden_size=1024, bottleneck_size=64, num_layers=25, dropout=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.num_layers = num_layers

        # 可学习层权重：每层一个标量，softmax归一化后加权求和
        # 初始化为全零 → softmax(0,...,0) = 1/N，即均匀权重
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # 瓶颈结构（bottleneck=64, 参数量从527K降至131K，缓解过拟合）
        self.down_project = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 可学习缩放因子，初始值较小，前向传播时 clamp 到 [0, 0.5] 防止过大
        self.adapter_scale = nn.Parameter(torch.tensor(0.1))

        # 近零初始化：训练初期等效恒等映射
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states, all_hidden_states=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 最后一层输出（兜底用）
            all_hidden_states: tuple of [batch_size, seq_len, hidden_size]，
                               wav2vec2 全部层输出。若提供，则做层权重聚合。
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # ★ 关键修复：保留原始最后一层输出作为残差，保证梯度完整回传到编码器
        # 修复前：hidden_states 被 weighted_sum 替换，梯度被 w_24≈1/25 衰减 ~25倍
        # 修复后：原始 hidden_states 通过残差直通，层权重聚合只提供跨层上下文给瓶颈
        residual = hidden_states  # 保存原始最后一层输出（梯度直通路径）

        # 层权重聚合：逐层加权求和，作为瓶颈变换的输入（跨层上下文信息）
        if all_hidden_states is not None and len(all_hidden_states) > 1:
            num_layers = min(len(all_hidden_states), self.num_layers)
            weights = F.softmax(self.layer_weights[:num_layers], dim=0)  # [L]
            adapter_input = torch.zeros_like(all_hidden_states[0])
            for i in range(num_layers):
                adapter_input = adapter_input + weights[i] * all_hidden_states[i]
        else:
            adapter_input = hidden_states

        # 瓶颈变换：将跨层上下文映射为小扰动
        down = self.down_project(adapter_input)
        activated = self.activation(down)
        up = self.up_project(activated)
        up = self.dropout(up)

        # clamp adapter_scale 防止过大，残差来自原始最后一层输出（非 weighted_sum）
        scale = self.adapter_scale.clamp(0.0, 0.5)
        output = self.layer_norm(residual + scale * up)

        return output

    def get_layer_weights(self):
        """返回归一化后的层权重，用于可视化分析"""
        return F.softmax(self.layer_weights, dim=0).detach().cpu().numpy()


class PronunciationVariationLayer(nn.Module):
    """
    发音变异感知层（多尺度卷积 + 全局注意力）

    解决问题：方言存在系统性的音素替换（如四川话 zh→z, sh→s）、省略和连读，
    Transformer 的全局注意力对局部相邻帧的细粒度变异不够敏感。

    创新点：语言学驱动的多尺度卷积核尺度选择
    wav2vec2 每帧 ≈ 20ms（下采样率320, 采样率16kHz），因此：
    - kernel=3  → ~60ms  → 声母级（捕捉声母替换，如 zh→z, n→l）
    - kernel=11 → ~220ms → 韵母/音节级（捕捉韵母变异，如前后鼻音合并）
    - kernel=21 → ~420ms → 双音节词级（捕捉连读变调、词级模式）
    每个核大小对应一个明确的语言学层级，而非任意选择。

    设计要点：
    - 语言学驱动的多尺度 1D 卷积分别捕捉声母级/韵母级/词级变异
    - 全局自注意力确保全句发音风格一致
    - 前馈网络增加非线性表达能力
    - 每步残差连接 + LayerNorm

    与 Adapter 的分工：
    - Adapter 做特征空间变换（域适配），解决"见没见过"的问题
    - 本层捕捉局部时序模式（发音规律），解决"音素怎么替换"的问题
    两者消融时各自有独立贡献，不重叠。

    参考：Gulati et al. 2020 (Conformer, CNN+Attention互补)

    Args:
        hidden_size: 输入维度 (默认1024)
        kernel_sizes: 卷积核大小列表 (默认[3, 11, 21]，对应声母/韵母/词级时间尺度)
        num_attention_heads: 注意力头数 (默认8)
        dropout: Dropout比率 (默认0.1)
    """

    def __init__(self, hidden_size=1024, kernel_sizes=None,
                 num_attention_heads=8, dropout=0.1):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 11, 21]  # 声母级/韵母级/词级时间尺度

        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.num_attention_heads = num_attention_heads

        # 多尺度卷积层
        branch_dim = hidden_size // len(kernel_sizes)
        # 确保 groups 能整除 in_channels 和 out_channels
        groups = max(1, min(8, branch_dim))
        while branch_dim % groups != 0 or hidden_size % groups != 0:
            groups -= 1
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=branch_dim,
                kernel_size=k,
                padding=k // 2,
                groups=groups  # 分组卷积减少参数
            )
            for k in kernel_sizes
        ])
        # 卷积分支拼接后维度可能不等于 hidden_size，需投影回来
        concat_dim = branch_dim * len(kernel_sizes)

        # 卷积输出投影
        self.conv_projection = nn.Linear(concat_dim, hidden_size)

        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络（2x扩展，比标准4x省一半显存，对方言特征足够）
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 可选
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states

        # 1. 多尺度卷积捕捉局部发音变异
        conv_input = hidden_states.transpose(1, 2)  # [B, D, T]

        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(conv_input)
            conv_out = F.gelu(conv_out)
            conv_outputs.append(conv_out)

        multi_scale = torch.cat(conv_outputs, dim=1)  # [B, D, T]
        multi_scale = multi_scale.transpose(1, 2)     # [B, T, D]

        multi_scale = self.conv_projection(multi_scale)
        multi_scale = self.dropout(multi_scale)
        hidden_states = self.layer_norm1(residual + multi_scale)

        # 2. 全局自注意力捕捉全句发音风格一致性
        residual = hidden_states

        attn_mask = None
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)

        attn_output, _ = self.self_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attn_mask
        )
        attn_output = self.dropout(attn_output)
        hidden_states = self.layer_norm2(residual + attn_output)

        # 3. 前馈网络
        residual = hidden_states
        ff_output = self.feed_forward(hidden_states)
        output = self.layer_norm3(residual + ff_output)

        return output


class DialectFeatureEnhancer(nn.Module):
    """
    方言特征增强器 - 整合两个增强模块

    支持通过配置灵活开关各模块，便于消融实验：
    - use_adapter=True,  use_variation_layer=True  → Full（完整方案）
    - use_adapter=True,  use_variation_layer=False → Adapter-only
    - use_adapter=False, use_variation_layer=True  → Variation-only
    - use_adapter=False, use_variation_layer=False → Baseline（透传）

    Args:
        hidden_size: 隐藏层维度
        config: 配置字典
    """

    def __init__(self, hidden_size=1024, config=None):
        super().__init__()

        default_config = {
            'use_adapter': True,
            'use_variation_layer': True,
            'bottleneck_size': 64,
            'num_layers': 25,
            'kernel_sizes': [3, 11, 21],
            'num_attention_heads': 8,
            'dropout': 0.1
        }

        if config is not None:
            default_config.update(config)

        self.config = default_config
        self.hidden_size = hidden_size

        if self.config['use_adapter']:
            self.dialect_adapter = DialectAdapter(
                hidden_size=hidden_size,
                bottleneck_size=self.config['bottleneck_size'],
                num_layers=self.config.get('num_layers', 25),
                dropout=self.config['dropout']
            )

        if self.config['use_variation_layer']:
            self.variation_layer = PronunciationVariationLayer(
                hidden_size=hidden_size,
                kernel_sizes=self.config['kernel_sizes'],
                num_attention_heads=self.config['num_attention_heads'],
                dropout=self.config['dropout']
            )

    def forward(self, hidden_states, attention_mask=None, all_hidden_states=None):
        """
        依次应用各增强模块

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 最后一层编码器输出
            attention_mask: [batch_size, seq_len] 可选
            all_hidden_states: tuple of tensors, wav2vec2各层输出，用于层权重聚合
        Returns:
            enhanced_features: [batch_size, seq_len, hidden_size]
        """
        if self.config['use_adapter']:
            hidden_states = self.dialect_adapter(hidden_states, all_hidden_states)

        if self.config['use_variation_layer']:
            hidden_states = self.variation_layer(hidden_states, attention_mask)

        return hidden_states


if __name__ == "__main__":
    print("测试方言特征增强模块...")

    batch_size = 2
    seq_len = 100
    hidden_size = 1024
    num_layers = 25  # wav2vec2-large: 1 CNN + 24 Transformer

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    # 模拟 wav2vec2 各层输出
    all_hidden_states = tuple(torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers))

    # 测试方言适配器（含层权重聚合）
    print("\n1. 测试层权重聚合方言适配器 (DialectAdapter)...")
    adapter = DialectAdapter(hidden_size=hidden_size, num_layers=num_layers)
    output1 = adapter(hidden_states, all_hidden_states)
    print(f"   输入形状: {hidden_states.shape}")
    print(f"   输出形状: {output1.shape}")
    print(f"   参数量: {sum(p.numel() for p in adapter.parameters()):,}")
    print(f"   层权重分布: {adapter.get_layer_weights()[:5]}... (前5层)")

    # 测试无 all_hidden_states 时的兜底模式
    print("\n   测试兜底模式（无 all_hidden_states）...")
    output1b = adapter(hidden_states, None)
    print(f"   兜底输出形状: {output1b.shape}")

    # 测试发音变异感知层（语言学驱动核尺度）
    print("\n2. 测试发音变异感知层 (PronunciationVariationLayer, kernels=[3,11,21])...")
    variation = PronunciationVariationLayer(hidden_size=hidden_size)
    output2 = variation(hidden_states, attention_mask)
    print(f"   输入形状: {hidden_states.shape}")
    print(f"   输出形状: {output2.shape}")
    print(f"   参数量: {sum(p.numel() for p in variation.parameters()):,}")
    print(f"   卷积核尺度: {variation.kernel_sizes} (声母/韵母/词级)")

    # 测试完整增强器（Full）
    print("\n3. 测试完整增强器 (Full: Adapter + Variation)...")
    enhancer_full = DialectFeatureEnhancer(hidden_size=hidden_size)
    output3 = enhancer_full(hidden_states, attention_mask, all_hidden_states)
    print(f"   输出形状: {output3.shape}")
    print(f"   总参数量: {sum(p.numel() for p in enhancer_full.parameters()):,}")

    # 测试消融配置：Adapter-only
    print("\n4. 测试消融配置 (Adapter-only)...")
    enhancer_adapter = DialectFeatureEnhancer(
        hidden_size=hidden_size,
        config={'use_adapter': True, 'use_variation_layer': False}
    )
    output4 = enhancer_adapter(hidden_states, attention_mask, all_hidden_states)
    print(f"   输出形状: {output4.shape}")
    print(f"   参数量: {sum(p.numel() for p in enhancer_adapter.parameters()):,}")

    # 测试消融配置：Variation-only
    print("\n5. 测试消融配置 (Variation-only)...")
    enhancer_var = DialectFeatureEnhancer(
        hidden_size=hidden_size,
        config={'use_adapter': False, 'use_variation_layer': True}
    )
    output5 = enhancer_var(hidden_states, attention_mask)
    print(f"   输出形状: {output5.shape}")
    print(f"   参数量: {sum(p.numel() for p in enhancer_var.parameters()):,}")

    print("\n✓ 所有模块测试通过！")
