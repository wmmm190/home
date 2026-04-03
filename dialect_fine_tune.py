"""
方言微调训练脚本
使用增强型Wav2Vec2模型（层权重聚合适配器 + 语言学驱动发音变异感知层）和数据增强进行训练
支持多任务学习：CTC语音识别 + 方言类型分类
支持消融实验的四种配置：Full / Adapter-only / Variation-only / Baseline
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
import platform
import random
import warnings
import logging
import csv
from datetime import datetime
from transformers import Wav2Vec2Processor, get_cosine_schedule_with_warmup
import numpy as np
from jiwer import cer as compute_cer

from dialect_model import create_dialect_model_from_pretrained
from dialect_augmentation import DialectDatasetAugmented, DialectAudioAugmenter
from preprocess_data import PreprocessedDataset

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """设置全局随机种子，保证实验可复现"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir="./logs"):
    """设置日志：同时输出到终端和文件"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logger = logging.getLogger("dialect_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 文件handler（详细信息）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    
    # 终端handler（同步输出）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"日志文件: {log_file}")
    return logger, log_file

# CUDA优化设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

# 限制 PyTorch 只使用物理显存的70%，避免溢出到慢速共享GPU内存
# 注意：其他进程（浏览器、DWM等）也会占用GPU显存，85%太激进
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.70, 0)


def collate_fn(batch):
    """数据批次整理函数（支持 dialect_id）"""
    # 找到批次中最长的序列
    max_length = max(item["input_values"].size(0) for item in batch)
    max_label_length = max(item["labels"].size(0) for item in batch)
    
    # 填充序列
    input_values_list = []
    attention_mask_list = []
    labels_list = []
    dialect_ids_list = []
    
    for item in batch:
        # 填充输入值
        input_values = item["input_values"]
        if input_values.size(0) < max_length:
            padding = torch.zeros(max_length - input_values.size(0))
            input_values = torch.cat([input_values, padding], dim=0)
        input_values_list.append(input_values)
        
        # 填充注意力掩码
        if item["attention_mask"] is not None:
            attention_mask = item["attention_mask"]
            if attention_mask.size(0) < max_length:
                padding = torch.zeros(max_length - attention_mask.size(0))
                attention_mask = torch.cat([attention_mask, padding], dim=0)
            attention_mask_list.append(attention_mask)
        else:
            attention_mask_list.append(torch.ones(max_length))
            
        # 填充标签
        labels = item["labels"]
        if labels.size(0) < max_label_length:
            padding = torch.full((max_label_length - labels.size(0),), -100)
            labels = torch.cat([labels, padding], dim=0)
        labels_list.append(labels)
        
        # 方言ID（兼容 int 和 tensor 两种存储格式）
        did = item.get("dialect_id", -1)
        dialect_ids_list.append(int(did) if isinstance(did, (int, float)) else did.item())
    
    result = {
        "input_values": torch.stack(input_values_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
        "dialect_labels": torch.tensor(dialect_ids_list, dtype=torch.long)
    }
    return result


class DialectModelTrainer:
    """方言模型训练器"""
    
    # 方言名称映射
    DIALECT_NAMES = {
        0: '武汉话', 1: '南昌话', 2: '上海话', 3: '四川话',
        4: '天津话', 5: '长沙话', 6: '郑州话'
    }
    
    def __init__(
        self,
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dialect_config=None,
        freeze_strategy="progressive"
    ):
        """
        Args:
            model_path: 预训练模型路径
            device: 训练设备
            dialect_config: 方言增强配置
            freeze_strategy: 冻结策略 ['progressive', 'adapter_only', 'full']
        """
        # 初始化日志
        self.logger, self.log_file = setup_logger()
        
        # 检查CUDA和GPU信息
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            capability = torch.cuda.get_device_capability(0)
            self.logger.info(f"计算能力: sm{capability[0]}{capability[1]}")
            device = "cuda"
        else:
            device = "cpu"
            self.logger.info("CUDA不可用，使用CPU进行训练")
            
        self.device = device
        self.freeze_strategy = freeze_strategy
        
        # 加载处理器
        self.logger.info("加载处理器...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        
        # 创建增强型模型
        self.logger.info("创建增强型方言模型...")
        
        if freeze_strategy == "adapter_only":
            # 只训练适配器
            self.unfreeze_last_n_layers = 0
            self.model = create_dialect_model_from_pretrained(
                model_path,
                dialect_config=dialect_config,
                freeze_feature_extractor=True,
                freeze_base_model=True,
                unfreeze_last_n_layers=0
            )
        elif freeze_strategy == "progressive":
            # 渐进式解冻：冻结主体，只解冻最后6层 + 增强模块 + 分类头
            # 从3层增加到6层：在中等数据量下增加编码器容量以更好拟合方言差异
            # 注意：显存和过拟合风险上升，短跑验证后再决定是否长期使用
            self.unfreeze_last_n_layers = 6
            self.model = create_dialect_model_from_pretrained(
                model_path,
                dialect_config=dialect_config,
                freeze_feature_extractor=True,
                freeze_base_model=True,
                unfreeze_last_n_layers=6
            )
        else:  # full
            # 完全微调
            self.unfreeze_last_n_layers = 24
            self.model = create_dialect_model_from_pretrained(
                model_path,
                dialect_config=dialect_config,
                freeze_feature_extractor=True,
                freeze_base_model=False,
                unfreeze_last_n_layers=24
            )
        
        self.model.to(self.device)
        
        # 启用 gradient checkpointing 节省显存
        if hasattr(self.model.wav2vec2, 'gradient_checkpointing_enable'):
            self.model.wav2vec2.gradient_checkpointing_enable()
            self.logger.info("✓ 已启用 gradient checkpointing（节省显存）")
        
        # AMP 混合精度训练
        self.use_amp = (self.device == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            self.logger.info("✓ 已启用 AMP 混合精度训练（节省显存 + 加速计算）")
        
        # 创建数据增强器
        self.augmenter = DialectAudioAugmenter(sample_rate=16000)

        # 输出解冻层数与模块打开情况
        cfg = dialect_config if dialect_config is not None else {}
        use_adapter = bool(cfg.get('use_adapter', False))
        use_variation_layer = bool(cfg.get('use_variation_layer', False))
        dialect_classifier_on = (getattr(self.model, 'dialect_classifier', None) is not None)
        self.logger.info(f"解冻编码器层数: 最后 {self.unfreeze_last_n_layers} 层")
        self.logger.info("模块打开情况: "
                 f"Adapter={'开' if use_adapter else '关'} | "
                 f"VariationLayer={'开' if use_variation_layer else '关'} | "
                 f"DialectClassifier={'开' if dialect_classifier_on else '关'} | "
                 f"OnlineAugmenter={'开' if self.augmenter is not None else '关'}")
        
        # 打印模型加载后的显存
        self._log_vram("模型加载完成")
    
    def _log_vram(self, tag=""):
        """打印当前GPU显存使用情况"""
        if self.device != "cuda":
            return
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        self.logger.info(f"[显存] {tag}: 已分配={allocated:.2f}GB, "
                         f"已预留={reserved:.2f}GB, 峰值={max_allocated:.2f}GB")
    
    def create_dataloaders(self, train_data, val_data, batch_size=2, 
                          augment_prob=0.8, num_workers=None):
        """创建数据加载器（支持预处理数据和原始数据两种模式）"""
        # Windows 下 num_workers>0 用 spawn 方式创建进程，对随机读 .pt 反而更慢
        if num_workers is None:
            num_workers = 0 if platform.system() == "Windows" else 2
            self.logger.info(f"数据加载工作进程数: {num_workers}")
        
        # 判断是否使用预处理数据
        if isinstance(train_data, str) and os.path.isdir(train_data):
            self.logger.info(f"使用预处理数据: {train_data}")
            train_dataset = PreprocessedDataset(train_data)
            val_dataset = PreprocessedDataset(val_data)
        else:
            self.logger.info("使用在线数据加载（较慢，建议先运行 preprocess_data.py）")
            train_dataset = DialectDatasetAugmented(
                train_data,
                self.processor,
                self.augmenter,
                augment_prob=augment_prob,
                training=True
            )
            val_dataset = DialectDatasetAugmented(
                val_data,
                self.processor,
                self.augmenter,
                augment_prob=0.0,
                training=False
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            worker_init_fn=lambda wid: np.random.seed(42 + wid),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            worker_init_fn=lambda wid: np.random.seed(42 + wid),
        )
        
        return train_loader, val_loader
    
    def setup_optimizer(self, learning_rate=3e-5, weight_decay=0.1):
        """设置优化器 - 使用分层学习率"""
        # 分组参数
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            # 方言增强模块 - 适中学习率（1.5x，从3x降低以缓解过拟合）
            {
                "params": [p for n, p in self.model.dialect_enhancer.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": weight_decay,
                "lr": learning_rate * 1.5
            },
            {
                "params": [p for n, p in self.model.dialect_enhancer.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": learning_rate * 1.5
            },
            # Transformer编码器 - 中等学习率
            {
                "params": [p for n, p in self.model.wav2vec2.encoder.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": weight_decay,
                "lr": learning_rate
            },
            {
                "params": [p for n, p in self.model.wav2vec2.encoder.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": learning_rate
            },
            # CTC头 - 较大学习率（2x）
            {
                "params": [p for n, p in self.model.lm_head.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": weight_decay,
                "lr": learning_rate * 2
            },
            {
                "params": [p for n, p in self.model.lm_head.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": learning_rate * 2
            },
        ]
        
        # 方言分类头 - 与base相同学习率（已detach梯度，不影响encoder）
        if self.model.dialect_classifier is not None:
            optimizer_grouped_parameters.extend([
                {
                    "params": [p for n, p in self.model.dialect_classifier.named_parameters()
                              if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": weight_decay,
                    "lr": learning_rate * 1  # 从10x降为1x，配合detach防止干扰ASR
                },
                {
                    "params": [p for n, p in self.model.dialect_classifier.named_parameters()
                              if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": learning_rate * 1
                },
            ])
        
        # 过滤空参数组
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
        
        optimizer = AdamW(optimizer_grouped_parameters)
        return optimizer
    
    def train(
        self,
        train_data,
        val_data,
        epochs=10,
        batch_size=2,
        learning_rate=2e-5,
        augment_prob=0.8,
        warmup_steps=500,
        gradient_accumulation_steps=1,
        early_stop_patience=3,
        save_path="./dialect_model",
        resume_from=None
    ):
        """
        训练模型
        
        Args:
            train_data: 训练数据列表
            val_data: 验证数据列表
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            augment_prob: 数据增强概率
            warmup_steps: 预热步数
            gradient_accumulation_steps: 梯度累积步数
            early_stop_patience: 早停轮数
            save_path: 模型保存路径
            resume_from: 断点续训的checkpoint路径（如 './dialect_model_interrupted'）
        """
        log = self.logger
        
        log.info(f"\n{'='*60}")
        log.info("开始方言模型训练")
        log.info(f"{'='*60}")
        
        # 正确获取数据集样本数（目录路径 vs 列表）
        if isinstance(train_data, str):
            import glob
            train_count = len(glob.glob(os.path.join(train_data, "sample_*.pt")))
            val_count = len(glob.glob(os.path.join(val_data, "sample_*.pt")))
        else:
            train_count = len(train_data)
            val_count = len(val_data)
        
        log.info(f"训练样本: {train_count}")
        log.info(f"验证样本: {val_count}")
        log.info(f"批次大小: {batch_size}")
        log.info(f"梯度累积: {gradient_accumulation_steps}")
        log.info(f"有效批次: {batch_size * gradient_accumulation_steps}")
        log.info(f"学习率: {learning_rate}")
        log.info(f"数据增强概率: {augment_prob}")
        log.info(f"早停判据: Macro CER（各方言等权平均）")
        log.info(f"日志文件: {self.log_file}")
        log.info(f"{'='*60}\n")
        
        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders(
            train_data, val_data, batch_size, augment_prob
        )
        
        # 设置优化器
        optimizer = self.setup_optimizer(learning_rate)
        
        # 设置学习率调度器（cosine比linear衰减更缓，后期保留更多学习能力）
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # 标准半余弦：从peak平滑降到0
        )
        
        # 方言动态损失加权：高CER方言获得更高损失权重
        # 初始各方言等权（1.0），训练过程中根据验证CER自适应调整
        # 效果优于静态过采样：因为它根据当前模型性能动态分配梯度信号
        self.dialect_loss_weights = torch.ones(len(self.DIALECT_NAMES), device=self.device)
        
        # 初始化CSV训练记录（续训时尝试追加到原CSV）
        resumed_csv = None
        if resume_from is not None:
            ckpt_state_path = os.path.join(resume_from, "training_state.pt")
            if os.path.exists(ckpt_state_path):
                tmp_ckpt = torch.load(ckpt_state_path, map_location="cpu", weights_only=False)
                resumed_csv = tmp_ckpt.get('csv_path', None)
                del tmp_ckpt
        
        if resumed_csv and os.path.exists(resumed_csv):
            # 续训：追加到原CSV文件
            csv_path = resumed_csv
            csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            log.info(f"续训CSV（追加模式）: {csv_path}")
        else:
            # 新训练：创建新CSV
            csv_path = os.path.join(os.path.dirname(self.log_file), 
                                    f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'epoch', 'train_loss', 'val_loss', 'val_cer', 'macro_cer',
                'dialect_accuracy', 'lr', 'epoch_time_sec',
                # 分方言CER列
                *[f'cer_{name}' for name in self.DIALECT_NAMES.values()]
            ])
            log.info(f"训练指标CSV: {csv_path}")
        
        # 训练循环
        best_val_cer = float('inf')
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        start_epoch = 0
        epoch = 0  # 防止循环未执行时 except 中引用未定义变量
        
        # 断点续训：恢复训练状态
        if resume_from is not None:
            checkpoint = self.load_checkpoint(resume_from, optimizer, scheduler, self.scaler)
            if checkpoint is not None:
                global_step = checkpoint['global_step']
                best_val_cer = checkpoint['best_val_cer']
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                patience_counter = checkpoint['patience_counter']
                
                # 关键：根据epoch是否完成决定从哪里开始
                epoch_completed = checkpoint.get('epoch_completed', True)
                saved_epoch = checkpoint['epoch']
                
                if epoch_completed:
                    # epoch正常完成后保存的checkpoint → 从下一个epoch开始
                    start_epoch = saved_epoch + 1
                    log.info(f"从 epoch {start_epoch + 1} 继续训练 "
                             f"(已完成 {saved_epoch + 1} 轮, 最佳CER: {best_val_cer:.4f})")
                else:
                    # epoch中途中断 → 从当前epoch头重新开始（不跳过未完成的数据）
                    start_epoch = saved_epoch
                    log.info(f"从 epoch {start_epoch + 1} 重新开始 "
                             f"(上次在该epoch中途中断, 最佳CER: {best_val_cer:.4f})")
                    log.info(f"注意: 将重新训练 epoch {start_epoch + 1} 的全部数据")
                
                # 恢复方言动态损失权重
                if 'dialect_loss_weights' in checkpoint and checkpoint['dialect_loss_weights'] is not None:
                    self.dialect_loss_weights = checkpoint['dialect_loss_weights'].to(self.device)
                    log.info(f"恢复方言损失权重: " +
                             " | ".join(f"{n}={self.dialect_loss_weights[i]:.2f}"
                                       for i, n in self.DIALECT_NAMES.items()))
            else:
                log.info("未能恢复训练状态，从头开始训练")
        
        csv_closed = False
        try:
            for epoch in range(start_epoch, epochs):
                epoch_start_time = time.time()
                self.model.train()
                
                total_loss = 0
                num_batches = 0
                optimizer.zero_grad(set_to_none=True)
                
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        # 移动数据到设备（non_blocking 配合 pin_memory 异步传输）
                        input_values = batch["input_values"].to(self.device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                        labels = batch["labels"].to(self.device, non_blocking=True)
                        dialect_labels = batch["dialect_labels"].to(self.device, non_blocking=True)
                        
                        # AMP 前向传播（混合精度）
                        with torch.amp.autocast("cuda", enabled=self.use_amp):
                            outputs = self.model(
                                input_values=input_values,
                                attention_mask=attention_mask,
                                labels=labels,
                                dialect_labels=dialect_labels
                            )
                            # 方言动态损失加权：CER高的方言获得更大梯度信号
                            batch_weights = self.dialect_loss_weights[dialect_labels.clamp(min=0)]
                            dialect_scale = batch_weights.mean()
                            loss = outputs.loss * dialect_scale / gradient_accumulation_steps
                        
                        # AMP 反向传播
                        self.scaler.scale(loss).backward()
                        
                        batch_loss = outputs.loss.item()
                        
                        # 及时释放显存：删除引用以便 GC 回收计算图
                        del outputs, loss
                        
                        # 梯度累积
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            # AMP unscale + 梯度裁剪
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                        
                        total_loss += batch_loss
                        num_batches += 1
                        
                        if (batch_idx + 1) % 10 == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            log.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                                  f"Loss: {batch_loss:.4f}, LR: {current_lr:.2e}")
                        
                        # 每100个batch打印显存
                        if (batch_idx + 1) % 100 == 0:
                            self._log_vram(f"Epoch {epoch+1} Batch {batch_idx+1}")
                            
                    except Exception as e:
                        log.info(f"处理批次 {batch_idx} 时出错: {e}")
                        continue
                
                # Epoch统计
                epoch_duration = time.time() - epoch_start_time
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                
                log.info(f"\n{'-'*60}")
                log.info(f"Epoch {epoch+1}/{epochs} 完成")
                log.info(f"平均训练损失: {avg_loss:.4f}")
                log.info(f"耗时: {epoch_duration:.2f}秒")
                log.info(f"{'-'*60}\n")
                
                # 验证（带CER/方言分类指标）
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
                val_cer = val_metrics['val_cer']
                macro_cer = val_metrics.get('macro_cer', val_cer)
                dialect_acc = val_metrics['dialect_accuracy']
                per_dialect_cer = val_metrics.get('per_dialect_cer', {})
                
                # 更新方言动态损失权重：CER越高的方言，下一轮训练获得越高的损失权重
                if per_dialect_cer:
                    new_weights = torch.ones(len(self.DIALECT_NAMES), device=self.device)
                    for d_id, d_name in self.DIALECT_NAMES.items():
                        if d_name in per_dialect_cer and per_dialect_cer[d_name] >= 0:
                            new_weights[d_id] = per_dialect_cer[d_name]
                    # 归一化使均值为1（保持总体梯度尺度不变）
                    w_mean = new_weights.mean()
                    if w_mean > 0:
                        new_weights = new_weights / w_mean
                    # 限制范围避免极端值导致训练不稳定（更保守的上下限）
                    new_weights = new_weights.clamp(0.7, 2.0)
                    # EMA平滑更新（提高平滑系数以减少波动对训练的影响）
                    self.dialect_loss_weights = 0.85 * self.dialect_loss_weights + 0.15 * new_weights
                    log.info(f"  方言损失权重: " +
                             " | ".join(f"{n}={self.dialect_loss_weights[i]:.2f}"
                                       for i, n in self.DIALECT_NAMES.items()))
                
                # 写入CSV
                csv_row = [
                    epoch + 1, f"{avg_loss:.6f}", f"{val_loss:.6f}", f"{val_cer:.6f}",
                    f"{macro_cer:.6f}",
                    f"{dialect_acc:.4f}", f"{current_lr:.2e}", f"{epoch_duration:.1f}",
                ]
                for dialect_name in self.DIALECT_NAMES.values():
                    csv_row.append(f"{per_dialect_cer.get(dialect_name, -1):.6f}")
                csv_writer.writerow(csv_row)
                csv_file.flush()  # 立即写入磁盘
                
                # 早停判据：使用macro CER（各方言等权平均，避免武汉话短文本主导指标）
                if macro_cer < best_val_cer:
                    best_val_cer = macro_cer
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    best_model_path = f"{save_path}_best"
                    self.save_model(best_model_path)
                    log.info(f"✓ 新最佳! Macro CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%), "
                             f"Micro CER: {val_cer:.4f}, Val Loss: {val_loss:.4f}, "
                             f"方言准确率: {dialect_acc*100:.2f}%")
                else:
                    patience_counter += 1
                    log.info(f"Macro CER未改善 (当前: {macro_cer:.4f}, 最佳: {best_val_cer:.4f}) "
                             f"({patience_counter}/{early_stop_patience})")
                    
                    if patience_counter >= early_stop_patience:
                        log.info(f"\n早停机制触发，训练结束")
                        break
                
                # 每个epoch结束保存checkpoint（用于断点续训）
                ckpt_dir = f"{save_path}_checkpoint"
                self.save_checkpoint(ckpt_dir, optimizer, scheduler, self.scaler,
                                     epoch, global_step, best_val_cer, best_val_loss,
                                     patience_counter, csv_path,
                                     epoch_completed=True)  # 标记为已完成
                
        except KeyboardInterrupt:
            log.info("\n\n训练被用户中断！正在保存checkpoint...")
            ckpt_dir = f"{save_path}_interrupted"
            self.save_checkpoint(ckpt_dir, optimizer, scheduler, self.scaler,
                                 epoch, global_step, best_val_cer, best_val_loss,
                                 patience_counter, csv_path,
                                 epoch_completed=False)  # 标记为未完成
            csv_file.close()
            csv_closed = True
            log.info(f"✓ Checkpoint已保存到: {ckpt_dir}")
            log.info(f"  续训方式: 将 main() 中 resume_from 改为 '{ckpt_dir}'")
            return
        finally:
            if not csv_closed:
                csv_file.close()
        
        # 保存最终模型
        final_model_path = f"{save_path}_final"
        self.save_model(final_model_path)
        log.info(f"\n✓ 训练完成！最终模型已保存到: {final_model_path}")
        log.info(f"✓ 最佳Macro CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%)")
        log.info(f"✓ 训练指标CSV: {csv_path}")
        log.info(f"✓ 完整日志: {self.log_file}")
    
    def validate(self, val_loader):
        """
        验证模型 - 计算loss + CER + 方言分类准确率 + 分方言CER
        
        Returns:
            dict: {val_loss, val_cer, dialect_accuracy, per_dialect_cer, sample_predictions}
        """
        log = self.logger
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_preds = []       # 解码后的预测文本
        all_refs = []         # 解码后的参考文本
        all_dialect_preds = []  # 方言分类预测
        all_dialect_refs = []   # 方言真实标签
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    input_values = batch["input_values"].to(self.device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                    labels = batch["labels"].to(self.device, non_blocking=True)
                    dialect_labels = batch["dialect_labels"].to(self.device, non_blocking=True)
                    
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        outputs = self.model(
                            input_values=input_values,
                            attention_mask=attention_mask,
                            labels=labels,
                            dialect_labels=dialect_labels
                        )
                    
                    total_loss += outputs.loss.item()
                    num_batches += 1
                    
                    # --- CER计算：解码预测和参考 ---
                    logits = outputs.logits  # (batch, seq_len, vocab_size)
                    pred_ids = torch.argmax(logits, dim=-1)  # (batch, seq_len)
                    
                    # 解码预测文本
                    pred_texts = self.processor.batch_decode(pred_ids)
                    
                    # 解码参考文本（需要把-100的padding去掉）
                    label_ids = labels.clone()
                    label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
                    ref_texts = self.processor.batch_decode(label_ids, group_tokens=False)
                    
                    all_preds.extend(pred_texts)
                    all_refs.extend(ref_texts)
                    
                    # --- 方言分类准确率 ---
                    if hasattr(outputs, 'dialect_logits') and outputs.dialect_logits is not None:
                        dialect_pred = outputs.dialect_logits.argmax(dim=-1)  # (batch,)
                        all_dialect_preds.extend(dialect_pred.cpu().tolist())
                        all_dialect_refs.extend(dialect_labels.cpu().tolist())
                    
                    del outputs
                    
                except Exception as e:
                    log.info(f"验证批次 {batch_idx} 时出错: {e}")
                    continue
        
        # 计算平均loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 计算CER（过滤空文本）
        valid_pairs = [(p.strip(), r.strip()) for p, r in zip(all_preds, all_refs) 
                       if r.strip()]
        if valid_pairs:
            valid_preds, valid_refs = zip(*valid_pairs)
            val_cer = compute_cer(list(valid_refs), list(valid_preds))
        else:
            val_cer = 1.0
        
        # 计算方言分类准确率
        dialect_acc = 0.0
        if all_dialect_preds and all_dialect_refs:
            correct = sum(1 for p, r in zip(all_dialect_preds, all_dialect_refs) 
                         if p == r and r >= 0)
            total_cls = sum(1 for r in all_dialect_refs if r >= 0)
            dialect_acc = correct / total_cls if total_cls > 0 else 0.0
        
        # 计算分方言CER
        per_dialect_cer = {}
        if all_dialect_refs:
            from collections import defaultdict
            dialect_preds_map = defaultdict(list)
            dialect_refs_map = defaultdict(list)
            for pred, ref, d_id in zip(all_preds, all_refs, all_dialect_refs):
                if d_id >= 0 and ref.strip():
                    d_name = self.DIALECT_NAMES.get(d_id, f'方言{d_id}')
                    dialect_preds_map[d_name].append(pred.strip())
                    dialect_refs_map[d_name].append(ref.strip())
            
            for d_name in dialect_preds_map:
                d_preds = dialect_preds_map[d_name]
                d_refs = dialect_refs_map[d_name]
                if d_preds and d_refs:
                    try:
                        per_dialect_cer[d_name] = compute_cer(d_refs, d_preds)
                    except Exception:
                        per_dialect_cer[d_name] = -1.0
        
        # 计算 Macro CER（各方言等权平均，避免验证集不均衡偏差）
        macro_cer = val_cer  # 默认回退到 micro CER
        if per_dialect_cer:
            valid_cers = [c for c in per_dialect_cer.values() if c >= 0]
            if valid_cers:
                macro_cer = sum(valid_cers) / len(valid_cers)
        
        # 打印验证结果
        log.info(f"{'='*60}")
        log.info(f"验证结果:")
        log.info(f"  验证损失:       {avg_loss:.4f}")
        log.info(f"  Micro CER:      {val_cer:.4f} ({val_cer*100:.2f}%)")
        log.info(f"  Macro CER:      {macro_cer:.4f} ({macro_cer*100:.2f}%)  ← 早停判据")
        log.info(f"  方言分类准确率: {dialect_acc:.4f} ({dialect_acc*100:.2f}%)")
        
        if per_dialect_cer:
            log.info(f"  分方言CER:")
            for d_name, d_cer in sorted(per_dialect_cer.items(), key=lambda x: x[1]):
                log.info(f"    {d_name}: {d_cer:.4f} ({d_cer*100:.2f}%)")
        
        # 打印几个预测示例
        if valid_pairs:
            log.info(f"  预测示例 (前3条):")
            for i, (pred, ref) in enumerate(valid_pairs[:3]):
                log.info(f"    [{i+1}] 参考: {ref[:50]}")
                log.info(f"    [{i+1}] 预测: {pred[:50]}")
        
        log.info(f"{'='*60}")
        
        self.model.train()
        return {
            'val_loss': avg_loss,
            'val_cer': val_cer,
            'macro_cer': macro_cer,
            'dialect_accuracy': dialect_acc,
            'per_dialect_cer': per_dialect_cer,
        }
    
    def save_model(self, save_path):
        """保存模型和方言配置"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        # 保存方言增强配置（评估时需要加载）
        dialect_config_path = os.path.join(save_path, "dialect_config.json")
        with open(dialect_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.model.dialect_config, f, ensure_ascii=False, indent=2)
        self.logger.info(f"模型已保存到: {save_path}")
    
    def save_checkpoint(self, save_path, optimizer, scheduler, scaler,
                        epoch, global_step, best_val_cer, best_val_loss,
                        patience_counter, csv_path, epoch_completed=True):
        """保存完整训练checkpoint（模型 + 优化器 + 调度器 + 训练状态），用于断点续训
        
        Args:
            epoch_completed: 当前epoch是否已完成。
                True  = epoch正常结束后保存（续训从下一个epoch开始）
                False = epoch中途中断保存（续训从当前epoch头重新开始）
        """
        self.save_model(save_path)
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'best_val_cer': best_val_cer,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'csv_path': csv_path,
            'freeze_strategy': self.freeze_strategy,
            'epoch_completed': epoch_completed,
            'dialect_loss_weights': self.dialect_loss_weights.cpu() if hasattr(self, 'dialect_loss_weights') else None,
            # 保存随机状态，确保续训时数据顺序和随机增强与连续训练一致
            'rng_python': random.getstate(),
            'rng_numpy': np.random.get_state(),
            'rng_torch': torch.random.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        ckpt_path = os.path.join(save_path, "training_state.pt")
        torch.save(checkpoint, ckpt_path)
        self.logger.info(f"Checkpoint已保存到: {save_path} (epoch {epoch+1}, step {global_step})")
    
    def load_checkpoint(self, ckpt_path, optimizer, scheduler, scaler):
        """从checkpoint恢复训练状态"""
        state_path = os.path.join(ckpt_path, "training_state.pt")
        if not os.path.exists(state_path):
            self.logger.info(f"未找到训练状态文件: {state_path}")
            return None
        
        checkpoint = torch.load(state_path, map_location=self.device, weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 恢复随机状态，确保续训时数据顺序与连续训练一致
        if 'rng_python' in checkpoint:
            random.setstate(checkpoint['rng_python'])
        if 'rng_numpy' in checkpoint:
            np.random.set_state(checkpoint['rng_numpy'])
        if 'rng_torch' in checkpoint:
            rng_torch = checkpoint['rng_torch']
            if isinstance(rng_torch, torch.Tensor):
                rng_torch = rng_torch.cpu().byte()
            else:
                rng_torch = torch.ByteTensor(rng_torch)
            torch.random.set_rng_state(rng_torch)
        if 'rng_cuda' in checkpoint and checkpoint['rng_cuda'] is not None and torch.cuda.is_available():
            rng_cuda = checkpoint['rng_cuda']
            if isinstance(rng_cuda, torch.Tensor):
                rng_cuda = rng_cuda.cpu().byte()
            else:
                rng_cuda = torch.ByteTensor(rng_cuda)
            torch.cuda.set_rng_state(rng_cuda)
        
        self.logger.info(f"✓ 从checkpoint恢复: epoch {checkpoint['epoch']+1}, "
                         f"step {checkpoint['global_step']}, "
                         f"best CER: {checkpoint['best_val_cer']:.4f}")
        if 'rng_python' in checkpoint:
            self.logger.info("✓ 随机状态已恢复（数据顺序将与连续训练一致）")
        return checkpoint


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="方言模型训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训的checkpoint路径，如 ./dialect_model_interrupted 或 ./dialect_model_checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证实验可复现（默认42）")
    args = parser.parse_args()

    # ========== 设置全局随机种子 ==========
    # 注意：续训时随机状态会被 load_checkpoint 覆盖，此处 set_seed 仅确保
    # 模型初始化阶段的确定性（如新增模块权重初始化）
    set_seed(args.seed)
    print(f"随机种子: {args.seed}")
    
    # ========== 断点续训检测 ==========
    resume_from = args.resume
    
    # 如果没指定 --resume，自动检测是否存在中断checkpoint
    if resume_from is None:
        for auto_ckpt in ["./dialect_model_interrupted", "./dialect_model_checkpoint"]:
            if os.path.exists(os.path.join(auto_ckpt, "training_state.pt")):
                print(f"检测到未完成的checkpoint: {auto_ckpt}")
                print(f"如需续训请运行: python dialect_fine_tune.py --resume {auto_ckpt}")
                print(f"如需从头训练请先删除该目录，或忽略此提示\n")
                break
    
    # 优先使用预处理数据（更快），否则回退到在线加载
    # 注意：训练时用 dev 集做验证（early stopping / 选最优模型），test 集只在最终评估时使用
    use_preprocessed = (
        os.path.isdir("./preprocessed_data/train") and 
        os.path.isdir("./preprocessed_data/dev")
    )
    
    if use_preprocessed:
        print("检测到预处理数据，使用快速加载模式")
        train_data = "./preprocessed_data/train"
        val_data = "./preprocessed_data/dev"
    elif os.path.exists("./split_data/train_set.json"):
        print("未找到预处理数据，使用在线加载（较慢）")
        print("提示: 运行 python preprocess_data.py 可预处理数据，训练速度提升 3-5 倍\n")
        with open("./split_data/train_set.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        # 训练验证使用 dev 集，而非 test 集
        dev_path = "./split_data/dev_set.json"
        if not os.path.exists(dev_path):
            # 向后兼容：如果还没有 dev_set.json，回退到 test_set.json
            print("⚠ 未找到 dev_set.json，回退使用 test_set.json 作为验证集")
            print("  建议重新运行 fen.py 生成 train/dev/test 三份数据\n")
            dev_path = "./split_data/test_set.json"
        with open(dev_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}（来自 {os.path.basename(dev_path)}）")
    else:
        print("错误: 未找到训练数据，请先运行 fen.py 分割数据")
        return
    
    # 方言增强配置
    dialect_config = {
        'use_adapter': True,                 
        'use_variation_layer': False,
        'bottleneck_size': 256,
        'num_layers': 25,              # wav2vec2-large: 1 CNN + 24 Transformer
        'kernel_sizes': [3, 11, 21],   
        'num_attention_heads': 8,
        'dropout': 0.3,
        'num_dialects': 7,
        'dialect_loss_weight': 0.005,    # 方言分类对主任务的影响
        'ctc_temperature': 1.0,        # 基线使用常规温度
    }
    
    # 确定模型加载路径：续训时从checkpoint加载，否则从预训练模型加载
    if resume_from is not None and os.path.exists(resume_from):
        model_path = resume_from
        print(f"断点续训模式: 从 {resume_from} 恢复模型权重和训练状态")
        # 续训时使用 checkpoint 中保存的配置，避免结构不匹配
        saved_config_path = os.path.join(resume_from, "dialect_config.json")
        if os.path.exists(saved_config_path):
            with open(saved_config_path, 'r', encoding='utf-8') as f:
                dialect_config = json.load(f)
            print(f"使用checkpoint中保存的模型配置（确保结构一致）")
    else:
        model_path = "./origin_model"
    
    # 创建训练器（Baseline 使用全量微调/冻结策略为 full）
    trainer = DialectModelTrainer(
        model_path=model_path,
        dialect_config=dialect_config,
        freeze_strategy="progressive"  # Baseline: 冻结参数不变
    )
    
    # 开始训练
    try:
        trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=30,                     # 增至30：cosine衰减更慢，难方言需更多epoch
            batch_size=4,
            learning_rate=2e-5,            # 调整为2e-5，略高以配合解冻更多编码器层
            augment_prob=0.0,             # 离线增强场景：训练时关闭在线增强（无增强实验）
            warmup_steps=500,              # 增大warmup以稳定较大学习率的初始阶段
            gradient_accumulation_steps=4, # 有效batch=16，梯度更稳定
            early_stop_patience=10,        # 从7增至10：cosine schedule后期LR较高，给更多探索空间
            save_path="./dialect_model",
            resume_from=resume_from
        )
    except Exception as e:
        trainer.logger.info(f"\n训练过程中发生错误: {e}")
        import traceback
        trainer.logger.info(traceback.format_exc())


if __name__ == "__main__":
    main()
