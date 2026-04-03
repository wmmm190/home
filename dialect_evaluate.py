"""
方言模型评估脚本
评估方言增强模型在测试集上的性能
支持：CTC识别评估（CER）+ 方言分类准确率 + 分方言统计
"""

import json
import torch
import torch.nn.functional as F
import librosa
import re
import warnings
import numpy as np
from transformers import Wav2Vec2Processor
from jiwer import cer
from collections import defaultdict
import os
import sys
from scipy.special import log_softmax

from dialect_model import EnhancedWav2Vec2ForDialect

# 导入 LM + Beam 解码模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from tools.eval_with_lm import KenLMScorer, CTCBeamDecoder

warnings.filterwarnings("ignore")

# 默认 LM 路径和参数
DEFAULT_LM_PATH = os.path.join(PROJECT_ROOT, "lm", "char_5gram.arpa")
DEFAULT_BEAM_WIDTH = 10
DEFAULT_LM_WEIGHT = 0.5
DEFAULT_WORD_SCORE = 0.0

# 方言名称映射
DIALECT_NAMES = {
    0: '武汉话', 1: '南昌话', 2: '上海话', 3: '四川话',
    4: '天津话', 5: '长沙话', 6: '郑州话'
}


def clean_text(text):
    """与 tools/eval_with_lm.py 保持一致：仅保留中文字符"""
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text.strip()


class DialectEvaluator:
    """方言模型评估器（支持 Greedy / Beam+LM 解码）"""
    
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu",
                 lm_path=None, beam_width=None, lm_weight=None, word_score=None):
        """
        Args:
            model_path: 模型路径
            device: 评估设备
            lm_path: 语言模型路径（默认自动检测 lm/char_5gram.arpa）
            beam_width: beam search 宽度（默认 10）
            lm_weight: LM 权重 alpha（默认 0.5）
            word_score: 长度奖励 beta（默认 0.0）
        """
        self.device = device
        self.lm = None
        self.beam_decoder = None
        print(f"使用设备: {device}")
        
        # 加载处理器和模型
        print(f"从 {model_path} 加载模型...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            
            # 尝试加载增强模型，如果失败则加载标准模型
            try:
                from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
                config = Wav2Vec2Config.from_pretrained(model_path)
                
                # 检查是否有方言增强配置
                if os.path.exists(os.path.join(model_path, "dialect_config.json")):
                    with open(os.path.join(model_path, "dialect_config.json"), "r") as f:
                        dialect_config = json.load(f)
                    self.model = EnhancedWav2Vec2ForDialect(config, dialect_config)
                    
                    # 支持 safetensors 和 pytorch_model.bin 两种格式
                    weight_path_st = os.path.join(model_path, "model.safetensors")
                    weight_path_bin = os.path.join(model_path, "pytorch_model.bin")
                    if os.path.exists(weight_path_st):
                        import safetensors.torch
                        state_dict = safetensors.torch.load_file(weight_path_st)
                        self.model.load_state_dict(state_dict, strict=False)
                    elif os.path.exists(weight_path_bin):
                        state_dict = torch.load(weight_path_bin, map_location="cpu")
                        self.model.load_state_dict(state_dict, strict=False)
                    else:
                        raise FileNotFoundError(f"未找到模型权重文件: {model_path}")
                    print("✓ 加载增强型方言模型")
                else:
                    # 标准模型
                    self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
                    print("✓ 加载标准Wav2Vec2模型")
                    
            except Exception as e:
                print(f"作为增强模型加载失败，尝试标准模型: {e}")
                from transformers import Wav2Vec2ForCTC
                self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
                print("✓ 加载标准Wav2Vec2模型")
        
        self.model.eval()
        self.model.to(device)
        
        # ---------- 初始化 LM + Beam 解码器 ----------
        _lm_path = lm_path if lm_path is not None else DEFAULT_LM_PATH
        _beam_width = beam_width if beam_width is not None else DEFAULT_BEAM_WIDTH
        _lm_weight = lm_weight if lm_weight is not None else DEFAULT_LM_WEIGHT
        _word_score = word_score if word_score is not None else DEFAULT_WORD_SCORE
        
        if os.path.exists(_lm_path):
            try:
                if not (_lm_path.endswith('.arpa') or _lm_path.endswith('.bin')):
                    raise ValueError("仅支持 KenLM 文件 (.arpa/.bin)")
                self.lm = KenLMScorer(_lm_path)
                vocab = self.processor.tokenizer.get_vocab()
                blank_id = vocab.get('[PAD]', 0)
                self.beam_decoder = CTCBeamDecoder(
                    vocab=vocab, blank_id=blank_id, lm=self.lm,
                    beam_width=_beam_width, alpha=_lm_weight, beta=_word_score
                )
                print(f"✓ Beam+LM 解码器已初始化 (beam={_beam_width}, alpha={_lm_weight}, beta={_word_score})")
            except Exception as e:
                print(f"⚠ 加载 LM 失败，将使用 Greedy 解码: {e}")
                self.lm = None
                self.beam_decoder = None
        else:
            print(f"⚠ 未找到 LM 文件 ({_lm_path})，将使用 Greedy 解码")
    
    def transcribe(self, audio_path):
        """
        转录单个音频文件，同时返回方言分类结果
        
        当 LM 可用时自动使用 Beam+LM 解码；否则回退为 Greedy。
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            transcription: 转录文本
            dialect_pred: 预测的方言ID（若模型支持，否则-1）
        """
        # 加载音频
        speech, _ = librosa.load(audio_path, sr=16000)
        
        # 预处理
        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # 移动到GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 方言分类：直接从模型输出获取
            dialect_pred = -1
            if hasattr(outputs, 'dialect_logits') and outputs.dialect_logits is not None:
                dialect_pred = outputs.dialect_logits.argmax(dim=-1).item()
        
        # ---------- 解码策略：Beam+LM 优先，Greedy 回退 ----------
        if self.beam_decoder is not None:
            # Beam + LM 解码
            lp = log_softmax(logits[0].cpu().float().numpy(), axis=-1)
            transcription = self.beam_decoder.decode(lp)
        else:
            # 与 tools/eval_with_lm.py 一致：decode + skip_special_tokens
            pred_ids = torch.argmax(logits[0], dim=-1).cpu().numpy()
            transcription = self.processor.decode(pred_ids, skip_special_tokens=True)
        
        return transcription, dialect_pred
    
    def evaluate_dataset(self, test_data, show_examples=5):
        """
        评估整个测试集
        
        Args:
            test_data: 测试数据列表
            show_examples: 显示多少个示例
            
        Returns:
            results: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"开始评估测试集 (共 {len(test_data)} 个样本)")
        print(f"{'='*60}\n")
        
        predictions = []
        references = []
        examples = []
        dialect_preds = []     # 方言分类预测
        dialect_refs = []      # 方言真实标签
        per_dialect_preds = defaultdict(list)  # 分方言统计
        per_dialect_refs = defaultdict(list)
        
        for i, item in enumerate(test_data):
            try:
                # 转录
                pred_text, dialect_pred = self.transcribe(item['path'])
                ref_text = item['sentence']
                dialect_ref = item.get('dialect_id', -1)
                dialect_name = item.get('dialect', DIALECT_NAMES.get(dialect_ref, '未知'))
                
                # 清洗文本
                clean_pred = clean_text(pred_text)
                clean_ref = clean_text(ref_text)
                
                predictions.append(clean_pred)
                references.append(clean_ref)
                
                # 方言分类统计
                if dialect_ref >= 0:
                    dialect_preds.append(dialect_pred)
                    dialect_refs.append(dialect_ref)
                    per_dialect_preds[dialect_name].append(clean_pred)
                    per_dialect_refs[dialect_name].append(clean_ref)
                
                # 保存示例
                if len(examples) < show_examples:
                    examples.append({
                        'audio': item['path'],
                        'reference': clean_ref,
                        'prediction': clean_pred,
                        'dialect': dialect_name,
                        'dialect_pred': DIALECT_NAMES.get(dialect_pred, '未知')
                    })
                
                # 进度显示
                if (i + 1) % 50 == 0:
                    print(f"已处理 {i + 1}/{len(test_data)} 个样本")
                    
            except Exception as e:
                print(f"处理样本 {i} 时出错: {str(e)}")
                # 添加空结果以保持对齐
                predictions.append("")
                references.append(item['sentence'])
        
        # 计算指标
        print(f"\n{'='*60}")
        print("计算评估指标...")
        print(f"{'='*60}\n")
        
        # 过滤空预测
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
        if len(valid_pairs) < len(predictions):
            print(f"警告: {len(predictions) - len(valid_pairs)} 个样本因错误被跳过")
        
        valid_preds, valid_refs = zip(*valid_pairs) if valid_pairs else ([], [])
        valid_preds, valid_refs = list(valid_preds), list(valid_refs)
        
        # 计算 CER
        if valid_preds and valid_refs:
            cer_score = cer(valid_refs, valid_preds)
        else:
            cer_score = 1.0
        
        # 计算字符级统计
        char_stats = self._compute_char_stats(valid_refs, valid_preds)
        
        results = {
            'cer': cer_score,
            'macro_cer': cer_score,
            'total_samples': len(test_data),
            'valid_samples': len(valid_pairs),
            'char_stats': char_stats,
            'examples': examples
        }
        
        # 方言分类准确率
        if dialect_preds and dialect_refs:
            correct = sum(1 for p, r in zip(dialect_preds, dialect_refs) if p == r)
            total_cls = len(dialect_refs)
            results['dialect_accuracy'] = correct / total_cls if total_cls > 0 else 0
            results['dialect_total'] = total_cls
            results['dialect_correct'] = correct
            
            # 混淆矩阵
            num_dialects = max(max(dialect_refs), max(dialect_preds)) + 1 if dialect_preds else 7
            confusion = [[0] * num_dialects for _ in range(num_dialects)]
            for ref_d, pred_d in zip(dialect_refs, dialect_preds):
                if 0 <= ref_d < num_dialects and 0 <= pred_d < num_dialects:
                    confusion[ref_d][pred_d] += 1
            results['dialect_confusion'] = confusion
        
        # 分方言 CER
        if per_dialect_preds:
            per_dialect_cer = {}
            for dialect_name in per_dialect_preds:
                d_preds = per_dialect_preds[dialect_name]
                d_refs = per_dialect_refs[dialect_name]
                valid_d = [(p, r) for p, r in zip(d_preds, d_refs) if p and r]
                if valid_d:
                    vp, vr = zip(*valid_d)
                    per_dialect_cer[dialect_name] = cer(list(vr), list(vp))
                else:
                    per_dialect_cer[dialect_name] = 1.0
            results['per_dialect_cer'] = per_dialect_cer
            if per_dialect_cer:
                results['macro_cer'] = float(np.mean(list(per_dialect_cer.values())))
        
        return results
    
    def _compute_char_stats(self, references, predictions):
        """计算字符级统计信息"""
        total_chars = sum(len(ref) for ref in references)
        total_errors = 0
        
        for ref, pred in zip(references, predictions):
            # 简单的编辑距离计算
            errors = self._edit_distance(ref, pred)
            total_errors += errors
        
        accuracy = max(0, 1 - (total_errors / total_chars)) if total_chars > 0 else 0
        
        return {
            'total_chars': total_chars,
            'total_errors': total_errors,
            'accuracy': accuracy
        }
    
    def _edit_distance(self, s1, s2):
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def print_results(self, results):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print("评估结果")
        print(f"{'='*60}")
        print(f"总样本数:        {results['total_samples']}")
        print(f"有效样本数:      {results['valid_samples']}")
        print(f"{'-'*60}")
        print(f"CER (字错误率):  {results['cer']:.4f} ({results['cer']*100:.2f}%)")
        print(f"Macro CER:       {results.get('macro_cer', results['cer']):.4f} ({results.get('macro_cer', results['cer'])*100:.2f}%)")
        print(f"{'-'*60}")
        print(f"字符统计:")
        print(f"  总字符数:      {results['char_stats']['total_chars']}")
        print(f"  错误字符数:    {results['char_stats']['total_errors']}")
        print(f"  字符准确率:    {results['char_stats']['accuracy']:.4f} ({results['char_stats']['accuracy']*100:.2f}%)")
        print(f"{'='*60}\n")
        
        # 方言分类结果
        if 'dialect_accuracy' in results:
            print(f"{'='*60}")
            print("方言分类结果")
            print(f"{'='*60}")
            print(f"分类准确率:      {results['dialect_accuracy']:.4f} ({results['dialect_accuracy']*100:.2f}%)")
            print(f"正确/总数:        {results['dialect_correct']}/{results['dialect_total']}")
            
            # 混淆矩阵
            if 'dialect_confusion' in results:
                print(f"\n混淆矩阵 (行=真实, 列=预测):")
                cm = results['dialect_confusion']
                header = "         " + " ".join(f"{DIALECT_NAMES.get(j, f'D{j}'):>6}" for j in range(len(cm)))
                print(header)
                for i, row in enumerate(cm):
                    name = DIALECT_NAMES.get(i, f'D{i}')
                    row_str = " ".join(f"{v:>6}" for v in row)
                    print(f"{name:>8} {row_str}")
            print(f"{'='*60}\n")
        
        # 分方言 CER
        if 'per_dialect_cer' in results:
            print(f"{'='*60}")
            print("分方言 CER")
            print(f"{'='*60}")
            for dialect_name, d_cer in sorted(results['per_dialect_cer'].items(), key=lambda x: x[1]):
                print(f"  {dialect_name:>8}: CER = {d_cer:.4f} ({d_cer*100:.2f}%)")
            print(f"{'='*60}\n")
        
        # 显示示例
        if results['examples']:
            print(f"\n{'='*60}")
            print("转录示例")
            print(f"{'='*60}")
            for i, example in enumerate(results['examples']):
                print(f"\n示例 {i+1}:")
                print(f"音频: {example['audio']}")
                print(f"方言: {example.get('dialect', '未知')} (预测: {example.get('dialect_pred', '未知')})")
                print(f"参考: {example['reference']}")
                print(f"预测: {example['prediction']}")
                print(f"{'-'*60}")


def compare_models(model_paths, test_data_path, model_names=None,
                   lm_path=None, beam_width=None, lm_weight=None, word_score=None):
    """
    比较多个模型的性能
    
    Args:
        model_paths: 模型路径列表
        test_data_path: 测试数据路径
        model_names: 模型名称列表（可选）
        lm_path: 语言模型路径
        beam_width: beam search 宽度
        lm_weight: LM 权重
        word_score: 长度奖励
    """
    # 加载测试数据
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    if model_names is None:
        model_names = [f"模型{i+1}" for i in range(len(model_paths))]
    
    results_list = []
    
    # 评估每个模型
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n\n{'#'*60}")
        print(f"# 评估模型: {model_name}")
        print(f"# 路径: {model_path}")
        print(f"{'#'*60}")
        
        evaluator = DialectEvaluator(
            model_path, lm_path=lm_path, beam_width=beam_width,
            lm_weight=lm_weight, word_score=word_score
        )
        results = evaluator.evaluate_dataset(test_data, show_examples=3)
        results['model_name'] = model_name
        results['model_path'] = model_path
        results_list.append(results)
        
        evaluator.print_results(results)
    
    # 打印比较结果
    print(f"\n\n{'='*60}")
    print("模型性能对比")
    print(f"{'='*60}")
    print(f"{'模型名称':<20} {'CER':>10} {'MacroCER':>10} {'准确率':>10}")
    print(f"{'-'*60}")
    
    for results in results_list:
        print(f"{results['model_name']:<20} "
              f"{results['cer']*100:>9.2f}% "
              f"{results.get('macro_cer', results['cer'])*100:>9.2f}% "
              f"{results['char_stats']['accuracy']*100:>9.2f}%")
    
    print(f"{'='*60}\n")
    
    return results_list


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="方言模型评估（支持 Beam+LM 解码）")
    parser.add_argument("--model", type=str, default="./dialect_model_best",
                       help="模型路径")
    parser.add_argument("--test_data", type=str, default="./split_data/test_set.json",
                       help="测试数据路径")
    parser.add_argument("--compare", action="store_true",
                       help="比较多个模型")
    parser.add_argument("--models", nargs="+",
                       help="要比较的模型路径列表")
    parser.add_argument("--names", nargs="+",
                       help="模型名称列表")
    # LM + Beam 解码参数
    parser.add_argument("--lm-path", type=str, default=None,
                       help=f"语言模型路径（默认自动检测 {DEFAULT_LM_PATH}）")
    parser.add_argument("--beam-width", type=int, default=None,
                       help=f"Beam 宽度（默认 {DEFAULT_BEAM_WIDTH}）")
    parser.add_argument("--lm-weight", type=float, default=None,
                       help=f"LM 权重 alpha（默认 {DEFAULT_LM_WEIGHT}）")
    parser.add_argument("--word-score", type=float, default=None,
                       help=f"长度奖励 beta（默认 {DEFAULT_WORD_SCORE}）")
    parser.add_argument("--no-lm", action="store_true",
                       help="禁用 LM，使用纯 Greedy 解码")
    
    args = parser.parse_args()
    
    # 检查测试数据
    if not os.path.exists(args.test_data):
        print(f"错误: 测试数据不存在: {args.test_data}")
        return
    
    # LM 参数：--no-lm 时传入一个不存在的路径使 LM 不加载
    lm_path = "__disabled__" if args.no_lm else args.lm_path
    
    if args.compare and args.models:
        # 比较模式
        compare_models(args.models, args.test_data, args.names,
                       lm_path=lm_path, beam_width=args.beam_width,
                       lm_weight=args.lm_weight, word_score=args.word_score)
    else:
        # 单模型评估
        if not os.path.exists(args.model):
            print(f"错误: 模型不存在: {args.model}")
            return
        
        # 加载测试数据
        with open(args.test_data, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # 评估
        evaluator = DialectEvaluator(
            args.model, lm_path=lm_path, beam_width=args.beam_width,
            lm_weight=args.lm_weight, word_score=args.word_score
        )
        results = evaluator.evaluate_dataset(test_data, show_examples=5)
        evaluator.print_results(results)
        
        # 保存结果
        results_path = f"{args.model}_evaluation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            # 移除不能序列化的部分
            save_results = {k: v for k, v in results.items() if k != 'examples'}
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 评估结果已保存到: {results_path}")


if __name__ == "__main__":
    # 如果没有命令行参数，使用默认评估
    import sys
    
    if len(sys.argv) == 1:
        print("使用默认配置进行评估...")
        print("提示: 使用 --help 查看更多选项\n")
        
        # 默认评估方言模型
        test_data_path = "./split_data/test_set.json"
        
        if not os.path.exists(test_data_path):
            print(f"错误: 测试数据不存在: {test_data_path}")
            print("请先运行 fen.py 分割数据")
            sys.exit(1)
        
        # 查找可用模型
        available_models = []
        model_names = []
        
        for model_dir in ["./dialect_model_best", "./dialect_model_final", 
                         "./fine_tuned_model_best", "./"]:
            if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
                available_models.append(model_dir)
                model_names.append(os.path.basename(model_dir) or "原始模型")
        
        if not available_models:
            print("错误: 未找到可用模型")
            sys.exit(1)
        
        print(f"找到 {len(available_models)} 个模型，开始比较评估...\n")
        compare_models(available_models, test_data_path, model_names)  # 自动检测 LM
    else:
        main()
