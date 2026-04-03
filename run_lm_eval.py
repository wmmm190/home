"""快速运行 LM 评估并保存结果到文件"""
import sys
import os
import signal

# 忽略中断信号
signal.signal(signal.SIGINT, signal.SIG_IGN)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 重定向输出到文件
log_file = open('logs/lm_eval_sampled.txt', 'w', encoding='utf-8')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import json
import time
import numpy as np
import torch
from scipy.special import log_softmax
from jiwer import cer as compute_cer

# 导入我们的模块
sys.path.insert(0, '.')
from tools.eval_with_lm import (
    KenLMScorer, CTCBeamDecoder, load_model_and_processor,
    get_logits, greedy_decode, clean_text, DIALECT_NAMES
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples = 350  # 每方言约 50 条
    beam_width = 10
    lm_weight = 0.5

    print(f"设备: {device}")
    print(f"样本数: {max_samples}")

    # 加载模型
    model, processor = load_model_and_processor("./dialect_model_best", device)
    vocab = processor.tokenizer.get_vocab()
    blank_id = vocab.get('[PAD]', 0)
    print(f"vocab: {len(vocab)}, blank: {blank_id}")

    # 加载测试数据
    with open("./split_data/test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 条")

    # 从每个方言均匀采样
    import random
    random.seed(42)
    from collections import defaultdict as dd
    by_dialect = dd(list)
    for item in test_data:
        by_dialect[item.get("dialect_id", -1)].append(item)
    
    per_dialect_n = max_samples // max(len(by_dialect), 1)
    sampled = []
    for did in sorted(by_dialect.keys()):
        items = by_dialect[did]
        random.shuffle(items)
        sampled.extend(items[:per_dialect_n])
    random.shuffle(sampled)
    test_data = sampled
    print(f"采样后: {len(test_data)} 条 (每方言 ~{per_dialect_n} 条)")

    # 加载 KenLM
    lm_path = "./lm/char_5gram.arpa"
    if not os.path.exists(lm_path):
        raise FileNotFoundError(f"未找到 KenLM 文件: {lm_path}")
    lm = KenLMScorer(lm_path)

    # 创建解码器
    decoder_lm = CTCBeamDecoder(vocab=vocab, blank_id=blank_id, lm=lm,
                                beam_width=beam_width, alpha=lm_weight, beta=0.0)
    decoder_no_lm = CTCBeamDecoder(vocab=vocab, blank_id=blank_id, lm=None,
                                    beam_width=beam_width, alpha=0, beta=0)

    # 评估
    greedy_preds, beam_preds, beam_lm_preds, refs, dids = [], [], [], [], []
    n = min(max_samples, len(test_data))
    t0 = time.time()

    for i in range(n):
        item = test_data[i]
        try:
            ref = clean_text(item.get("sentence", ""))
            if not ref or not os.path.exists(item.get("path", "")):
                continue

            logits = get_logits(model, processor, item["path"], device)
            lp = log_softmax(logits, axis=-1)

            g = clean_text(greedy_decode(logits, processor))
            b = clean_text(decoder_no_lm.decode(lp))
            bl = clean_text(decoder_lm.decode(lp))

            greedy_preds.append(g)
            beam_preds.append(b)
            beam_lm_preds.append(bl)
            refs.append(ref)
            dids.append(item.get("dialect_id", -1))

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                eta = (n - i - 1) / speed if speed > 0 else 0
                print(f"  [{i+1}/{n}] {speed:.1f} 条/秒, ETA {eta:.0f}s")

        except Exception as e:
            if i < 3:
                print(f"  错误 {i}: {e}")

    elapsed = time.time() - t0
    print(f"\n评估完成: {len(refs)} 有效, {elapsed:.1f}s")

    # 计算 CER
    from collections import defaultdict

    def calc_cer_grouped(preds):
        valid = [(p, r) for p, r in zip(preds, refs) if p and r]
        if not valid:
            return 1.0, {}
        vp, vr = zip(*valid)
        overall = compute_cer(list(vr), list(vp))

        per_d = {}
        dp, dr = defaultdict(list), defaultdict(list)
        for p, r, d in zip(preds, refs, dids):
            if p and r and d >= 0:
                dn = DIALECT_NAMES.get(d, f"方言{d}")
                dp[dn].append(p)
                dr[dn].append(r)
        for dn in dp:
            per_d[dn] = compute_cer(dr[dn], dp[dn])
        return overall, per_d

    g_cer, g_per = calc_cer_grouped(greedy_preds)
    b_cer, b_per = calc_cer_grouped(beam_preds)
    bl_cer, bl_per = calc_cer_grouped(beam_lm_preds)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"{'策略':<18} {'CER':>8}")
    print(f"{'-'*30}")
    print(f"{'Greedy (基线)':<18} {g_cer*100:>7.2f}%")
    print(f"{'Beam (无LM)':<18} {b_cer*100:>7.2f}%")
    print(f"{'Beam + LM':<18} {bl_cer*100:>7.2f}%")

    delta = (bl_cer - g_cer) * 100
    rel = (bl_cer - g_cer) / g_cer * 100 if g_cer > 0 else 0
    print(f"\nLM 收益: {delta:+.2f}% 绝对, {rel:+.1f}% 相对")

    print(f"\n{'='*60}")
    print("分方言 CER:")
    print(f"{'方言':<10} {'Greedy':>8} {'Beam':>8} {'Beam+LM':>8} {'Delta':>8}")
    print(f"{'-'*50}")
    for dn in sorted(set(list(g_per.keys()) + list(bl_per.keys()))):
        g_v = g_per.get(dn, -1)
        b_v = b_per.get(dn, -1)
        bl_v = bl_per.get(dn, -1)
        d = (bl_v - g_v) * 100 if g_v >= 0 and bl_v >= 0 else 0
        print(f"{dn:<10} {g_v*100:>7.2f}% {b_v*100:>7.2f}% {bl_v*100:>7.2f}% {d:>+7.2f}%")

    print(f"{'='*60}")

    # 保存
    results = {
        "greedy_cer": g_cer, "beam_cer": b_cer, "beam_lm_cer": bl_cer,
        "greedy_per_dialect": g_per, "beam_per_dialect": b_per, "beam_lm_per_dialect": bl_per,
        "params": {"beam_width": beam_width, "lm_weight": lm_weight, "samples": len(refs)}
    }
    with open("logs/lm_eval_results_sampled.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("结果已保存到 logs/lm_eval_results_sampled.json")

if __name__ == "__main__":
    main()
