"""
构建字符级 KenLM n-gram 语言模型（ARPA 格式）

流程：
  1. 从 train_set.json 提取文本
  2. 检测重复并去重（避免 LM 过拟合特定句子）
  3. 用 Python 生成 ARPA 格式 n-gram 模型（绝对折扣 + 回退平滑）
  4. 用 kenlm.Model 验证可正常加载

为什么要去重？
  训练集中的大量重复文本会导致：
  - n-gram 计数严重偏向重复句子
  - LM 过度 "记忆" 特定短语而非学习通用语言模式
  - beam search 解码时不合理地偏好重复过的短语
  去重后 LM 更加平衡，对各种输入泛化更好。

使用方式：
  # 基本用法（去重 + 构建 5-gram）
  python tools/build_kenlm.py

  # 自定义参数
  python tools/build_kenlm.py --order 4 --prune 2 --output ./lm/char_4gram.arpa

  # 保留部分重复（每句最多 2 份）
  python tools/build_kenlm.py --max-copies 2

依赖：kenlm（仅用于验证）
"""

import os
import sys
import json
import re
import math
import argparse
import time
from collections import Counter, defaultdict

# ============================================================
# 文本提取与去重
# ============================================================

def clean_text_for_lm(text):
    """清洗文本，只保留中文字符"""
    return re.sub(r'[^\u4e00-\u9fa5]', '', text).strip()


def extract_and_dedup(train_json, max_copies=1):
    """
    从训练集 JSON 提取文本并去重

    Args:
        train_json: 训练集 JSON 路径
        max_copies: 每条唯一句子最多保留几份（1=完全去重）

    Returns:
        deduped: list of str（去重后的句子列表）
        stats: dict（去重统计信息）
    """
    print(f"读取训练数据: {train_json}")
    with open(train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_texts = []
    for item in data:
        text = clean_text_for_lm(item.get('sentence', ''))
        if len(text) >= 2:
            raw_texts.append(text)

    # 统计重复
    text_counts = Counter(raw_texts)
    n_unique = len(text_counts)
    n_total = len(raw_texts)
    n_dup = n_total - n_unique

    # 重复次数分布
    dup_dist = Counter()
    for _, count in text_counts.items():
        dup_dist[count] += 1

    # 重复最多的句子
    top_dups = [(t, c) for t, c in text_counts.most_common(20) if c > 1]

    # 应用 max_copies
    deduped = []
    for text, count in text_counts.items():
        copies = min(count, max_copies)
        deduped.extend([text] * copies)

    # 统计字符
    total_chars = sum(len(t) for t in deduped)
    vocab = set()
    for t in deduped:
        vocab.update(t)

    stats = {
        'total_raw': n_total,
        'unique': n_unique,
        'duplicated': n_dup,
        'dup_rate': n_dup / n_total * 100 if n_total > 0 else 0,
        'after_dedup': len(deduped),
        'total_chars': total_chars,
        'vocab_size': len(vocab),
        'dup_distribution': sorted(dup_dist.items()),
        'top_duplicates': top_dups[:10]
    }

    return deduped, stats


def print_dedup_stats(stats):
    """打印去重统计信息"""
    print(f"\n{'─'*55}")
    print("  去重统计")
    print(f"{'─'*55}")
    print(f"  原始句子数:     {stats['total_raw']:>8,}")
    print(f"  唯一句子数:     {stats['unique']:>8,}")
    print(f"  重复句子数:     {stats['duplicated']:>8,} ({stats['dup_rate']:.1f}%)")
    print(f"  去重后句子数:   {stats['after_dedup']:>8,}")
    print(f"  总字符数:       {stats['total_chars']:>8,}")
    print(f"  字符集大小:     {stats['vocab_size']:>8,}")

    print(f"\n  重复次数分布:")
    for count, n_sents in stats['dup_distribution'][:10]:
        label = f"出现{count}次" if count > 1 else "唯一"
        print(f"    {label}: {n_sents:,} 条句子")

    if stats['top_duplicates']:
        print(f"\n  重复最多的句子 (影响 LM 最大):")
        for i, (text, count) in enumerate(stats['top_duplicates'][:5], 1):
            display = text[:40] + "..." if len(text) > 40 else text
            print(f"    {i}. [{count:>4}次] {display}")
    print(f"{'─'*55}")


# ============================================================
# ARPA 格式 n-gram 模型生成器
# ============================================================

class ARPABuilder:
    """
    从字符序列构建 ARPA 格式 n-gram 模型

    使用绝对折扣 (Absolute Discounting) + Katz 回退
    适合字符级中文语言模型

    算法:
      - 高阶 n-gram: P_disc(w|h) = max(c(h,w) - d, 0) / c(h)
      - 预留概率: reserved(h) = d * T(h) / c(h)  (T = 唯一扩展数)
      - 回退权重: bow(h) = reserved(h) / (1 - Σ P_lower(w, seen))
      - 保证概率归一化
    """

    def __init__(self, order=5, discount=0.75, prune_thresholds=None):
        """
        Args:
            order: n-gram 最大阶数（5 对字符级通常够用）
            discount: 绝对折扣值 (0 < d < 1, 推荐 0.75)
            prune_thresholds: dict {n: threshold} 或 int
                count <= threshold 的 n-gram 被剪枝（仅对 order>=3 生效）
        """
        self.order = order
        self.discount = discount

        if prune_thresholds is None:
            self.prune_thresholds = {n: 0 for n in range(1, order + 1)}
        elif isinstance(prune_thresholds, int):
            pt = prune_thresholds
            self.prune_thresholds = {
                n: (pt if n >= 3 else 0) for n in range(1, order + 1)
            }
        else:
            self.prune_thresholds = prune_thresholds

    def build(self, sentences, output_path):
        """
        从文本句子列表构建 ARPA 文件

        Args:
            sentences: list of str（中文句子列表）
            output_path: 输出 .arpa 文件路径

        Returns:
            output_path
        """
        char_sentences = [list(s) for s in sentences if s]

        print(f"\n{'='*60}")
        print(f"构建 {self.order}-gram ARPA 模型")
        print(f"  训练句子数: {len(char_sentences):,}")
        print(f"  折扣值: {self.discount}")
        print(f"  剪枝: {self.prune_thresholds}")
        print(f"{'='*60}")

        t0 = time.time()

        # Step 1: 统计 n-gram
        print("\n[1/4] 统计 n-gram 计数...")
        ngram_counts, contexts = self._count_ngrams(char_sentences)

        # Step 2: 剪枝
        print("\n[2/4] 剪枝低频 n-gram...")
        self._prune(ngram_counts, contexts)

        # Step 3: 计算概率和回退权重
        print("\n[3/4] 计算概率和回退权重...")
        entries = self._compute_probs(ngram_counts, contexts)

        # Step 4: 写入 ARPA
        print("\n[4/4] 写入 ARPA 文件...")
        self._write_arpa(entries, output_path)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        total_entries = sum(len(entries[n]) for n in entries)
        print(f"\n  完成! 耗时 {elapsed:.1f}s")
        print(f"  总条目数: {total_entries:,}")
        print(f"  文件大小: {size_mb:.1f} MB")
        print(f"  输出路径: {output_path}")

        return output_path

    def _count_ngrams(self, char_sentences):
        """统计所有阶数的 n-gram"""
        BOS = '<s>'
        EOS = '</s>'
        order = self.order

        ngram_counts = {n: Counter() for n in range(1, order + 1)}

        for sent_chars in char_sentences:
            if not sent_chars:
                continue
            padded = [BOS] * (order - 1) + sent_chars + [EOS]
            for n in range(1, order + 1):
                for i in range(n - 1, len(padded)):
                    ngram = tuple(padded[i - n + 1: i + 1])
                    ngram_counts[n][ngram] += 1

        # 构建上下文信息
        contexts = {}
        for n in range(1, order + 1):
            ctx_info = {}
            for ngram, count in ngram_counts[n].items():
                ctx = ngram[:-1]
                word = ngram[-1]
                if ctx not in ctx_info:
                    ctx_info[ctx] = {'total': 0, 'words': {}}
                ctx_info[ctx]['total'] += count
                ctx_info[ctx]['words'][word] = count
            contexts[n] = ctx_info

        for n in range(1, order + 1):
            print(f"  {n}-gram: {len(ngram_counts[n]):>10,} 条"
                  f"  ({len(contexts[n]):,} 个上下文)")

        return ngram_counts, contexts

    def _prune(self, ngram_counts, contexts):
        """剪枝低频 n-gram"""
        for n in range(1, self.order + 1):
            threshold = self.prune_thresholds.get(n, 0)
            if threshold <= 0:
                continue

            before = len(ngram_counts[n])
            to_remove = [ng for ng, c in ngram_counts[n].items() if c <= threshold]
            for ng in to_remove:
                del ngram_counts[n][ng]
            after = len(ngram_counts[n])

            if before > after:
                print(f"  {n}-gram: {before:,} -> {after:,} (剪枝 {before - after:,} 条)")

                # 重建上下文信息
                ctx_info = {}
                for ngram, count in ngram_counts[n].items():
                    ctx = ngram[:-1]
                    word = ngram[-1]
                    if ctx not in ctx_info:
                        ctx_info[ctx] = {'total': 0, 'words': {}}
                    ctx_info[ctx]['total'] += count
                    ctx_info[ctx]['words'][word] = count
                contexts[n] = ctx_info

    def _compute_probs(self, ngram_counts, contexts):
        """计算 log10 概率和回退权重"""
        d = self.discount
        order = self.order

        entries = {}  # entries[n] = {ngram: [log_prob, backoff]}

        # ---------- 1-gram (MLE) ----------
        total_uni = sum(ngram_counts[1].values())
        entries[1] = {}
        for ngram, count in ngram_counts[1].items():
            entries[1][ngram] = [math.log10(count / total_uni), 0.0]
        print(f"  1-gram 概率: {len(entries[1]):,} 条 (total={total_uni:,})")

        # ---------- 高阶: 绝对折扣 ----------
        for n in range(2, order + 1):
            entries[n] = {}
            for ngram, count in ngram_counts[n].items():
                ctx = ngram[:-1]
                ctx_total = contexts[n][ctx]['total']
                disc = max(count - d, 0) / ctx_total if ctx_total > 0 else 1e-10
                entries[n][ngram] = [math.log10(max(disc, 1e-10)), 0.0]
            print(f"  {n}-gram 概率: {len(entries[n]):,} 条")

        # ---------- 回退权重 ----------
        for n in range(2, order + 1):
            n_bows = 0
            for ctx, info in contexts[n].items():
                ctx_total = info['total']
                n_types = len(info['words'])

                # 预留概率质量 = d * T(ctx) / N(ctx)
                reserved = d * n_types / ctx_total if ctx_total > 0 else 0
                reserved = min(reserved, 0.9999)

                # 低阶概率之和（仅限当前上下文看到的词）
                sum_lower = 0.0
                lower_n = n - 1
                for word in info['words']:
                    lower_ngram = ctx[1:] + (word,)
                    if lower_ngram in entries[lower_n]:
                        sum_lower += 10 ** entries[lower_n][lower_ngram][0]

                denom = 1.0 - sum_lower
                if denom > 1e-10 and reserved > 1e-10:
                    bow = math.log10(reserved / denom)
                elif reserved < 1e-10:
                    bow = -10.0
                else:
                    bow = 0.0

                # 将回退权重存储到 (n-1)-gram 条目中
                if ctx in entries[n - 1]:
                    entries[n - 1][ctx][1] = bow
                    n_bows += 1
                else:
                    # ctx 作为高阶上下文存在，但自身不是 (n-1)-gram 条目
                    # 这在未剪枝低阶 n-gram 时不应发生
                    # 但为安全起见，创建该条目
                    if lower_n >= 1:
                        entries[n - 1][ctx] = [-99.0, bow]
                        n_bows += 1

            print(f"  {n - 1}-gram 回退权重: {n_bows:,} 条")

        return entries

    def _write_arpa(self, entries, output_path):
        """写入 ARPA 格式文件（含 <unk> 条目）"""
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
            exist_ok=True
        )

        # 确保 unigram 中有 <unk> 条目（kenlm 需要）
        unk_ngram = ('<unk>',)
        if unk_ngram not in entries[1]:
            entries[1][unk_ngram] = [-100.0, 0.0]

        order = self.order
        total_lines = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write('\n\\data\\\n')
            for n in range(1, order + 1):
                f.write(f'ngram {n}={len(entries[n])}\n')
            f.write('\n')

            # N-gram sections
            for n in range(1, order + 1):
                f.write(f'\\{n}-grams:\n')
                sorted_ngrams = sorted(entries[n].keys())
                for ngram in sorted_ngrams:
                    log_prob, bow = entries[n][ngram]
                    ngram_str = ' '.join(ngram)
                    if n < order and abs(bow) > 1e-8:
                        f.write(f'{log_prob:.6f}\t{ngram_str}\t{bow:.6f}\n')
                    else:
                        f.write(f'{log_prob:.6f}\t{ngram_str}\n')
                    total_lines += 1
                f.write('\n')

            f.write('\\end\\\n')

        print(f"  写入 {total_lines:,} 行 ARPA 数据")


# ============================================================
# 验证
# ============================================================

def verify_model(arpa_path):
    """用 kenlm 加载并验证模型"""
    try:
        import kenlm # type: ignore[import-untyped]
        print(f"\n加载验证: {arpa_path}")
        t0 = time.time()
        model = kenlm.Model(arpa_path)
        elapsed = time.time() - t0
        print(f"  加载耗时: {elapsed:.1f}s")
        print(f"  模型阶数: {model.order}")

        # 测试评分
        test_cases = [
            "我 是 中 国 人",
            "今 天 天 气 不 错",
            "你 好 世 界",
        ]
        print(f"\n  测试评分 (空格分隔字符):")
        for sent in test_cases:
            score = model.score(sent, bos=True, eos=True)
            ppl = 10 ** (-score / max(len(sent.split()), 1))
            print(f"    '{sent}' -> log10P={score:.2f}, PPL≈{ppl:.1f}")

        # 测试 OOV
        oov_sent = "这 是 一 个 foobar 测 试"
        score_oov = model.score(oov_sent, bos=True, eos=True)
        print(f"    '{oov_sent}' (含OOV) -> log10P={score_oov:.2f}")

        print(f"\n  KenLM 模型加载验证通过 ✓")
        return True

    except Exception as e:
        print(f"\n  KenLM 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="构建字符级 KenLM n-gram 语言模型（ARPA 格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认参数构建（去重 + 5-gram + 剪枝）
  python tools/build_kenlm.py

  # 4-gram，更强剪枝
  python tools/build_kenlm.py --order 4 --prune 2

  # 不去重（对比去重效果）
  python tools/build_kenlm.py --max-copies 9999 --output ./lm/char_5gram_nodup.arpa

构建完成后，用于评估:
  python tools/eval_with_lm.py --lm-path ./lm/char_5gram.arpa --grid-search
"""
    )

    parser.add_argument("--train-json", type=str,
                        default="./split_data/train_set.json",
                        help="训练集 JSON 路径 (默认: ./split_data/train_set.json)")
    parser.add_argument("--output", type=str,
                        default="./lm/char_5gram.arpa",
                        help="输出 ARPA 文件路径 (默认: ./lm/char_5gram.arpa)")
    parser.add_argument("--order", type=int, default=5,
                        help="n-gram 阶数 (默认: 5)")
    parser.add_argument("--discount", type=float, default=0.75,
                        help="绝对折扣值 (默认: 0.75)")
    parser.add_argument("--max-copies", type=int, default=1,
                        help="每条唯一句子最多保留几份 (默认: 1, 即完全去重)")
    parser.add_argument("--prune", type=int, default=1,
                        help="剪枝阈值: count <= prune 的高阶(>=3) n-gram 被丢弃 (默认: 1)")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 kenlm 验证")

    args = parser.parse_args()

    print("=" * 60)
    print("  字符级 KenLM 语言模型构建工具")
    print("=" * 60)

    # Step 1: 提取和去重
    print(f"\n▶ 步骤 1: 提取文本并去重 (max_copies={args.max_copies})")
    sentences, stats = extract_and_dedup(args.train_json, args.max_copies)
    print_dedup_stats(stats)

    # Step 2: 构建 ARPA
    print(f"\n▶ 步骤 2: 构建 ARPA 模型")
    builder = ARPABuilder(
        order=args.order,
        discount=args.discount,
        prune_thresholds=args.prune
    )
    builder.build(sentences, args.output)

    # Step 3: 验证
    if not args.no_verify:
        print(f"\n▶ 步骤 3: 用 kenlm 验证模型")
        ok = verify_model(args.output)
        if not ok:
            print("\n⚠ 模型验证失败，请检查 ARPA 文件格式")
            sys.exit(1)

    # 总结
    print(f"\n{'='*60}")
    print("  构建完成！")
    print(f"{'='*60}")
    print(f"  ARPA 模型: {args.output}")
    print(f"  后续步骤:")
    print(f"    # 网格搜索最优 LM 参数")
    print(f"    python tools/eval_with_lm.py \\")
    print(f"        --model ./dialect_model_best \\")
    print(f"        --test-json ./split_data/test_set.json \\")
    print(f"        --lm-path {args.output} \\")
    print(f"        --grid-search")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
