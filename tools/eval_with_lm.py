"""
LM 辅助解码评估脚本

LM 后端：
    - KenLM (.arpa 文件): 推荐，C++ 引擎，速度快 10-50 倍（需 pip install kenlm）

功能：
  1. 使用字符级 n-gram 语言模型辅助 CTC 解码
  2. 使用 CTC prefix beam search + n-gram LM 解码
  3. 对比 greedy / beam(无LM) / beam+LM 三种解码策略的 CER
  4. 按方言输出分组 CER，便于分析 LM 对各方言的收益
  5. 网格搜索最优 LM 参数（预缓存 logits，高效搜索）

使用方式:
  # 第一步：构建 KenLM ARPA 模型（推荐，只需一次）
  python tools/build_kenlm.py

    # 第二步：运行评估对比（使用 .arpa）
  python tools/eval_with_lm.py ^
      --model ./dialect_model_best ^
      --test-json ./split_data/test_set.json ^
      --lm-path ./lm/char_5gram.arpa ^
      --beam-width 20 ^
      --lm-weight 0.5

  # 第三步：网格搜索最优 LM 参数（预缓存 logits，很快）
  python tools/eval_with_lm.py ^
      --model ./dialect_model_best ^
      --test-json ./split_data/test_set.json ^
      --lm-path ./lm/char_5gram.arpa ^
      --grid-search

依赖：transformers, torch, librosa, numpy, scipy, jiwer
可选：kenlm（使用 .arpa 文件时需要）
"""

import os
import sys
import json
import re
import math
import argparse
import warnings
import time
from collections import defaultdict

import numpy as np
import torch
import librosa
from scipy.special import log_softmax
from jiwer import cer as compute_cer

# 添加项目根目录到 Python 路径，以便导入 dialect_model 等模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

# ---------- 方言名称映射 ----------
DIALECT_NAMES = {
    0: '武汉话', 1: '南昌话', 2: '上海话', 3: '四川话',
    4: '天津话', 5: '长沙话', 6: '郑州话'
}


# ==============================================================================
# 第一部分：KenLM 评分器（高性能 C++ 后端）
# ==============================================================================

class KenLMScorer:
    """
    kenlm.Model 包装器，用于 CTC beam search 中的 LM 评分

    优势：
    - 查询速度快 10-50 倍（C++ trie 结构）
    - 内存效率高（trie 压缩）
    - 支持 state-based API（增量评分，无需重复计算上下文）
    - ARPA 格式标准，可与其他工具互通

    使用方式:
      scorer = KenLMScorer('./lm/char_5gram.arpa')
      state = scorer.get_initial_state()
      score, new_state = scorer.score_and_advance(state, '你')
    """

    def __init__(self, arpa_path):
        import kenlm  # type: ignore[import-untyped]
        self._kenlm_mod = kenlm
        print(f"加载 KenLM 模型: {arpa_path}")
        self.model = kenlm.Model(arpa_path)
        self.order = self.model.order
        print(f"  order={self.order}, 加载完成")

    def get_initial_state(self):
        """返回句首 (BOS) 状态"""
        state = self._kenlm_mod.State()
        self.model.BeginSentenceWrite(state)
        return state

    def score_and_advance(self, state, char):
        """
        给定当前状态和字符，返回 (log10_prob, new_state)

        Args:
            state: kenlm.State, 当前上下文状态
            char: str, 目标字符

        Returns:
            log10_prob: float, log10 条件概率
            new_state: kenlm.State, 更新后的状态
        """
        out_state = self._kenlm_mod.State()
        log_prob = self.model.BaseScore(state, char, out_state)
        return log_prob, out_state

    def score_eos(self, state):
        """返回句末 </s> 的 log10 概率"""
        out_state = self._kenlm_mod.State()
        return self.model.BaseScore(state, '</s>', out_state)

    def score(self, context, char):
        """
        兼容通用接口: P(char | context) 的 log10 概率
        注意：此方法效率较低，beam search 中优先使用 state-based API
        """
        state = self.get_initial_state()
        for c in context:
            if c == '<s>':
                continue  # BeginSentenceWrite 已处理 BOS
            _, state = self.score_and_advance(state, c)
        log_prob, _ = self.score_and_advance(state, char)
        return log_prob


# ==============================================================================
# 第二部分：CTC Prefix Beam Search 解码器
# ==============================================================================

def _logsumexp(a, b):
    """稳定的 log(exp(a) + exp(b))"""
    if a == -float('inf'):
        return b
    if b == -float('inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


class CTCBeamDecoder:
    """
    CTC Prefix Beam Search 解码器（纯 Python，已优化）

    优化措施:
    - 使用字符串前缀代替元组（更快的拼接）
    - 使用列表索引代替字典查找（id→char）
    - 预过滤 top-k token（排除 blank / 特殊 token）
    - 正确处理 CTC blank 和重复字符合并

    算法基于: Hannun et al., 2014
    """

    def __init__(self, vocab, blank_id, lm=None,
                 beam_width=20, alpha=0.5, beta=0.0):
        self.blank_id = int(blank_id)
        self.lm = lm
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.vocab_size = len(vocab)

        # 构建 id→char 列表（用索引替代字典，更快）
        max_id = max(vocab.values())
        self.id2char = [''] * (max_id + 1)
        self.skip_ids = set()
        for tok, tid in vocab.items():
            if tok in ('[UNK]', '[PAD]', '<s>', '</s>'):
                self.skip_ids.add(tid)
            else:
                self.id2char[tid] = tok

        self._use_lm = (lm is not None and alpha > 0)
        if self._use_lm and not isinstance(lm, KenLMScorer):
            raise TypeError("仅支持 KenLMScorer 作为 LM 后端")

    def decode(self, log_probs):
        """
        CTC prefix beam search 解码（支持 KenLM state-based API）

        Args:
            log_probs: numpy array, shape (T, V), log 概率（已做 log_softmax）

        Returns:
            str: 最佳解码文本
        """
        T, V = log_probs.shape
        NEG_INF = float('-inf')
        beam_width = self.beam_width
        blank_id = self.blank_id
        id2char = self.id2char
        skip_ids = self.skip_ids
        use_lm = self._use_lm
        _logsumexp_fn = _logsumexp

        # 用字符串作为 prefix key
        # beams: dict {prefix_str: [log_p_blank, log_p_non_blank]}
        beams = {'': [0.0, NEG_INF]}

        # KenLM state 管理（与 beams 分开存储，避免改变 beam 格式）
        if use_lm:
            lm_states = {'': self.lm.get_initial_state()}
        else:
            lm_states = None

        for t in range(T):
            frame = log_probs[t]
            blank_prob = float(frame[blank_id])

            # 预计算 top-k token（排除 blank 和特殊 token）
            top_k = min(beam_width + 10, V)
            top_ids = np.argpartition(frame, -top_k)[-top_k:]
            valid_tokens = []
            for cid in top_ids:
                cid_int = int(cid)
                if cid_int == blank_id or cid_int in skip_ids:
                    continue
                ch = id2char[cid_int]
                if ch:
                    valid_tokens.append((cid_int, ch, float(frame[cid_int])))

            # 剪枝：只保留 top-k beams
            if len(beams) > beam_width:
                sorted_items = sorted(
                    beams.items(),
                    key=lambda x: _logsumexp_fn(x[1][0], x[1][1]),
                    reverse=True
                )[:beam_width]
            else:
                sorted_items = list(beams.items())

            new_beams = {}
            if use_lm:
                new_lm_states = {}

            for prefix, pb_pnb in sorted_items:
                p_b = pb_pnb[0]
                p_nb = pb_pnb[1]
                p_total = _logsumexp_fn(p_b, p_nb)

                # --- blank 扩展: 前缀不变 ---
                entry = new_beams.get(prefix)
                if entry is None:
                    new_beams[prefix] = [p_total + blank_prob, NEG_INF]
                    if use_lm and prefix not in new_lm_states:
                        new_lm_states[prefix] = lm_states.get(prefix)
                else:
                    entry[0] = _logsumexp_fn(entry[0], p_total + blank_prob)

                # --- 非 blank 扩展 ---
                last_char = prefix[-1] if prefix else ''

                # 获取当前 prefix 的 LM state
                if use_lm:
                    parent_state = lm_states.get(prefix)

                for c_id, c_char, c_prob in valid_tokens:
                    new_prefix = prefix + c_char

                    # LM 评分
                    if use_lm:
                        lm_score, child_state = self.lm.score_and_advance(
                            parent_state, c_char
                        )
                        lm_bonus = self.alpha * lm_score
                    else:
                        lm_bonus = 0.0

                    if c_char != last_char:
                        # 非重复字符
                        entry = new_beams.get(new_prefix)
                        new_nb = p_total + c_prob + lm_bonus
                        if entry is None:
                            new_beams[new_prefix] = [NEG_INF, new_nb]
                        else:
                            entry[1] = _logsumexp_fn(entry[1], new_nb)
                        # 记录 kenlm state
                        if use_lm and new_prefix not in new_lm_states:
                            new_lm_states[new_prefix] = child_state
                    else:
                        # 重复字符:
                        # 1) blank 结尾路径可以产生新的重复
                        entry = new_beams.get(new_prefix)
                        new_nb = p_b + c_prob + lm_bonus
                        if entry is None:
                            new_beams[new_prefix] = [NEG_INF, new_nb]
                        else:
                            entry[1] = _logsumexp_fn(entry[1], new_nb)
                        if use_lm and new_prefix not in new_lm_states:
                            new_lm_states[new_prefix] = child_state
                        # 2) non-blank 结尾路径: 延续当前前缀（不增加字符）
                        entry2 = new_beams.get(prefix)
                        if entry2 is not None:
                            entry2[1] = _logsumexp_fn(entry2[1], p_nb + c_prob)

            beams = new_beams
            if use_lm:
                lm_states = new_lm_states

        # 选择最佳 beam
        best_prefix = ''
        best_score = NEG_INF
        beta = self.beta
        alpha = self.alpha
        for prefix, pb_pnb in beams.items():
            score = _logsumexp_fn(pb_pnb[0], pb_pnb[1])
            score += beta * len(prefix)
            # EOS LM 评分
            if use_lm and prefix:
                st = lm_states.get(prefix)
                if st is not None:
                    eos_score = self.lm.score_eos(st)
                    score += alpha * eos_score
            if score > best_score:
                best_score = score
                best_prefix = prefix

        return best_prefix.replace('|', '')


# ==============================================================================
# 第三部分：加载模型并获取 logits
# ==============================================================================

def load_model_and_processor(model_path, device="cuda"):
    """加载声学模型和 processor"""
    from transformers import Wav2Vec2Processor, Wav2Vec2Config

    processor = Wav2Vec2Processor.from_pretrained(model_path)

    dialect_config_path = os.path.join(model_path, "dialect_config.json")
    if os.path.exists(dialect_config_path):
        from dialect_model import EnhancedWav2Vec2ForDialect
        with open(dialect_config_path, "r") as f:
            dialect_config = json.load(f)
        config = Wav2Vec2Config.from_pretrained(model_path)
        model = EnhancedWav2Vec2ForDialect(config, dialect_config)

        weight_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(weight_path):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(weight_path)
        else:
            state_dict = torch.load(
                os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"
            )
        model.load_state_dict(state_dict, strict=False)
        print("加载增强型方言模型")
    else:
        from transformers import Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
        print("加载标准 Wav2Vec2 模型")

    model.eval()
    model.to(device)
    return model, processor


def get_logits(model, processor, audio_path, device="cuda"):
    """对单条音频获取模型 logits，返回 numpy (T, V)"""
    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(**inputs)

    return outputs.logits[0].cpu().float().numpy()


# ==============================================================================
# 第四部分：Greedy 解码（基线）
# ==============================================================================

def greedy_decode(logits, processor):
    """Greedy 解码（当前基线）"""
    pred_ids = np.argmax(logits, axis=-1)
    pred_text = processor.decode(pred_ids, skip_special_tokens=True)
    return pred_text


# ==============================================================================
# 第五部分：评估与对比
# ==============================================================================

def clean_text(text):
    """与训练/评估脚本一致的文本清洗"""
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text.strip()


def evaluate_decoding(model, processor, test_data, device,
                      beam_decoder_lm=None, beam_decoder_no_lm=None,
                      max_samples=None):
    """
    评估三种解码策略的 CER

    Returns:
        results: dict with greedy/beam/beam_lm CER (overall + per-dialect)
    """
    greedy_preds = []
    beam_preds = []
    beam_lm_preds = []
    refs = []
    dialect_ids = []

    n = len(test_data) if max_samples is None else min(max_samples, len(test_data))
    print(f"\n评估 {n} 条样本...")

    t0 = time.time()
    skipped = 0

    for i in range(n):
        item = test_data[i]
        try:
            ref = clean_text(item.get("sentence", ""))
            if not ref:
                skipped += 1
                continue

            audio_path = item.get("path", "")
            if not os.path.exists(audio_path):
                skipped += 1
                continue

            logits = get_logits(model, processor, audio_path, device)
            lp = log_softmax(logits, axis=-1)

            # Greedy
            g_pred = greedy_decode(logits, processor)
            greedy_preds.append(clean_text(g_pred))

            # Beam (no LM)
            if beam_decoder_no_lm is not None:
                b_pred = beam_decoder_no_lm.decode(lp)
                beam_preds.append(clean_text(b_pred))

            # Beam + LM
            if beam_decoder_lm is not None:
                bl_pred = beam_decoder_lm.decode(lp)
                beam_lm_preds.append(clean_text(bl_pred))

            refs.append(ref)
            dialect_ids.append(item.get("dialect_id", -1))

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                eta = (n - i - 1) / speed if speed > 0 else 0
                print(f"  已处理 {i+1}/{n}... ({speed:.1f} 条/秒, ETA {eta:.0f}s)")

        except Exception as e:
            if skipped < 5:
                print(f"  跳过样本 {i}: {e}")
            skipped += 1
            continue

    elapsed = time.time() - t0
    print(f"  评估完成，耗时 {elapsed:.1f}s ({len(refs)} 有效 / {skipped} 跳过)")

    # 计算 CER
    results = {}

    def calc_cer(preds, refs_list):
        valid = [(p, r) for p, r in zip(preds, refs_list) if p.strip() and r.strip()]
        if not valid:
            return 1.0, 1.0, {}
        vp, vr = zip(*valid)
        overall = compute_cer(list(vr), list(vp))

        per_dialect = {}
        dialect_p = defaultdict(list)
        dialect_r = defaultdict(list)
        for p, r, did in zip(preds, refs_list, dialect_ids):
            if p.strip() and r.strip() and did >= 0:
                dname = DIALECT_NAMES.get(did, f"方言{did}")
                dialect_p[dname].append(p)
                dialect_r[dname].append(r)
        for dname in dialect_p:
            per_dialect[dname] = compute_cer(dialect_r[dname], dialect_p[dname])

        # Macro CER: 各方言 CER 的算术平均（各方言等权）
        macro_cer = (float(np.mean(list(per_dialect.values()))) if per_dialect
                     else overall)

        return overall, macro_cer, per_dialect

    g_cer, g_macro, g_per = calc_cer(greedy_preds, refs)
    results["greedy"] = {"cer": g_cer, "macro_cer": g_macro, "per_dialect": g_per}

    if beam_preds:
        b_cer, b_macro, b_per = calc_cer(beam_preds, refs)
        results["beam_no_lm"] = {"cer": b_cer, "macro_cer": b_macro, "per_dialect": b_per}

    if beam_lm_preds:
        bl_cer, bl_macro, bl_per = calc_cer(beam_lm_preds, refs)
        results["beam_lm"] = {"cer": bl_cer, "macro_cer": bl_macro, "per_dialect": bl_per}

    return results


def print_results(results):
    """打印对比结果表格"""
    print(f"\n{'='*70}")
    print("解码策略对比")
    print(f"{'='*70}")

    print(f"\n{'策略':<20} {'CER':>10} {'Macro CER':>12}")
    print(f"{'-'*48}")
    for strategy in ["greedy", "beam_no_lm", "beam_lm"]:
        if strategy in results:
            cer_val = results[strategy]["cer"]
            macro_val = results[strategy].get("macro_cer", -1)
            label = {"greedy": "Greedy (基线)",
                     "beam_no_lm": "Beam (无LM)",
                     "beam_lm": "Beam + LM"}[strategy]
            macro_str = f"{macro_val*100:>11.2f}%" if macro_val >= 0 else f"{'N/A':>12}"
            print(f"{label:<20} {cer_val*100:>9.2f}% {macro_str}")

    # Greedy -> Beam+LM 改进
    if "greedy" in results and "beam_lm" in results:
        g = results["greedy"]["cer"]
        bl = results["beam_lm"]["cer"]
        delta = (bl - g) * 100
        rel = (bl - g) / g * 100 if g > 0 else 0
        sign = "+" if delta > 0 else ""
        print(f"\n  LM 收益 (CER): {sign}{delta:.2f}% 绝对, {sign}{rel:.1f}% 相对")
        g_m = results["greedy"].get("macro_cer", -1)
        bl_m = results["beam_lm"].get("macro_cer", -1)
        if g_m >= 0 and bl_m >= 0:
            delta_m = (bl_m - g_m) * 100
            rel_m = (bl_m - g_m) / g_m * 100 if g_m > 0 else 0
            sign_m = "+" if delta_m > 0 else ""
            print(f"  LM 收益 (Macro): {sign_m}{delta_m:.2f}% 绝对, {sign_m}{rel_m:.1f}% 相对")

    # 分方言
    if any("per_dialect" in results.get(s, {}) for s in results):
        print(f"\n{'='*70}")
        print("分方言 CER 对比")
        print(f"{'='*70}")

        all_dialects = set()
        for s in results:
            all_dialects.update(results[s].get("per_dialect", {}).keys())

        header = f"{'方言':<10}"
        for strategy in ["greedy", "beam_no_lm", "beam_lm"]:
            if strategy in results:
                label = {"greedy": "Greedy", "beam_no_lm": "Beam",
                         "beam_lm": "Beam+LM"}[strategy]
                header += f" {label:>10}"
        if "greedy" in results and "beam_lm" in results:
            header += f" {'Delta':>10}"
        print(header)
        print(f"{'-'*70}")

        for dname in sorted(all_dialects):
            row = f"{dname:<10}"
            g_val = None
            bl_val = None
            for strategy in ["greedy", "beam_no_lm", "beam_lm"]:
                if strategy in results:
                    val = results[strategy]["per_dialect"].get(dname, -1)
                    if val >= 0:
                        row += f" {val*100:>9.2f}%"
                    else:
                        row += f" {'N/A':>10}"
                    if strategy == "greedy":
                        g_val = val
                    if strategy == "beam_lm":
                        bl_val = val
            if g_val is not None and bl_val is not None and g_val >= 0 and bl_val >= 0:
                delta = (bl_val - g_val) * 100
                sign = "+" if delta > 0 else ""
                row += f" {sign}{delta:>8.2f}%"
            print(row)

    print(f"{'='*70}\n")


def grid_search(model, processor, test_data, device, lm, max_samples=200):
    """
    网格搜索最优 LM 参数（优化版：预缓存 logits）

    优化：声学模型 logits 只计算一次，之后所有参数组合复用
    这使得搜索时间从 O(params * samples * model_time) 降至
    O(samples * model_time + params * samples * decode_time)
    """
    print(f"\n{'='*70}")
    print(f"网格搜索 LM 参数（预缓存模式，前 {max_samples} 条样本）")
    print(f"{'='*70}")

    vocab = processor.tokenizer.get_vocab()
    blank_id = vocab.get('[PAD]', 0)

    # ---- 阶段 1: 预缓存所有样本的 logits ----
    print("\n[阶段 1] 预计算声学模型 logits（只需一次）...")
    logits_cache = []  # list of numpy (T, V) log_softmax
    refs_cache = []
    dialect_ids_cache = []

    n = min(max_samples, len(test_data))
    t0 = time.time()
    skipped = 0

    for i in range(n):
        item = test_data[i]
        try:
            ref = clean_text(item.get("sentence", ""))
            if not ref:
                skipped += 1
                continue
            audio_path = item.get("path", "")
            if not os.path.exists(audio_path):
                skipped += 1
                continue

            logits_raw = get_logits(model, processor, audio_path, device)
            lp = log_softmax(logits_raw, axis=-1)
            logits_cache.append(lp)
            refs_cache.append(ref)
            dialect_ids_cache.append(item.get("dialect_id", -1))

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                eta = (n - i - 1) / speed if speed > 0 else 0
                print(f"  已处理 {i+1}/{n}... ({speed:.1f} 条/秒, ETA {eta:.0f}s)")

        except Exception as e:
            if skipped < 5:
                print(f"  跳过样本 {i}: {e}")
            skipped += 1

    elapsed = time.time() - t0
    print(f"  缓存完成: {len(logits_cache)} 条有效样本, "
          f"耗时 {elapsed:.1f}s ({skipped} 跳过)")

    # ---- 先计算 Greedy 基线 ----
    greedy_preds = []
    for lp in logits_cache:
        pred_ids = np.argmax(lp, axis=-1)
        pred_text = processor.decode(pred_ids, skip_special_tokens=True)
        greedy_preds.append(clean_text(pred_text))

    valid_g = [(p, r) for p, r in zip(greedy_preds, refs_cache)
               if p.strip() and r.strip()]
    if valid_g:
        gp, gr = zip(*valid_g)
        greedy_cer = compute_cer(list(gr), list(gp))
    else:
        greedy_cer = 1.0
    print(f"  Greedy 基线 CER: {greedy_cer*100:.2f}%")

    # ---- 阶段 2: 网格搜索 ----
    print(f"\n[阶段 2] 搜索最优参数...")

    beam_widths = [10, 20]
    alphas = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    betas = [0.0, 0.5, 1.0, 1.5]

    best_cer = 1.0
    best_params = {}
    all_results = []
    total = len(beam_widths) * len(alphas) * len(betas)
    count = 0

    for bw in beam_widths:
        for alpha in alphas:
            for beta in betas:
                count += 1
                t1 = time.time()
                try:
                    decoder = CTCBeamDecoder(
                        vocab=vocab, blank_id=blank_id,
                        lm=lm, beam_width=bw,
                        alpha=alpha, beta=beta
                    )

                    preds = []
                    for lp in logits_cache:
                        pred = decoder.decode(lp)
                        preds.append(clean_text(pred))

                    valid = [(p, r) for p, r in zip(preds, refs_cache)
                             if p.strip() and r.strip()]
                    if valid:
                        vp, vr = zip(*valid)
                        lm_cer = compute_cer(list(vr), list(vp))
                    else:
                        lm_cer = 1.0

                    dt = time.time() - t1
                    is_best = lm_cer < best_cer
                    if is_best:
                        best_cer = lm_cer
                        best_params = {
                            "beam_width": bw, "alpha": alpha, "beta": beta
                        }

                    delta = (lm_cer - greedy_cer) * 100
                    sign = "+" if delta > 0 else ""
                    print(f"  [{count:3d}/{total}] bw={bw:2d}  α={alpha:.1f}  "
                          f"β={beta:>4.1f}  -> CER={lm_cer*100:6.2f}%  "
                          f"(vs greedy {sign}{delta:.2f}%)  "
                          f"[{dt:.1f}s]"
                          f"{'  ★ BEST' if is_best else ''}")

                    all_results.append({
                        "beam_width": bw, "alpha": alpha, "beta": beta,
                        "cer": lm_cer, "delta_vs_greedy": delta
                    })

                except Exception as e:
                    print(f"  [{count:3d}/{total}] bw={bw:2d}  α={alpha:.1f}  "
                          f"β={beta:>4.1f}  -> ERROR: {e}")

    # ---- 结果汇总 ----
    print(f"\n{'='*70}")
    print(f"网格搜索结果")
    print(f"{'='*70}")
    print(f"  Greedy 基线 CER:  {greedy_cer*100:.2f}%")
    print(f"  最优 Beam+LM CER: {best_cer*100:.2f}%")
    if best_params:
        delta = (best_cer - greedy_cer) * 100
        rel = (best_cer - greedy_cer) / greedy_cer * 100 if greedy_cer > 0 else 0
        sign = "+" if delta > 0 else ""
        print(f"  LM 收益: {sign}{delta:.2f}% 绝对, {sign}{rel:.1f}% 相对")
        print(f"  最优参数:")
        print(f"    beam_width = {best_params['beam_width']}")
        print(f"    alpha (LM权重) = {best_params['alpha']}")
        print(f"    beta (长度奖励) = {best_params['beta']}")

    # Top 5 结果
    if all_results:
        all_results.sort(key=lambda x: x['cer'])
        print(f"\n  Top 5 参数组合:")
        for i, r in enumerate(all_results[:5], 1):
            print(f"    {i}. bw={r['beam_width']}, α={r['alpha']}, "
                  f"β={r['beta']} -> CER={r['cer']*100:.2f}%")

    print(f"{'='*70}")

    # 保存搜索结果
    results_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    grid_path = os.path.join(results_dir, f"grid_search_{ts}.json")
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump({
            "greedy_cer": greedy_cer,
            "best_cer": best_cer,
            "best_params": best_params,
            "all_results": all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"  搜索结果已保存: {grid_path}")

    return best_params, best_cer


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LM 辅助解码评估（KenLM 后端）"
    )

    # 评估
    parser.add_argument("--model", type=str, default="./dialect_model_best",
                        help="声学模型路径")
    parser.add_argument("--test-json", type=str, default="./split_data/test_set.json",
                        help="测试集 JSON")
    parser.add_argument("--dev-json", type=str, default="./split_data/dev_set.json",
                        help="验证集 JSON（网格搜索用，避免测试集泄露）")
    parser.add_argument("--lm-path", type=str, default=None,
                        help="KenLM 文件路径（.arpa/.bin，不提供则只做 greedy/beam 对比）")
    parser.add_argument("--beam-width", type=int, default=20,
                        help="Beam 宽度（默认 20）")
    parser.add_argument("--lm-weight", type=float, default=0.5,
                        help="LM 权重 alpha（默认 0.5）")
    parser.add_argument("--word-score", type=float, default=0.0,
                        help="长度奖励 beta（默认 0.0）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多评估样本数（默认全部）")

    # 网格搜索
    parser.add_argument("--grid-search", action="store_true",
                        help="网格搜索最优 LM 参数")
    parser.add_argument("--grid-samples", type=int, default=200,
                        help="网格搜索样本数（默认 200）")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # ---------- 评估模式 ----------
    if not args.model or not args.test_json:
        parser.print_help()
        return

    if not os.path.exists(args.test_json):
        print(f"错误: 测试数据不存在: {args.test_json}")
        return
    if not os.path.exists(args.model):
        print(f"错误: 模型不存在: {args.model}")
        return

    # 加载模型
    model, processor = load_model_and_processor(args.model, args.device)
    vocab = processor.tokenizer.get_vocab()
    blank_id = vocab.get('[PAD]', 0)
    print(f"vocab 大小: {len(vocab)}, blank_id([PAD]): {blank_id}")

    # 加载测试数据
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 条")

    # 加载 LM（仅支持 KenLM .arpa/.bin）
    lm = None
    if args.lm_path and os.path.exists(args.lm_path):
        if args.lm_path.endswith('.arpa') or args.lm_path.endswith('.bin'):
            try:
                lm = KenLMScorer(args.lm_path)
                print(f"  使用 KenLM 后端 (C++ 加速)")
            except ImportError:
                print("错误: 使用 .arpa 文件需要安装 kenlm: pip install kenlm")
                return
            except Exception as e:
                print(f"错误: 加载 KenLM 模型失败: {e}")
                return
        else:
            print(f"错误: 仅支持 KenLM 文件 (.arpa/.bin): {args.lm_path}")
            return
    elif args.lm_path:
        print(f"警告: LM 文件不存在: {args.lm_path}，将只做 greedy 和 beam 对比")

    # 网格搜索模式
    if args.grid_search:
        if lm is None:
            print("错误: 网格搜索需要 --lm-path 指向有效的 LM 文件")
            return

        # 优先用验证集搜索，避免测试集泄露
        grid_data = test_data  # 默认回退到测试集
        grid_source = args.test_json
        if args.dev_json and os.path.exists(args.dev_json):
            with open(args.dev_json, "r", encoding="utf-8") as f:
                grid_data = json.load(f)
            grid_source = args.dev_json
            print(f"\n网格搜索使用验证集: {args.dev_json} ({len(grid_data)} 条)")
            print(f"  (测试集保留为最终评估，避免参数过拟合)")
        else:
            print(f"\n警告: 验证集不存在 ({args.dev_json})，回退到测试集进行搜索")
            print(f"  建议运行 fen.py 生成验证集: python fen.py")

        best_params, best_cer = grid_search(
            model, processor, grid_data, args.device,
            lm, max_samples=args.grid_samples
        )
        print(f"\n建议使用最优参数在测试集上运行完整评估:")
        print(f"  python tools/eval_with_lm.py "
              f"--model {args.model} --test-json {args.test_json} "
              f"--lm-path {args.lm_path} "
              f"--beam-width {best_params.get('beam_width', 20)} "
              f"--lm-weight {best_params.get('alpha', 0.5)} "
              f"--word-score {best_params.get('beta', 0.0)}")
        return

    # 构建解码器
    beam_decoder_lm = None
    beam_decoder_no_lm = None

    if lm is not None:
        print(f"\n创建 Beam+LM 解码器 (alpha={args.lm_weight}, beta={args.word_score},"
              f" beam={args.beam_width})...")
        beam_decoder_lm = CTCBeamDecoder(
            vocab=vocab, blank_id=blank_id, lm=lm,
            beam_width=args.beam_width,
            alpha=args.lm_weight, beta=args.word_score
        )

    print(f"创建 Beam(无LM) 解码器 (beam={args.beam_width})...")
    beam_decoder_no_lm = CTCBeamDecoder(
        vocab=vocab, blank_id=blank_id, lm=None,
        beam_width=args.beam_width, alpha=0, beta=0
    )

    # 运行验证集评估（如果可用）
    val_results = None
    if args.dev_json and os.path.exists(args.dev_json):
        with open(args.dev_json, "r", encoding="utf-8") as f:
            dev_data = json.load(f)
        print(f"\n验证集: {len(dev_data)} 条 ({args.dev_json})")
        val_results = evaluate_decoding(
            model, processor, dev_data, args.device,
            beam_decoder_lm=beam_decoder_lm,
            beam_decoder_no_lm=beam_decoder_no_lm,
            max_samples=args.max_samples
        )
        print("\n" + "=" * 70)
        print("验证集 (Val) 结果")
        print("=" * 70)
        print_results(val_results)

    # 运行测试集评估
    results = evaluate_decoding(
        model, processor, test_data, args.device,
        beam_decoder_lm=beam_decoder_lm,
        beam_decoder_no_lm=beam_decoder_no_lm,
        max_samples=args.max_samples
    )

    print("\n" + "=" * 70)
    print("测试集 (Test) 结果")
    print("=" * 70)
    print_results(results)

    # 保存结果
    results_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_data = {"test": results}
    if val_results is not None:
        save_data["val"] = val_results
    output_path = os.path.join(results_dir, f"lm_eval_results_{ts}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
