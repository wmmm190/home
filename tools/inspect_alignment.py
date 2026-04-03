#!/usr/bin/env python3
"""
检查音文对齐分布并保存报告：
 - 生成 `all_items.csv`（所有样本）
 - 生成 `filtered_items.csv`（chars/sec 超出阈值的样本）
 - 生成 `chars_per_sec_hist.png`（直方图）

用法示例：
  python tools/inspect_alignment.py --data ./split_data/train_set.json --output ./analysis --min 1.5 --max 12.0
"""
import os
import json
import csv
import re
import argparse
import warnings
from tqdm import tqdm
import librosa
import numpy as np


def clean_text(text: str) -> str:
    if text is None:
        return ""
    # 与 preprocess_data.py 保持一致的简单清洗
    return re.sub(r"[^\u4e00-\u9fa5\w\s，。、；：\"']", '', text).strip()


def inspect(data_path, output_dir, sr=16000, min_rate=1.5, max_rate=12.0, top_k=20, plot=False):
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    all_csv_path = os.path.join(output_dir, 'all_items.csv')
    filtered_csv_path = os.path.join(output_dir, 'filtered_items.csv')

    rates = []
    rows = []

    with open(all_csv_path, 'w', encoding='utf-8', newline='') as all_f, \
         open(filtered_csv_path, 'w', encoding='utf-8', newline='') as filt_f:
        all_writer = csv.writer(all_f)
        filt_writer = csv.writer(filt_f)
        header = ['path', 'duration', 'chars', 'chars_per_sec', 'sentence', 'reason']
        all_writer.writerow(header)
        filt_writer.writerow(header)

        for item in tqdm(data_list, desc=f'Inspect {os.path.basename(data_path)}'):
            path = item.get('path')
            sentence = clean_text(item.get('sentence', ''))
            chars = len(sentence)
            duration = None
            reason = ''
            chars_per_sec = ''

            if not path or not os.path.exists(path):
                reason = 'missing_file'
                all_writer.writerow([path, '', chars, '', sentence, reason])
                filt_writer.writerow([path, '', chars, '', sentence, reason])
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    audio, _ = librosa.load(path, sr=sr)
                duration = len(audio) / sr if len(audio) > 0 else 0.0
                if duration <= 0:
                    reason = 'zero_duration'
                    chars_per_sec = 0.0
                else:
                    chars_per_sec = chars / duration
                all_writer.writerow([path, f'{duration:.3f}', chars, f'{chars_per_sec:.3f}', sentence, reason])

                if duration <= 0 or chars_per_sec < min_rate or chars_per_sec > max_rate:
                    if not reason:
                        if chars_per_sec < min_rate:
                            reason = 'low_chars_per_sec'
                        elif chars_per_sec > max_rate:
                            reason = 'high_chars_per_sec'
                    filt_writer.writerow([path, f'{duration:.3f}', chars, f'{chars_per_sec:.3f}', sentence, reason])

                if isinstance(chars_per_sec, (int, float)):
                    rates.append((chars_per_sec, path, chars, duration, sentence))

            except Exception as e:
                reason = f'error:{e}'
                all_writer.writerow([path, '', chars, '', sentence, reason])
                filt_writer.writerow([path, '', chars, '', sentence, reason])

    # 汇总统计
    rates_vals = [r[0] for r in rates if r[0] is not None]
    if rates_vals:
        arr = np.array(rates_vals)
        stats = {
            'count': len(arr),
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
        }
    else:
        stats = {}

    # 不生成图像，仅保存 CSV 和摘要
    hist_path = None

    # 输出 top K
    rates_sorted = sorted(rates, key=lambda x: x[0])
    lowest = rates_sorted[:top_k]
    highest = rates_sorted[-top_k:][::-1]

    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as sf:
        sf.write(json.dumps({'stats': stats}, ensure_ascii=False, indent=2))
        sf.write('\n\nLowest samples:\n')
        for r in lowest:
            sf.write(f'{r[0]:.3f}\t{r[1]}\tchars={r[2]}\tdur={r[3]:.3f}\n')
        sf.write('\nHighest samples:\n')
        for r in highest:
            sf.write(f'{r[0]:.3f}\t{r[1]}\tchars={r[2]}\tdur={r[3]:.3f}\n')

    print('Done.')
    print(f'All CSV: {all_csv_path}')
    print(f'Filtered CSV: {filtered_csv_path}')
    print(f'Hist: {hist_path}')
    print(f'Summary: {summary_path}')


def main():
    parser = argparse.ArgumentParser(description='Inspect chars/sec distribution for dataset')
    parser.add_argument('--data', type=str, required=True, help='path to JSON dataset (train_set.json or test_set.json)')
    parser.add_argument('--output', type=str, default='./analysis', help='output folder')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--min', type=float, default=1.5)
    parser.add_argument('--max', type=float, default=12.0)
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()

    inspect(args.data, args.output, sr=args.sr, min_rate=args.min, max_rate=args.max, top_k=args.top)


if __name__ == '__main__':
    main()
