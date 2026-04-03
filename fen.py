"""
多方言数据集分割与平衡脚本

功能：
1. 合并 datajson/ 下 7 种方言 JSON 文件
2. 从文件路径自动提取方言标签（dialect_id）
3. 分层采样（每种方言各 70% 训练、15% 验证、15% 测试）
4. 训练集过采样平衡（少数方言重复至目标数量，配合在线增强避免过拟合）
5. 输出带 dialect_id 字段的 train_set.json / dev_set.json / test_set.json

验证集用途：网格搜索 LM 参数、early stopping 等调参，避免测试集泄露
"""

import json
import random
import os
import re
from collections import defaultdict

# 设置随机种子以确保可复现性
random.seed(42)

# ============================================================
# 方言映射与数据文件配置
# ============================================================

DIALECT_MAP = {
    'wuhan': 0,
    'nanchang': 1,
    'shanghai': 2,
    'sichuan': 3,
    'tianjin': 4,
    'changsha': 5,
    'zhengzhou': 6,
}

DIALECT_NAMES = {v: k for k, v in DIALECT_MAP.items()}

# datajson/ 下各方言对应的文件名
DIALECT_FILES = {
    'wuhan':    'updated_paths.json',
    'nanchang': 'nanchang_converted.json',
    'shanghai': 'shanghai_converted.json',
    'sichuan':  'sichuan_converted.json',
    'tianjin':  'tianjin_converted.json',
    'changsha': 'changsha_converted.json',
    'zhengzhou':'zhengzhou_converted.json',
}


def load_all_dialects(datajson_dir):
    """
    加载所有方言数据并附加 dialect_id

    Returns:
        dialect_groups: dict[str, list]  按方言名分组的数据
    """
    dialect_groups = defaultdict(list)

    for dialect_name, filename in DIALECT_FILES.items():
        filepath = os.path.join(datajson_dir, filename)
        if not os.path.exists(filepath):
            print(f"  警告: 文件不存在，跳过 {filepath}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            items = json.load(f)

        dialect_id = DIALECT_MAP[dialect_name]
        for item in items:
            item['dialect_id'] = dialect_id
            item['dialect'] = dialect_name
        dialect_groups[dialect_name] = items

        print(f"  {dialect_name}: {len(items)} 条")

    return dialect_groups


def stratified_split(dialect_groups, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    """
    分层采样：每种方言各自按比例拆分为训练集、验证集和测试集

    Args:
        dialect_groups: 按方言分组的数据
        train_ratio: 训练集比例（默认 0.7）
        dev_ratio: 验证集比例（默认 0.15）
        test_ratio: 测试集比例（默认 0.15）

    Returns:
        train_items, dev_items, test_items
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为 1.0, 当前: {train_ratio + dev_ratio + test_ratio}"

    train_items = []
    dev_items = []
    test_items = []

    for dialect_name, items in dialect_groups.items():
        shuffled = items.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        dev_end = int(n * (train_ratio + dev_ratio))

        train_items.extend(shuffled[:train_end])
        dev_items.extend(shuffled[train_end:dev_end])
        test_items.extend(shuffled[dev_end:])

    return train_items, dev_items, test_items


def balance_training_data(train_items, target_per_dialect=2000):
    """
    平衡训练集：
    - 样本数 > target 的方言：随机降采样
    - 样本数 < target 的方言：重复过采样（训练时在线增强保证多样性）

    Args:
        train_items: 训练数据列表（已含 dialect_id）
        target_per_dialect: 每种方言的目标样本数

    Returns:
        balanced: 平衡后的训练数据列表
    """
    groups = defaultdict(list)
    for item in train_items:
        groups[item['dialect']].append(item)

    balanced = []
    for dialect_name, items in groups.items():
        n = len(items)
        if n >= target_per_dialect:
            sampled = random.sample(items, target_per_dialect)
            balanced.extend(sampled)
            print(f"  {dialect_name}: {n} → {target_per_dialect} (降采样)")
        else:
            balanced.extend(items)
            extra = target_per_dialect - n
            oversampled = random.choices(items, k=extra)
            balanced.extend(oversampled)
            print(f"  {dialect_name}: {n} → {target_per_dialect} (过采样 +{extra})")

    random.shuffle(balanced)
    return balanced


def main():
    datajson_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datajson')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_data')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载全部方言数据
    print("=" * 60)
    print("步骤1: 加载7种方言数据")
    print("=" * 60)
    dialect_groups = load_all_dialects(datajson_dir)

    total_samples = sum(len(v) for v in dialect_groups.values())
    print(f"\n总样本数: {total_samples}")
    print(f"方言数量: {len(dialect_groups)}")

    # 2. 分层采样拆分
    print("\n" + "=" * 60)
    print("步骤2: 分层采样 (70% 训练 / 15% 验证 / 15% 测试)")
    print("=" * 60)
    train_items, dev_items, test_items = stratified_split(
        dialect_groups, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15
    )
    print(f"训练集: {len(train_items)}")
    print(f"验证集: {len(dev_items)}")
    print(f"测试集: {len(test_items)}")

    # 显示各方言分布
    for split_name, split_data in [("训练集", train_items), ("验证集", dev_items), ("测试集", test_items)]:
        print(f"\n{split_name}方言分布:")
        dist = defaultdict(int)
        for item in split_data:
            dist[item['dialect']] += 1
        for d, c in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {d}: {c}")

    # 3. 训练集过采样平衡
    print("\n" + "=" * 60)
    print("步骤3: 训练集过采样平衡 (目标: 每种方言 2000 条)")
    print("=" * 60)
    balanced_train = balance_training_data(train_items, target_per_dialect=2000)
    print(f"\n平衡后训练集大小: {len(balanced_train)}")

    # 验证平衡后的分布
    print("\n平衡后训练集方言分布:")
    balanced_dist = defaultdict(int)
    for item in balanced_train:
        balanced_dist[item['dialect']] += 1
    for d, c in sorted(balanced_dist.items()):
        print(f"  {d}: {c}")

    # 4. 保存
    print("\n" + "=" * 60)
    print("步骤4: 保存数据集")
    print("=" * 60)

    train_path = os.path.join(output_dir, 'train_set.json')
    dev_path = os.path.join(output_dir, 'dev_set.json')
    test_path = os.path.join(output_dir, 'test_set.json')

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_train, f, ensure_ascii=False, indent=2)

    with open(dev_path, 'w', encoding='utf-8') as f:
        json.dump(dev_items, f, ensure_ascii=False, indent=2)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_items, f, ensure_ascii=False, indent=2)

    print(f"训练集已保存至: {train_path}")
    print(f"验证集已保存至: {dev_path}")
    print(f"测试集已保存至: {test_path}")

    # 5. 输出摘要
    print("\n" + "=" * 60)
    print("数据集摘要")
    print("=" * 60)
    print(f"原始总样本数:     {total_samples}")
    print(f"训练集(平衡后):   {len(balanced_train)} (每方言 ~2000)")
    print(f"验证集(原始分布): {len(dev_items)}")
    print(f"测试集(原始分布): {len(test_items)}")
    print(f"数据格式:         {{path, sentence, dialect_id, dialect}}")
    print("=" * 60)

    # 显示样例
    print("\n前3个训练样本:")
    for i in range(min(3, len(balanced_train))):
        item = balanced_train[i]
        print(f"  [{item['dialect']}(id={item['dialect_id']})] {item['sentence'][:40]}...")

    print("\n前3个测试样本:")
    for i in range(min(3, len(test_items))):
        item = test_items[i]
        print(f"  [{item['dialect']}(id={item['dialect_id']})] {item['sentence'][:40]}...")


if __name__ == "__main__":
    main()
