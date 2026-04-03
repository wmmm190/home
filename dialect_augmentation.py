"""
方言语音数据增强模块
提供多种音频增强方法以提升模型在方言识别任务上的泛化能力
"""

import librosa
import numpy as np
import torch
import random
from typing import Tuple, Optional
import warnings


class DialectAudioAugmenter:
    """
    方言语音数据增强器
    
    支持的增强方法:
    1. 语速变化 (Time Stretching)
    2. 音调偏移 (Pitch Shifting)
    3. 噪声添加 (Noise Addition)
    4. 音量调整 (Volume Adjustment)
    5. 混响效果 (Reverberation)
    6. SpecAugment (频谱增强)
    """
    
    def __init__(self, sample_rate=16000):
        """
        Args:
            sample_rate: 采样率，默认16000Hz
        """
        self.sample_rate = sample_rate
    
    def time_stretch(self, audio: np.ndarray, rate: float = None, 
                     rate_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """
        语速变化
        
        Args:
            audio: 输入音频 [n_samples]
            rate: 拉伸比率，>1加快，<1减慢
            rate_range: 随机范围，当rate为None时使用
            
        Returns:
            stretched_audio: 变换后的音频
        """
        if rate is None:
            rate = random.uniform(rate_range[0], rate_range[1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stretched = librosa.effects.time_stretch(audio, rate=rate)
        
        return stretched
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float = None,
                   n_steps_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
        """
        音调偏移
        
        Args:
            audio: 输入音频 [n_samples]
            n_steps: 偏移半音数，正数升调，负数降调
            n_steps_range: 随机范围，当n_steps为None时使用
            
        Returns:
            shifted_audio: 变换后的音频
        """
        if n_steps is None:
            n_steps = random.uniform(n_steps_range[0], n_steps_range[1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shifted = librosa.effects.pitch_shift(
                audio, 
                sr=self.sample_rate, 
                n_steps=n_steps
            )
        
        return shifted
    
    def add_noise(self, audio: np.ndarray, noise_level: float = None,
                 snr_db_range: Tuple[float, float] = (15, 30)) -> np.ndarray:
        """
        添加高斯白噪声
        
        Args:
            audio: 输入音频 [n_samples]
            noise_level: 噪声强度（相对于信号）
            snr_db_range: 信噪比范围(dB)，当noise_level为None时使用
            
        Returns:
            noisy_audio: 添加噪声后的音频
        """
        if noise_level is None:
            # 根据SNR计算噪声强度
            snr_db = random.uniform(snr_db_range[0], snr_db_range[1])
            signal_power = np.mean(audio ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_level = np.sqrt(noise_power)
        
        noise = np.random.randn(len(audio)) * noise_level
        noisy_audio = audio + noise
        
        # 归一化以避免削波
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
        
        return noisy_audio
    
    def adjust_volume(self, audio: np.ndarray, gain_db: float = None,
                     gain_range: Tuple[float, float] = (-3, 3)) -> np.ndarray:
        """
        音量调整
        
        Args:
            audio: 输入音频 [n_samples]
            gain_db: 增益(dB)，正数增大，负数减小
            gain_range: 随机范围，当gain_db为None时使用
            
        Returns:
            adjusted_audio: 调整后的音频
        """
        if gain_db is None:
            gain_db = random.uniform(gain_range[0], gain_range[1])
        
        gain_linear = 10 ** (gain_db / 20)
        adjusted = audio * gain_linear
        
        # 归一化以避免削波
        max_val = np.max(np.abs(adjusted))
        if max_val > 1.0:
            adjusted = adjusted / max_val
        
        return adjusted
    
    def add_reverb(self, audio: np.ndarray, room_size: float = None,
                  room_size_range: Tuple[float, float] = (0.05, 0.3)) -> np.ndarray:
        """
        添加简单的混响效果
        
        Args:
            audio: 输入音频 [n_samples]
            room_size: 房间大小参数（0-1），影响混响时间
            room_size_range: 随机范围
            
        Returns:
            reverb_audio: 添加混响后的音频
        """
        if room_size is None:
            room_size = random.uniform(room_size_range[0], room_size_range[1])
        
        # 简单的延迟混响模拟
        delay_ms = int(room_size * 100)  # 延迟时间（毫秒）
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        # 创建延迟版本
        delayed = np.zeros_like(audio)
        if delay_samples < len(audio):
            delayed[delay_samples:] = audio[:-delay_samples]
        
        # 混合原始信号和延迟信号
        decay = 0.3 * room_size
        reverb = audio + decay * delayed
        
        # 归一化
        max_val = np.max(np.abs(reverb))
        if max_val > 1.0:
            reverb = reverb / max_val
        
        return reverb
    
    def spec_augment(self, mel_spec: np.ndarray, 
                    time_mask_param: int = 70,
                    freq_mask_param: int = 15,
                    num_time_masks: int = 2,
                    num_freq_masks: int = 2) -> np.ndarray:
        """
        SpecAugment频谱增强
        
        Args:
            mel_spec: Mel频谱 [n_mels, n_frames]
            time_mask_param: 时间掩码的最大宽度
            freq_mask_param: 频率掩码的最大宽度
            num_time_masks: 时间掩码数量
            num_freq_masks: 频率掩码数量
            
        Returns:
            augmented_spec: 增强后的频谱
        """
        mel_spec = mel_spec.copy()
        n_mels, n_frames = mel_spec.shape
        
        # 频率掩码
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, n_mels - f)
            mel_spec[f0:f0+f, :] = 0
        
        # 时间掩码
        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, n_frames))
            t0 = random.randint(0, n_frames - t)
            mel_spec[:, t0:t0+t] = 0
        
        return mel_spec
    
    def random_augment(self, audio: np.ndarray, 
                      augment_prob: float = 0.8,
                      apply_time_stretch: bool = True,
                      apply_pitch_shift: bool = True,
                      apply_noise: bool = True,
                      apply_volume: bool = True,
                      apply_reverb: bool = False) -> np.ndarray:
        """
        随机应用多种增强方法
        
        Args:
            audio: 输入音频
            augment_prob: 每种增强被应用的概率
            apply_*: 是否启用特定的增强方法
            
        Returns:
            augmented_audio: 增强后的音频
        """
        augmented = audio.copy()
        
        # 语速变化
        if apply_time_stretch and random.random() < augment_prob:
            augmented = self.time_stretch(augmented)
        
        # 音调偏移
        if apply_pitch_shift and random.random() < augment_prob:
            augmented = self.pitch_shift(augmented)
        
        # 音量调整
        if apply_volume and random.random() < augment_prob:
            augmented = self.adjust_volume(augmented)
        
        # 混响
        if apply_reverb and random.random() < augment_prob:
            augmented = self.add_reverb(augmented)
        
        # 噪声（通常放在最后）
        if apply_noise and random.random() < augment_prob:
            augmented = self.add_noise(augmented)
        
        return augmented


class DialectDatasetAugmented:
    """
    带增强的方言数据集包装器
    
    在加载音频时自动应用数据增强
    """
    
    def __init__(self, data_list, processor, augmenter=None, 
                 augment_prob=0.8, training=True):
        """
        Args:
            data_list: 数据列表，每项包含'path'和'sentence'
            processor: Wav2Vec2Processor
            augmenter: DialectAudioAugmenter实例
            augment_prob: 增强概率
            training: 是否为训练模式（仅训练时增强）
        """
        self.data_list = data_list
        self.processor = processor
        self.augmenter = augmenter or DialectAudioAugmenter()
        self.augment_prob = augment_prob
        self.training = training
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 加载音频
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(item['path'], sr=16000)
        
        # 训练时应用数据增强
        if self.training:
            audio = self.augmenter.random_augment(
                audio, 
                augment_prob=self.augment_prob
            )
        
        # 预处理
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest"
        )
        
        # 处理标签
        import re
        sentence = re.sub(r'[^\u4e00-\u9fa5\w\s，。、；："'']', '', item["sentence"])
        
        with torch.no_grad():
            labels = self.processor.tokenizer(
                sentence,
                return_tensors="pt",
                padding="longest"
            ).input_ids
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0) if "attention_mask" in inputs else None,
            "labels": labels.squeeze(0),
            "dialect_id": item.get('dialect_id', -1)
        }


def demo_augmentations(audio_path: str, output_dir: str = "./augmented_samples"):
    """
    演示各种数据增强效果
    
    Args:
        audio_path: 输入音频文件路径
        output_dir: 输出目录
    """
    import soundfile as sf
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 保存原始音频
    sf.write(os.path.join(output_dir, "original.wav"), audio, sr)
    
    # 创建增强器
    augmenter = DialectAudioAugmenter(sample_rate=sr)
    
    # 1. 语速变化
    print("生成语速变化样本...")
    for rate in [0.9, 1.1]:
        stretched = augmenter.time_stretch(audio, rate=rate)
        sf.write(os.path.join(output_dir, f"time_stretch_{rate}.wav"), stretched, sr)
    
    # 2. 音调偏移
    print("生成音调偏移样本...")
    for n_steps in [-2, 2]:
        shifted = augmenter.pitch_shift(audio, n_steps=n_steps)
        sf.write(os.path.join(output_dir, f"pitch_shift_{n_steps}.wav"), shifted, sr)
    
    # 3. 噪声添加
    print("生成噪声样本...")
    for snr in [15, 25]:
        noisy = augmenter.add_noise(audio, snr_db_range=(snr, snr))
        sf.write(os.path.join(output_dir, f"noise_snr{snr}.wav"), noisy, sr)
    
    # 4. 音量调整
    print("生成音量调整样本...")
    for gain in [-3, 3]:
        adjusted = augmenter.adjust_volume(audio, gain_db=gain)
        sf.write(os.path.join(output_dir, f"volume_{gain}db.wav"), adjusted, sr)
    
    # 5. 混响
    print("生成混响样本...")
    for room_size in [0.2, 0.4]:
        reverb = augmenter.add_reverb(audio, room_size=room_size)
        sf.write(os.path.join(output_dir, f"reverb_{room_size}.wav"), reverb, sr)
    
    # 6. 组合增强
    print("生成组合增强样本...")
    for i in range(3):
        combined = augmenter.random_augment(audio, augment_prob=0.8)
        sf.write(os.path.join(output_dir, f"combined_{i+1}.wav"), combined, sr)
    
    print(f"\n✓ 增强样本已保存到: {output_dir}")


if __name__ == "__main__":
    print("测试方言语音数据增强模块...")
    
    # 创建测试音频（1秒的正弦波）
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # 创建增强器
    augmenter = DialectAudioAugmenter(sample_rate=sample_rate)
    
    # 测试各种增强
    print("\n1. 测试语速变化...")
    stretched = augmenter.time_stretch(test_audio, rate=1.2)
    print(f"   原始长度: {len(test_audio)}, 变换后: {len(stretched)}")
    
    print("\n2. 测试音调偏移...")
    shifted = augmenter.pitch_shift(test_audio, n_steps=2)
    print(f"   输出长度: {len(shifted)}")
    
    print("\n3. 测试噪声添加...")
    noisy = augmenter.add_noise(test_audio, snr_db_range=(20, 20))
    snr_actual = 10 * np.log10(np.mean(test_audio**2) / np.mean((noisy - test_audio)**2))
    print(f"   实际SNR: {snr_actual:.2f} dB")
    
    print("\n4. 测试音量调整...")
    adjusted = augmenter.adjust_volume(test_audio, gain_db=6)
    print(f"   原始RMS: {np.sqrt(np.mean(test_audio**2)):.4f}")
    print(f"   调整后RMS: {np.sqrt(np.mean(adjusted**2)):.4f}")
    
    print("\n5. 测试混响...")
    reverb = augmenter.add_reverb(test_audio, room_size=0.3)
    print(f"   输出长度: {len(reverb)}")
    
    print("\n6. 测试SpecAugment...")
    mel_spec = np.random.randn(80, 100)
    augmented_spec = augmenter.spec_augment(mel_spec)
    masked_ratio = np.sum(augmented_spec == 0) / augmented_spec.size
    print(f"   掩码比例: {masked_ratio*100:.2f}%")
    
    print("\n7. 测试随机组合增强...")
    for i in range(3):
        combined = augmenter.random_augment(test_audio)
        print(f"   样本{i+1}长度: {len(combined)}")
    
    print("\n✓ 所有测试通过！")
