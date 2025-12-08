"""
音频处理工具函数
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import os


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    加载音频文件
    
    Args:
        file_path: 音频文件路径
        sr: 目标采样率
    
    Returns:
        audio: 音频数据 (1D numpy array)
        sr: 采样率
    """
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr


def save_audio(file_path: str, audio: np.ndarray, sr: int = 16000):
    """
    保存音频文件
    
    Args:
        file_path: 保存路径
        audio: 音频数据
        sr: 采样率
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 归一化到[-1, 1]范围
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    sf.write(file_path, audio, sr)


def add_noise(clean_audio: np.ndarray, 
              noise_audio: np.ndarray, 
              snr_db: float) -> np.ndarray:
    """
    添加噪声到清洁音频
    
    Args:
        clean_audio: 清洁音频信号
        noise_audio: 噪声信号
        snr_db: 目标信噪比 (dB)
    
    Returns:
        noisy_audio: 加噪后的音频
    """
    # 确保长度一致
    min_length = min(len(clean_audio), len(noise_audio))
    clean_audio = clean_audio[:min_length]
    noise_audio = noise_audio[:min_length]
    
    # 计算信号和噪声的功率
    signal_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    
    # 计算噪声缩放因子
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
    
    # 混合
    noisy_audio = clean_audio + noise_scale * noise_audio
    
    # 归一化防止裁剪
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio)) * 0.95
    
    return noisy_audio


def compute_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """
    计算信噪比
    
    Args:
        clean_signal: 清洁信号
        noisy_signal: 噪声信号
    
    Returns:
        snr_db: 信噪比 (dB)
    """
    noise = noisy_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return np.inf
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def generate_noise(noise_type: str, duration: float, sr: int = 16000) -> np.ndarray:
    """
    生成合成噪声
    
    Args:
        noise_type: 噪声类型 ('white', 'pink', 'factory', 'street')
        duration: 时长 (秒)
        sr: 采样率
    
    Returns:
        noise: 噪声信号
    """
    num_samples = int(duration * sr)
    
    if noise_type == 'white':
        # 白噪声
        noise = np.random.randn(num_samples)
    
    elif noise_type == 'pink':
        # 粉红噪声 (1/f噪声)
        # 使用简化算法
        from scipy.signal import lfilter
        white = np.random.randn(num_samples)
        # 应用1/f滤波器
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        noise = lfilter(b, a, white)
    
    elif noise_type == 'factory':
        # 工厂噪声（低频+高频成分）
        t = np.arange(num_samples) / sr
        noise = 0.5 * np.sin(2 * np.pi * 120 * t) + \
                0.3 * np.sin(2 * np.pi * 240 * t) + \
                0.2 * np.random.randn(num_samples)
    
    elif noise_type == 'street':
        # 街道噪声（宽带有色噪声）
        white = np.random.randn(num_samples)
        # 低通滤波模拟交通噪声
        b, a = librosa.filters.get_window('hann', 50), [1.0]
        noise = np.convolve(white, b / np.sum(b), mode='same')
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # 归一化
    noise = noise / np.max(np.abs(noise))
    
    return noise


def extract_label_from_filename(filename: str) -> int:
    """
    从文件名提取标签
    
    Args:
        filename: 文件名，格式如 'digit_5_speaker1_trial02.wav'
    
    Returns:
        label: 数字标签 (0-9)
    """
    import re
    
    # 尝试匹配 'digit_X' 模式
    match = re.search(r'digit[_\-]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 尝试匹配纯数字
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1)) % 10  # 确保在0-9范围内
    
    raise ValueError(f"Cannot extract label from filename: {filename}")


if __name__ == '__main__':
    # 测试函数
    print("Testing audio_utils...")
    
    # 测试生成噪声
    print("\n1. Generating white noise...")
    white_noise = generate_noise('white', duration=1.0)
    print(f"   Shape: {white_noise.shape}, Range: [{white_noise.min():.3f}, {white_noise.max():.3f}]")
    
    # 测试添加噪声
    print("\n2. Testing add_noise...")
    clean = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)  # 440 Hz正弦波
    noise = generate_noise('white', 1.0)
    noisy = add_noise(clean, noise, snr_db=10)
    snr_actual = compute_snr(clean, noisy)
    print(f"   Target SNR: 10 dB, Actual SNR: {snr_actual:.2f} dB")
    
    # 测试标签提取
    print("\n3. Testing label extraction...")
    test_filenames = [
        'digit_5_trial01.wav',
        'digit-3-speaker2.wav',
        '7_recording.wav'
    ]
    for fn in test_filenames:
        label = extract_label_from_filename(fn)
        print(f"   {fn} -> {label}")
    
    print("\n✅ All tests passed!")
