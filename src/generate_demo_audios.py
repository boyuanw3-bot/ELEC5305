"""
生成ICA演示所需的所有音频文件
Generate all audio files needed for ICA demonstration
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# 如果从项目目录运行，添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from audio_utils import AudioProcessor
    from ica import ICASourceSeparator
except ImportError:
    print("⚠️  警告: 无法导入audio_utils和ica模块")
    print("将使用简化版本生成音频")
    USE_SIMPLE = True
else:
    USE_SIMPLE = False


class SimpleAudioGenerator:
    """简化版音频生成器（如果无法导入完整模块时使用）"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def generate_digit(self, digit_name, duration=1.0):
        """
        生成单个数字的合成语音
        使用正弦波 + 泛音模拟
        """
        # 数字到基频的映射
        digit_freqs = {
            'zero': 200, 'one': 220, 'two': 240, 'three': 260,
            'four': 280, 'five': 300, 'six': 320, 'seven': 340,
            'eight': 360, 'nine': 380
        }
        
        f0 = digit_freqs.get(digit_name, 250)
        n_samples = int(duration * self.sr)
        t = np.linspace(0, duration, n_samples)
        
        # 基频 + 2个泛音
        signal = (np.sin(2 * np.pi * f0 * t) + 
                 0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
                 0.25 * np.sin(2 * np.pi * 3 * f0 * t))
        
        # 包络（模拟起音-保持-衰减）
        envelope = np.ones(n_samples)
        attack = int(0.05 * n_samples)
        release = int(0.1 * n_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        
        signal = signal * envelope
        
        # 归一化
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return signal


def generate_all_demo_audios(output_base_dir='E:/5305/ELEC5305_Project'):
    """
    生成所有演示音频文件
    
    Args:
        output_base_dir: 项目根目录路径
    """
    
    print("=" * 70)
    print("  生成ICA演示音频文件")
    print("=" * 70)
    
    # 创建输出目录
    output_base = Path(output_base_dir)
    demo_dir = output_base / 'data' / 'audio_demos'
    results_input_dir = output_base / 'results' / 'audio_samples' / 'input'
    results_output_dir = output_base / 'results' / 'audio_samples' / 'output'
    
    demo_dir.mkdir(parents=True, exist_ok=True)
    results_input_dir.mkdir(parents=True, exist_ok=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 输出目录:")
    print(f"  - {demo_dir}")
    print(f"  - {results_input_dir}")
    print(f"  - {results_output_dir}")
    
    # 初始化生成器
    sr = 16000
    
    if USE_SIMPLE:
        print("\n⚙️  使用简化版音频生成器")
        generator = SimpleAudioGenerator(sr=sr)
        
        # 生成数字3和7
        print("\n1️⃣  生成原始语音信号...")
        digit_3 = generator.generate_digit('three', duration=1.0)
        digit_7 = generator.generate_digit('seven', duration=1.0)
        
    else:
        print("\n⚙️  使用完整版音频生成器")
        processor = AudioProcessor(sample_rate=sr)
        
        # 生成数字3和7
        print("\n1️⃣  生成原始语音信号...")
        digit_3 = processor.generate_speech('three')
        digit_7 = processor.generate_speech('seven')
    
    # 确保长度一致
    min_len = min(len(digit_3), len(digit_7))
    digit_3 = digit_3[:min_len]
    digit_7 = digit_7[:min_len]
    
    print(f"   ✓ digit_3: {len(digit_3)} 采样点 ({len(digit_3)/sr:.2f}秒)")
    print(f"   ✓ digit_7: {len(digit_7)} 采样点 ({len(digit_7)/sr:.2f}秒)")
    
    # 保存原始清晰语音
    print("\n2️⃣  保存原始清晰语音...")
    sf.write(demo_dir / 'demo_clean_3.wav', digit_3, sr)
    print(f"   ✓ {demo_dir / 'demo_clean_3.wav'}")
    
    sf.write(demo_dir / 'demo_clean_7.wav', digit_7, sr)
    print(f"   ✓ {demo_dir / 'demo_clean_7.wav'}")
    
    # 创建混合信号
    print("\n3️⃣  创建混合信号...")
    
    # 混合矩阵 (模拟双麦克风接收)
    A = np.array([[0.8, 0.6],
                  [0.4, 0.9]])
    
    print(f"   混合矩阵 A = \n{A}")
    
    # 堆叠源信号
    sources = np.vstack([digit_3, digit_7])
    print(f"   源信号形状: {sources.shape}")
    
    # 混合
    mixed = A @ sources
    print(f"   混合信号形状: {mixed.shape}")
    
    # 保存混合信号 (第一个通道)
    sf.write(demo_dir / 'demo_mixed.wav', mixed[0], sr)
    print(f"   ✓ {demo_dir / 'demo_mixed.wav'}")
    
    # 同时保存到results/input
    sf.write(results_input_dir / 'mixed_noisy.wav', mixed[0], sr)
    print(f"   ✓ {results_input_dir / 'mixed_noisy.wav'}")
    
    # ICA分离
    print("\n4️⃣  执行ICA盲源分离...")
    
    if USE_SIMPLE:
        # 简化版ICA (使用sklearn)
        from sklearn.decomposition import FastICA
        
        ica = FastICA(n_components=2, max_iter=200, random_state=42)
        
        # 转置以符合sklearn格式 (n_samples, n_features)
        mixed_T = mixed.T
        separated_T = ica.fit_transform(mixed_T)
        separated = separated_T.T
        
        print("   ✓ ICA收敛完成")
        
    else:
        # 使用自定义ICA模块
        ica_separator = ICASourceSeparator(n_components=2)
        separated = ica_separator.separate(mixed, return_all=False)
        print("   ✓ ICA分离完成")
    
    print(f"   分离信号形状: {separated.shape}")
    
    # 源对齐 (找出哪个是digit_3, 哪个是digit_7)
    print("\n5️⃣  对齐分离源...")
    
    # 计算与原始信号的相关性
    corr_3_0 = np.abs(np.corrcoef(separated[0], digit_3)[0, 1])
    corr_3_1 = np.abs(np.corrcoef(separated[1], digit_3)[0, 1])
    
    corr_7_0 = np.abs(np.corrcoef(separated[0], digit_7)[0, 1])
    corr_7_1 = np.abs(np.corrcoef(separated[1], digit_7)[0, 1])
    
    print(f"   相关性分析:")
    print(f"   - separated[0] vs digit_3: {corr_3_0:.3f}")
    print(f"   - separated[0] vs digit_7: {corr_7_0:.3f}")
    print(f"   - separated[1] vs digit_3: {corr_3_1:.3f}")
    print(f"   - separated[1] vs digit_7: {corr_7_1:.3f}")
    
    # 判断哪个是3，哪个是7
    if corr_3_0 > corr_3_1:
        # separated[0]是3, separated[1]是7
        separated_3 = separated[0]
        separated_7 = separated[1]
        print(f"   ✓ 对齐结果: source_1=digit_3, source_2=digit_7")
    else:
        # separated[1]是3, separated[0]是7
        separated_3 = separated[1]
        separated_7 = separated[0]
        print(f"   ✓ 对齐结果: source_1=digit_7, source_2=digit_3 (已交换)")
    
    # 归一化分离信号
    separated_3 = separated_3 / np.max(np.abs(separated_3)) * 0.8
    separated_7 = separated_7 / np.max(np.abs(separated_7)) * 0.8
    
    # 保存分离结果
    print("\n6️⃣  保存分离结果...")
    
    # 保存到demo目录
    sf.write(demo_dir / 'demo_separated_1.wav', separated_3, sr)
    print(f"   ✓ {demo_dir / 'demo_separated_1.wav'} (digit_3)")
    
    sf.write(demo_dir / 'demo_separated_2.wav', separated_7, sr)
    print(f"   ✓ {demo_dir / 'demo_separated_2.wav'} (digit_7)")
    
    # 保存到results/output
    sf.write(results_output_dir / 'separated_source1.wav', separated_3, sr)
    print(f"   ✓ {results_output_dir / 'separated_source1.wav'}")
    
    sf.write(results_output_dir / 'separated_source2.wav', separated_7, sr)
    print(f"   ✓ {results_output_dir / 'separated_source2.wav'}")
    
    # 计算分离质量指标
    print("\n7️⃣  计算分离质量指标...")
    
    # 信号失真比 (SDR)
    def compute_sdr(estimated, reference):
        """计算信号失真比"""
        # 确保长度一致
        min_len = min(len(estimated), len(reference))
        estimated = estimated[:min_len]
        reference = reference[:min_len]
        
        # SDR = 10 * log10(||s||^2 / ||s - s_hat||^2)
        signal_power = np.sum(reference ** 2)
        error_power = np.sum((reference - estimated) ** 2)
        
        if error_power < 1e-10:
            return 100.0  # 近乎完美
        
        sdr = 10 * np.log10(signal_power / error_power)
        return sdr
    
    sdr_3 = compute_sdr(separated_3, digit_3)
    sdr_7 = compute_sdr(separated_7, digit_7)
    
    print(f"   SDR (digit_3): {sdr_3:.2f} dB")
    print(f"   SDR (digit_7): {sdr_7:.2f} dB")
    print(f"   平均 SDR: {(sdr_3 + sdr_7)/2:.2f} dB")
    
    # 生成摘要报告
    print("\n" + "=" * 70)
    print("  ✅ 所有音频文件生成完成！")
    print("=" * 70)
    
    print(f"\n📊 生成文件清单:")
    print(f"\n【演示音频目录】{demo_dir}")
    print(f"  1. demo_clean_3.wav      - 原始清晰的数字3")
    print(f"  2. demo_clean_7.wav      - 原始清晰的数字7")
    print(f"  3. demo_mixed.wav        - 混合信号 (3+7)")
    print(f"  4. demo_separated_1.wav  - ICA分离出的数字3")
    print(f"  5. demo_separated_2.wav  - ICA分离出的数字7")
    
    print(f"\n【实验结果目录】{results_input_dir}")
    print(f"  1. mixed_noisy.wav       - 混合信号 (副本)")
    
    print(f"\n【实验结果目录】{results_output_dir}")
    print(f"  1. separated_source1.wav - 分离信号1 (digit_3)")
    print(f"  2. separated_source2.wav - 分离信号2 (digit_7)")
    
    print(f"\n📈 分离质量:")
    print(f"  - SDR (digit_3): {sdr_3:.2f} dB")
    print(f"  - SDR (digit_7): {sdr_7:.2f} dB")
    print(f"  - 平均 SDR: {(sdr_3 + sdr_7)/2:.2f} dB")
    
    print(f"\n🎬 视频录制播放顺序:")
    print(f"  1. demo_clean_3.wav      (3秒) - '这是清晰的数字3'")
    print(f"  2. demo_clean_7.wav      (3秒) - '这是清晰的数字7'")
    print(f"  3. demo_mixed.wav        (5秒) - '现在它们混在一起了'")
    print(f"  4. demo_separated_1.wav  (3秒) - 'ICA分离出的第一个信号'")
    print(f"  5. demo_separated_2.wav  (3秒) - 'ICA分离出的第二个信号'")
    
    print("\n" + "=" * 70)
    
    return {
        'demo_dir': demo_dir,
        'results_input_dir': results_input_dir,
        'results_output_dir': results_output_dir,
        'sdr_3': sdr_3,
        'sdr_7': sdr_7
    }


if __name__ == '__main__':
    # 可以通过命令行参数指定输出目录
    import argparse
    
    parser = argparse.ArgumentParser(description='生成ICA演示音频文件')
    parser.add_argument('--output', '-o', 
                       default='E:/5305/ELEC5305_Project',
                       help='项目根目录路径 (默认: E:/5305/ELEC5305_Project)')
    
    args = parser.parse_args()
    
    try:
        results = generate_all_demo_audios(args.output)
        print("\n✅ 成功！所有文件已生成。")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
