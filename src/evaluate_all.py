"""
综合评估脚本 - 对比四种配置
对比实验配置：
1. Baseline: 标准MFCC特征
2. ICA增强: ICA + MFCC
3. 耳蜗特征: Gammatone耳蜗滤波器特征
4. 完整系统: ICA + Gammatone + MFCC

作者: ELEC5305 Course Project
日期: 2024
"""

import os
import sys
import numpy as np
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
import os
import sys
from pathlib import Path

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# 导入项目模块
try:
    from config import Config
    from audio_utils import AudioProcessor
    from ica import ICAModule
    from gammatone import GammatoneFilterBank
    from baseline_mfcc import BaselineMFCC
except ImportError:
    import config
    import audio_utils
    import ica
    import gammatone
    import baseline_mfcc
    Config = config.Config
    AudioProcessor = audio_utils.AudioProcessor
    ICAModule = ica.ICAModule
    GammatoneFilterBank = gammatone.GammatoneFilterBank
    BaselineMFCC = baseline_mfcc.BaselineMFCC
from baseline_mfcc import BaselineMFCC

class ComprehensiveEvaluator:
    """综合评估器 - 对比四种配置"""
    
    def __init__(self):
        self.config = Config()
        self.audio_processor = AudioProcessor(self.config)
        self.ica_module = ICAModule()
        self.gammatone = GammatoneFilterBank(
            n_filters=32,
            sample_rate=self.config.sample_rate
        )
        
        # 存储结果
        self.results = {
            'baseline': {},
            'ica_enhanced': {},
            'cochlear': {},
            'full_system': {}
        }
        
    def generate_test_signals(self, n_samples=10, snr_db=0):    #origional  10
        """
        生成测试信号集
        
        参数:
            n_samples: 每个词汇的样本数
            snr_db: 信噪比（dB）
            
        返回:
            clean_signals: 干净信号列表
            noisy_signals: 噪声污染信号列表
            labels: 标签列表
        """
        print(f"\n生成测试数据集...")
        print(f"  每个词汇样本数: {n_samples}")
        print(f"  信噪比: {snr_db} dB")
        
        clean_signals = []
        noisy_signals = []
        labels = []
        
        for word_idx, word in enumerate(self.config.vocabulary):
            for i in range(n_samples):
                # 生成干净信号
                signal = self.audio_processor.generate_speech(word)
                clean_signals.append(signal)
                labels.append(word_idx)
                
                # 添加噪声
                noise = np.random.randn(len(signal))
                signal_power = np.mean(signal ** 2)
                noise_power = np.mean(noise ** 2)
                noise_scale = np.sqrt(signal_power / (10 ** (snr_db / 10)) / noise_power)
                noisy_signal = signal + noise_scale * noise
                noisy_signals.append(noisy_signal)
        
        print(f"  ✓ 生成 {len(clean_signals)} 个测试样本")
        
        return clean_signals, noisy_signals, np.array(labels)
    
    def extract_baseline_features(self, signals):
        """提取基线MFCC特征"""
        features = []
        for signal in signals:
            mfcc = self.audio_processor.extract_mfcc(signal)
            # 展平为一维特征向量
            feat = mfcc.flatten()
            features.append(feat)
        return np.array(features)
    
    def extract_ica_enhanced_features(self, noisy_signals, clean_signals):
        """提取ICA增强的MFCC特征"""
        features = []
        
        for noisy_signal, clean_signal in zip(noisy_signals, clean_signals):
            # 创建混合信号矩阵 (2通道)
            mixed_signals = np.vstack([noisy_signal, clean_signal])
            
            # ICA分离
            separated = self.ica_module.separate(mixed_signals)
            
            # 对齐并选择最佳分离信号
            best_idx = self.ica_module.align_sources(separated, clean_signal)
            enhanced_signal = separated[best_idx]
            
            # 提取MFCC
            mfcc = self.audio_processor.extract_mfcc(enhanced_signal)
            feat = mfcc.flatten()
            features.append(feat)
            
        return np.array(features)
    
    def extract_cochlear_features(self, signals):
        """提取Gammatone耳蜗特征"""
        features = []
        
        for signal in signals:
            # 应用Gammatone滤波器组
            filtered = self.gammatone.filter(signal)
            
            # 提取耳蜗特征（包含delta特征）
            cochlear_feat = self.gammatone.extract_features(filtered)
            
            # 展平为一维特征向量
            feat = cochlear_feat.flatten()
            features.append(feat)
            
        return np.array(features)
    
    def extract_full_system_features(self, noisy_signals, clean_signals):
        """提取完整系统特征 (ICA + Gammatone + MFCC)"""
        features = []
        
        for noisy_signal, clean_signal in zip(noisy_signals, clean_signals):
            # 1. ICA分离
            mixed_signals = np.vstack([noisy_signal, clean_signal])
            separated = self.ica_module.separate(mixed_signals)
            best_idx = self.ica_module.align_sources(separated, clean_signal)
            enhanced_signal = separated[best_idx]
            
            # 2. Gammatone耳蜗特征
            filtered = self.gammatone.filter(enhanced_signal)
            cochlear_feat = self.gammatone.extract_features(filtered)
            
            # 3. MFCC特征
            mfcc = self.audio_processor.extract_mfcc(enhanced_signal)
            
            # 4. 特征融合
            feat_cochlear = cochlear_feat.flatten()
            feat_mfcc = mfcc.flatten()
            
            # 归一化后拼接
            feat_cochlear = (feat_cochlear - feat_cochlear.mean()) / (feat_cochlear.std() + 1e-8)
            feat_mfcc = (feat_mfcc - feat_mfcc.mean()) / (feat_mfcc.std() + 1e-8)
            
            feat = np.concatenate([feat_cochlear, feat_mfcc])
            features.append(feat)
            
        return np.array(features)
    
    def evaluate_configuration(self, features, labels, config_name):
        """
        评估单个配置
        
        参数:
            features: 特征矩阵 (n_samples, n_features)
            labels: 标签向量 (n_samples,)
            config_name: 配置名称
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report
        
        print(f"\n{'='*70}")
        print(f"评估配置: {config_name}")
        print(f"{'='*70}")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        print(f"  训练集大小: {X_train.shape[0]}")
        print(f"  测试集大小: {X_test.shape[0]}")
        print(f"  特征维度: {X_train.shape[1]}")
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练SVM分类器
        print(f"\n  训练SVM分类器...")
        start_time = time.time()
        
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        print(f"    ✓ 训练完成 (耗时: {training_time:.2f}秒)")
        
        # 预测
        y_pred = clf.predict(X_test_scaled)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n  测试准确率: {accuracy*100:.2f}%")
        
        # 分类报告
        print(f"\n  详细分类报告:")
        target_names = self.config.vocabulary
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)
        
        # 存储结果
        self.results[config_name] = {
            'accuracy': accuracy,
            'training_time': training_time,
            'n_features': X_train.shape[1],
            'report': report
        }
        
        return accuracy
    
    def run_all_experiments(self, n_samples=10, snr_db=10):
        """
        运行所有实验配置
        
        参数:
            n_samples: 每个词汇的样本数
            snr_db: 信噪比（dB）
        """
        print("\n" + "="*70)
        print(" 综合评估实验 - 对比四种配置")
        print("="*70)
        
        # 生成测试数据
        clean_signals, noisy_signals, labels = self.generate_test_signals(
            n_samples=n_samples, snr_db=snr_db
        )
        
        # 配置1: Baseline MFCC
        print(f"\n\n配置1: 标准MFCC特征（基线）")
        print("-" * 70)
        features_baseline = self.extract_baseline_features(noisy_signals)
        self.evaluate_configuration(features_baseline, labels, 'baseline')
        
        # 配置2: ICA + MFCC
        print(f"\n\n配置2: ICA增强 + MFCC特征")
        print("-" * 70)
        features_ica = self.extract_ica_enhanced_features(noisy_signals, clean_signals)
        self.evaluate_configuration(features_ica, labels, 'ica_enhanced')
        
        # 配置3: Gammatone耳蜗特征
        print(f"\n\n配置3: Gammatone耳蜗滤波器特征")
        print("-" * 70)
        features_cochlear = self.extract_cochlear_features(noisy_signals)
        self.evaluate_configuration(features_cochlear, labels, 'cochlear')
        
        # 配置4: ICA + Gammatone + MFCC
        print(f"\n\n配置4: 完整系统 (ICA + Gammatone + MFCC)")
        print("-" * 70)
        features_full = self.extract_full_system_features(noisy_signals, clean_signals)
        self.evaluate_configuration(features_full, labels, 'full_system')
        
        # 打印对比总结
        self.print_summary()
    
    def print_summary(self):
        """打印对比总结"""
        print("\n\n" + "="*70)
        print(" 实验结果对比总结")
        print("="*70)
        
        print(f"\n{'配置名称':<30} {'准确率':<15} {'特征维度':<15} {'训练时间':<15}")
        print("-" * 70)
        
        config_names = {
            'baseline': '基线MFCC',
            'ica_enhanced': 'ICA + MFCC',
            'cochlear': 'Gammatone耳蜗',
            'full_system': '完整系统'
        }
        
        for config_key, config_display in config_names.items():
            if config_key in self.results:
                result = self.results[config_key]
                accuracy = result['accuracy'] * 100
                n_features = result['n_features']
                train_time = result['training_time']
                
                print(f"{config_display:<30} {accuracy:>6.2f}%        {n_features:>8}        {train_time:>6.2f}秒")
        
        # 找出最佳配置
        best_config = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_name = config_names[best_config[0]]
        best_acc = best_config[1]['accuracy'] * 100
        
        print("\n" + "="*70)
        print(f"✅ 最佳配置: {best_name} (准确率: {best_acc:.2f}%)")
        print("="*70)
        
        # 计算改进百分比
        baseline_acc = self.results['baseline']['accuracy']
        full_acc = self.results['full_system']['accuracy']
        improvement = ((full_acc - baseline_acc) / baseline_acc) * 100
        
        print(f"\n完整系统相对基线的改进: {improvement:+.2f}%")
        print(f"  基线准确率: {baseline_acc*100:.2f}%")
        print(f"  完整系统准确率: {full_acc*100:.2f}%")
        

def main():
    """主函数"""
    print("\n" + "="*70)
    print(" ELEC5305 Project - 综合评估实验")
    print(" 感知增强语音识别系统")
    print("="*70)
    
    evaluator = ComprehensiveEvaluator()
    
    # 运行实验
    # 参数可调整：n_samples控制数据集大小，snr_db控制噪声水平
    evaluator.run_all_experiments(n_samples=20, snr_db=10)
    
    print("\n✅ 所有实验完成！\n")


if __name__ == "__main__":
    main()
