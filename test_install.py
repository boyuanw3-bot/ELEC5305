#!/usr/bin/env python3
"""
测试安装是否成功
Test Installation Script
"""

import sys

print("=" * 60)
print("ELEC5305 Project - Installation Test")
print("=" * 60)

# 测试导入
packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'librosa': 'Librosa',
    'soundfile': 'SoundFile',
    'sklearn': 'Scikit-learn',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'pandas': 'Pandas',
    'tqdm': 'tqdm'
}

print("\n检查已安装的包...")
print("-" * 60)

failed = []
for module, name in packages.items():
    try:
        if module == 'sklearn':
            import sklearn
            version = sklearn.__version__
        else:
            mod = __import__(module)
            version = mod.__version__ if hasattr(mod, '__version__') else '未知'
        
        print(f"✓ {name:20} v{version}")
    except ImportError:
        print(f"✗ {name:20} 未安装")
        failed.append(name)

print("-" * 60)

if failed:
    print(f"\n❌ 以下包未安装: {', '.join(failed)}")
    print("\n请运行以下命令安装:")
    print(f"pip install {' '.join([p.lower() for p in failed])}")
    sys.exit(1)

# 测试基本功能
print("\n测试基本功能...")
print("-" * 60)

try:
    import numpy as np
    print("✓ NumPy: 创建数组")
    arr = np.array([1, 2, 3])
    
    import librosa
    print("✓ Librosa: 生成测试音频")
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
    
    print("✓ Librosa: 提取MFCC特征")
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    print(f"  MFCC shape: {mfcc.shape}")
    
    from sklearn.mixture import GaussianMixture
    print("✓ Scikit-learn: 创建GMM模型")
    gmm = GaussianMixture(n_components=2)
    
    import matplotlib.pyplot as plt
    print("✓ Matplotlib: 创建图表")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.close(fig)
    
    print("-" * 60)
    print("\n✅ 所有测试通过！")
    print("\n🚀 你可以运行以下命令开始演示:")
    print("   python demo_quickstart.py")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
