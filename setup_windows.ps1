# ELEC5305 Project Setup Script for Windows
# Run this in Command Prompt or PowerShell

Write-Host "===================================="
Write-Host "ELEC5305 Project Setup"
Write-Host "===================================="

# Check Python version
Write-Host "`nChecking Python version..."
python --version

# Upgrade pip
Write-Host "`nUpgrading pip..."
python -m pip install --upgrade pip

# Install packages one by one (more reliable on Windows)
Write-Host "`nInstalling audio processing libraries..."
pip install librosa soundfile audioread

Write-Host "`nInstalling scientific computing libraries..."
pip install numpy scipy

Write-Host "`nInstalling machine learning libraries..."
pip install scikit-learn

Write-Host "`nInstalling visualization libraries..."
pip install matplotlib seaborn

Write-Host "`nInstalling utilities..."
pip install tqdm pandas

Write-Host "`n===================================="
Write-Host "Installation Complete!"
Write-Host "===================================="

# Verify installation
Write-Host "`nVerifying installation..."
python -c "import librosa; import numpy; import sklearn; print('All packages imported successfully!')"

Write-Host "`nYou can now run: python demo_quickstart.py"
