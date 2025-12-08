@echo off
REM ELEC5305 Project Setup Script for Windows
REM Run this in Command Prompt

echo ====================================
echo ELEC5305 Project Setup
echo ====================================

echo.
echo Checking Python version...
python --version

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing packages...
echo This may take a few minutes...

pip install librosa
pip install soundfile  
pip install audioread
pip install numpy
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install tqdm
pip install pandas

echo.
echo ====================================
echo Installation Complete!
echo ====================================

echo.
echo Verifying installation...
python -c "import librosa; import numpy; import sklearn; print('✓ All packages installed successfully!')"

echo.
echo You can now run: python demo_quickstart.py
echo.

pause
