#!/usr/bin/env python3
"""
感知機教學啟動腳本
自動檢查環境並啟動 Jupyter Notebook
"""

import sys
import subprocess
import os

def check_and_install_package(package_name):
    """檢查並安裝套件"""
    try:
        __import__(package_name)
        print(f"✅ {package_name} 已安裝")
        return True
    except ImportError:
        print(f"📦 正在安裝 {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} 安裝成功")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ {package_name} 安裝失敗")
            return False

def main():
    print("🚀 感知機教學環境檢查")
    print("=" * 50)
    
    # 檢查必要套件
    required_packages = ["numpy", "matplotlib", "notebook"]
    all_installed = True
    
    for package in required_packages:
        if not check_and_install_package(package):
            all_installed = False
    
    if not all_installed:
        print("\n❌ 有套件安裝失敗，請手動安裝：")
        print("pip install numpy matplotlib notebook")
        return False
    
    # 檢查感知機模組
    try:
        from simple_perceptron import SimplePerceptron
        print("✅ 感知機模組可用")
    except ImportError:
        print("❌ 找不到 simple_perceptron.py，請確認檔案在當前目錄")
        return False
    
    print("\n🎉 所有套件都已準備就緒！")
    print("=" * 50)
    
    # 啟動 Jupyter Notebook
    print("正在啟動 Jupyter Notebook...")
    try:
        subprocess.run([sys.executable, "-m", "notebook", "perceptron_tutorial.ipynb"])
    except KeyboardInterrupt:
        print("\n👋 Jupyter Notebook 已關閉")
    except Exception as e:
        print(f"❌ 啟動 Jupyter 失敗: {e}")
        print("請手動執行：jupyter notebook perceptron_tutorial.ipynb")
    
    return True

if __name__ == "__main__":
    # 確保在正確的目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
