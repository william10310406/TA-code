# 感知機教學檔案使用指南

## 📁 檔案說明

- `perceptron_convergence_proof.pdf` - 感知機收斂定理的嚴謹數學證明
- `simple_perceptron.py` - 感知機 Python 實作（無學習率版本）
- `perceptron_tutorial.ipynb` - 互動式教學 Jupyter Notebook
- `README.md` - 本使用指南

## 🛠️ 環境設定

### 必要套件

請確保您的 Python 環境中安裝了以下套件：

```bash
pip install numpy matplotlib jupyter
```

### 檢查安裝

在終端機中執行以下命令來檢查套件是否正確安裝：

```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
```

## 🚀 使用方式

### 方法 1：一鍵啟動（最簡單）

```bash
cd "/Users/jianweiheng/Desktop/TA code/10-9"
python start_tutorial.py
```

這個腳本會：
- 自動檢查和安裝所需套件
- 驗證環境設定
- 直接開啟教學 Notebook

### 方法 2：手動啟動 Jupyter Notebook

1. 啟動 Jupyter Notebook：
   ```bash
   cd "/Users/jianweiheng/Desktop/TA code/10-9"
   jupyter notebook
   ```

2. 在瀏覽器中開啟 `perceptron_tutorial.ipynb`

3. **重要**：如果遇到套件導入問題，請在 Notebook 中：
   - 點選 "Kernel" → "Change Kernel" → 選擇 "Perceptron Tutorial"
   - 或直接執行第一個 cell，它會自動安裝缺失的套件

### 方法 3：直接執行 Python 檔案

```bash
cd "/Users/jianweiheng/Desktop/TA code/10-9"
python simple_perceptron.py
```

這會執行感知機的演示程式，展示完整的訓練過程。

### 方法 4：環境測試

```bash
cd "/Users/jianweiheng/Desktop/TA code/10-9"
python test_environment.py
```

這會測試所有套件是否正確安裝。

## 🔧 常見問題解決

### 問題 1：ModuleNotFoundError: No module named 'numpy'

**解決方案：**
```bash
pip install numpy matplotlib
```

如果使用 conda：
```bash
conda install numpy matplotlib
```

### 問題 2：Jupyter Notebook 找不到套件

**可能原因：** Jupyter 使用的 Python 環境與您安裝套件的環境不同

**解決方案：**
1. 確認 Jupyter 使用的 Python 環境：
   ```python
   import sys
   print(sys.executable)
   ```

2. 在正確的環境中安裝套件：
   ```bash
   /path/to/correct/python -m pip install numpy matplotlib
   ```

3. 或者重新安裝 Jupyter：
   ```bash
   pip install --upgrade jupyter
   ```

### 問題 3：中文字體顯示問題

**症狀：** 圖表中的中文顯示為方塊

**解決方案：**
- macOS：系統通常有 PingFang TC 字體，程式會自動處理
- Windows：可能需要安裝中文字體或修改字體設定
- Linux：可能需要安裝中文字體套件

**臨時解決方案：** 將圖表標題改為英文

## 📚 教學內容

### 第一部分：歷史背景
- 1957年 Rosenblatt 的感知機
- 機器學習的起源
- 神經網路的基礎

### 第二部分：理論基礎
- 感知機收斂定理
- 線性可分性
- 幾何直觀

### 第三部分：程式實作
- 無學習率的感知機
- 錯誤修正規則
- 視覺化學習過程

### 第四部分：實驗探索
- 不同資料集測試
- 權重變化軌跡
- 動手練習

## 🎯 學習目標

完成本教學後，學生將能夠：

1. **理解歷史**：知道感知機在機器學習史上的重要地位
2. **掌握理論**：理解感知機收斂定理的核心概念
3. **實作演算法**：能夠從零實作感知機演算法
4. **分析結果**：能夠解讀學習過程和結果
5. **獨立探索**：能夠設計實驗驗證理論

## 💡 延伸學習

- 多層感知機 (MLP)
- 支援向量機 (SVM)
- 邏輯回歸
- 深度學習基礎

## 📞 技術支援

如果遇到技術問題，請：

1. 檢查 Python 和套件版本
2. 確認所有檔案在同一資料夾
3. 查看錯誤訊息的詳細內容
4. 嘗試重新安裝相關套件

---

**祝您學習愉快！** 🎉
