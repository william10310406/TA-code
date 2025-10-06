"""
Simple Perceptron Implementation (Without Learning Rate)
簡單感知機實作（無學習率版本）

Based on Rosenblatt's early formulation of the Perceptron Convergence Theorem.
基於 Rosenblatt 早期感知機收斂定理的表述。

Author: Teaching Assistant
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class SimplePerceptron:
    """
    Simple Perceptron without learning rate (Rosenblatt's early form)
    簡單感知機（無學習率，Rosenblatt 早期形式）
    
    This implementation follows the classical error-correction rule:
    - If prediction is wrong: w = w + x (for positive class) or w = w - x (for negative class)
    - If prediction is correct: no update
    
    這個實作遵循經典的錯誤修正規則：
    - 如果預測錯誤：w = w + x（正類）或 w = w - x（負類）
    - 如果預測正確：不更新
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Initialize the perceptron
        初始化感知機
        
        Args:
            threshold: Decision threshold θ (default: 0.0)
                      決策閾值 θ（預設：0.0）
        """
        self.weights = None
        self.threshold = threshold
        self.training_history = []
        self.converged = False
        self.n_updates = 0
    
    def predict(self, x: np.ndarray) -> int:
        """
        Make prediction for input x
        對輸入 x 進行預測
        
        Args:
            x: Input vector (numpy array)
               輸入向量
        
        Returns:
            +1 if w·x > θ, -1 otherwise
            如果 w·x > θ 則返回 +1，否則返回 -1
        """
        if self.weights is None:
            raise ValueError("Model not trained yet! 模型尚未訓練！")
        
        activation = np.dot(self.weights, x)
        return 1 if activation > self.threshold else -1
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 1000) -> bool:
        """
        Train the perceptron using error-correction rule
        使用錯誤修正規則訓練感知機
        
        Args:
            X: Training data (n_samples, n_features)
               訓練資料
            y: Training labels (+1 or -1)
               訓練標籤（+1 或 -1）
            max_epochs: Maximum number of epochs
                       最大訓練輪數
        
        Returns:
            True if converged, False if max_epochs reached
            如果收斂則返回 True，達到最大輪數則返回 False
        """
        n_samples, n_features = X.shape
        
        # Initialize weights to zero (classical approach)
        # 將權重初始化為零（經典方法）
        self.weights = np.zeros(n_features)
        self.training_history = []
        self.n_updates = 0
        
        print(f"開始訓練感知機...")
        print(f"資料點數量: {n_samples}, 特徵維度: {n_features}")
        print(f"閾值 θ = {self.threshold}")
        print("-" * 50)
        
        for epoch in range(max_epochs):
            n_errors = 0
            epoch_updates = []
            
            # Go through all training samples
            # 遍歷所有訓練樣本
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                
                # Make prediction
                # 進行預測
                prediction = self.predict(x_i)
                
                # Check if prediction is wrong
                # 檢查預測是否錯誤
                if prediction != y_i:
                    # Error correction update (no learning rate!)
                    # 錯誤修正更新（無學習率！）
                    if y_i == 1:
                        self.weights += x_i  # w = w + x
                    else:
                        self.weights -= x_i  # w = w - x
                    
                    self.n_updates += 1
                    n_errors += 1
                    
                    epoch_updates.append({
                        'sample_idx': i,
                        'input': x_i.copy(),
                        'true_label': y_i,
                        'prediction': prediction,
                        'weights_after': self.weights.copy()
                    })
            
            # Record training history
            # 記錄訓練歷史
            self.training_history.append({
                'epoch': epoch + 1,
                'n_errors': n_errors,
                'weights': self.weights.copy(),
                'updates': epoch_updates
            })
            
            print(f"Epoch {epoch + 1:3d}: {n_errors} 個錯誤, 權重 = {self.weights}")
            
            # Check convergence
            # 檢查收斂
            if n_errors == 0:
                self.converged = True
                print(f"\n🎉 感知機已收斂！")
                print(f"總更新次數: {self.n_updates}")
                print(f"最終權重: {self.weights}")
                return True
        
        print(f"\n⚠️  達到最大訓練輪數 ({max_epochs})，未完全收斂")
        return False
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                             title: str = "感知機決策邊界"):
        """
        Plot decision boundary (only works for 2D data)
        繪製決策邊界（僅適用於 2D 資料）
        """
        if X.shape[1] != 2:
            print("只能繪製 2D 資料的決策邊界")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        # 繪製資料點
        positive_mask = y == 1
        negative_mask = y == -1
        
        plt.scatter(X[positive_mask, 0], X[positive_mask, 1], 
                   c='red', marker='o', s=100, label='正類 (+1)', alpha=0.7)
        plt.scatter(X[negative_mask, 0], X[negative_mask, 1], 
                   c='blue', marker='s', s=100, label='負類 (-1)', alpha=0.7)
        
        # Plot decision boundary
        # 繪製決策邊界
        if self.weights is not None:
            w1, w2 = self.weights
            
            # Decision boundary: w1*x1 + w2*x2 = θ
            # 決策邊界：w1*x1 + w2*x2 = θ
            x_min, x_max = plt.xlim()
            
            if abs(w2) > 1e-10:  # Avoid division by zero
                x_boundary = np.linspace(x_min, x_max, 100)
                y_boundary = (self.threshold - w1 * x_boundary) / w2
                plt.plot(x_boundary, y_boundary, 'g-', linewidth=2, 
                        label=f'決策邊界: {w1:.2f}x₁ + {w2:.2f}x₂ = {self.threshold}')
            else:
                # Vertical line
                x_boundary = self.threshold / w1 if abs(w1) > 1e-10 else 0
                plt.axvline(x=x_boundary, color='g', linewidth=2, 
                           label=f'決策邊界: x₁ = {x_boundary:.2f}')
        
        plt.xlabel('特徵 1 (x₁)')
        plt.ylabel('特徵 2 (x₂)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    
    def get_training_summary(self) -> dict:
        """
        Get summary of training process
        獲取訓練過程摘要
        """
        return {
            'converged': self.converged,
            'n_epochs': len(self.training_history),
            'n_updates': self.n_updates,
            'final_weights': self.weights.copy() if self.weights is not None else None,
            'threshold': self.threshold
        }


def create_linearly_separable_data(n_samples: int = 20, 
                                 random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple linearly separable 2D dataset
    創建簡單的線性可分 2D 資料集
    
    Args:
        n_samples: Number of samples per class
                  每類的樣本數量
        random_state: Random seed for reproducibility
                     隨機種子
    
    Returns:
        X: Feature matrix (2*n_samples, 2)
        y: Labels (2*n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate positive class samples (upper right)
    # 生成正類樣本（右上方）
    X_pos = np.random.randn(n_samples, 2) + [2, 2]
    y_pos = np.ones(n_samples)
    
    # Generate negative class samples (lower left)
    # 生成負類樣本（左下方）
    X_neg = np.random.randn(n_samples, 2) + [-2, -2]
    y_neg = -np.ones(n_samples)
    
    # Combine data
    # 合併資料
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    
    # Shuffle data
    # 打亂資料
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def demonstrate_perceptron():
    """
    Demonstrate the perceptron algorithm
    演示感知機演算法
    """
    print("=" * 60)
    print("感知機演算法演示 (Perceptron Algorithm Demonstration)")
    print("=" * 60)
    
    # Create dataset
    # 創建資料集
    print("\n1. 創建線性可分資料集...")
    X, y = create_linearly_separable_data(n_samples=10, random_state=42)
    print(f"資料形狀: {X.shape}")
    print(f"標籤: {np.unique(y, return_counts=True)}")
    
    # Initialize and train perceptron
    # 初始化並訓練感知機
    print("\n2. 初始化感知機...")
    perceptron = SimplePerceptron(threshold=0.0)
    
    print("\n3. 開始訓練...")
    success = perceptron.fit(X, y, max_epochs=100)
    
    # Show results
    # 顯示結果
    print("\n4. 訓練結果:")
    summary = perceptron.get_training_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Test predictions
    # 測試預測
    print("\n5. 測試預測:")
    for i in range(min(5, len(X))):
        pred = perceptron.predict(X[i])
        print(f"   樣本 {i}: 真實={int(y[i]):2d}, 預測={pred:2d}, "
              f"輸入={X[i]}, 正確={pred == y[i]}")
    
    # Plot results
    # 繪製結果
    print("\n6. 繪製決策邊界...")
    perceptron.plot_decision_boundary(X, y, "感知機學習結果")
    
    return perceptron, X, y


if __name__ == "__main__":
    # Run demonstration
    # 執行演示
    perceptron, X, y = demonstrate_perceptron()
