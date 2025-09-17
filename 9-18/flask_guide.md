# Flask 入門指南 🚀

## 什麼是 Flask？

Flask 就是讓你的 Python Functions 變成網頁服務！

**之前：** 在程式裡呼叫 `greet("Alice")`  
**現在：** 在瀏覽器輸入 `http://127.0.0.1:5000/greet/Alice`

## 快速開始

### 1. 安裝 Flask
```bash
pip install flask
```

### 2. 執行程式
```bash
python app.py
```

### 3. 打開瀏覽器測試
- 首頁：`http://127.0.0.1:5000/`
- 問候：`http://127.0.0.1:5000/hello`
- 客製問候：`http://127.0.0.1:5000/greet/你的名字`
- 計算：`http://127.0.0.1:5000/add/10/20`

## 核心概念

### @app.route() 裝飾器
```python
@app.route('/hello')
def web_hello():
    return "Hello World!"
```
- `@app.route('/hello')` = 告訴 Flask 這個網址對應這個 function
- 訪問 `/hello` 就會執行 `web_hello()` 函數

### 網址參數
```python
@app.route('/greet/<name>')
def web_greet(name):
    return f"Hello {name}!"
```
- `<name>` = 網址中的變數
- 訪問 `/greet/Alice` 時，`name` 就是 `"Alice"`

### 數字參數
```python
@app.route('/add/<int:a>/<int:b>')
def web_add(a, b):
    return f"{a} + {b} = {a + b}"
```
- `<int:a>` = 強制轉換為整數
- 訪問 `/add/10/20` 時，`a=10, b=20`

## 從 Functions 到 Flask 的轉換

| 原本的 Function | Flask Route | 瀏覽器網址 |
|----------------|-------------|-----------|
| `say_hello()` | `@app.route('/hello')` | `/hello` |
| `greet("Alice")` | `@app.route('/greet/<name>')` | `/greet/Alice` |
| `add(10, 20)` | `@app.route('/add/<int:a>/<int:b>')` | `/add/10/20` |

## 練習建議

1. **修改回傳內容** - 試著改變 `return` 的文字
2. **加入新的 route** - 創建自己的網址和功能
3. **使用之前學的 functions** - 把你寫過的函數變成網頁版
4. **邀請朋友測試** - 讓別人用瀏覽器試試你的程式！

## 下一步學習

- HTML 模板
- 表單處理
- 資料庫連接
- 部署到雲端

---

*輔仁大學資訊數學系程式設計助教*  
*LINE ID: 22303248*
