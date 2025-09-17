# Flask 入門 - 從 Functions 到 Web
# 輔仁大學資訊數學系助教 | LINE: 22303248

from flask import Flask

app = Flask(__name__)

# 回顧：我們學過的 Functions
def say_hello():
    return "Hello!"

def greet(name):
    return f"Hello {name}!"

def add(a, b):
    return a + b

# Flask Routes - 讓 Functions 變成網頁
@app.route('/')
def home():
    return "歡迎來到我的第一個網站！"

@app.route('/hello')
def web_hello():
    return say_hello()

@app.route('/greet/<name>')
def web_greet(name):
    return greet(name)

@app.route('/add/<int:a>/<int:b>')
def web_add(a, b):
    result = add(a, b)
    return f"{a} + {b} = {result}"

if __name__ == '__main__':
    print("🚀 網站啟動：http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)