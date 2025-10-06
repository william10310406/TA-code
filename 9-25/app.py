from flask import Flask, render_template, request, redirect

# 這是一個最小可行的 Flask 網站
# Flask 是一個 Python 的網站框架（framework），可以讓我們把「網址」對應到「函式」，
# 然後在函式裡回傳一段 HTML 給瀏覽器顯示。
app = Flask(__name__)


@app.route("/")
def index():
    # 當使用者造訪根目錄「/」時，回傳 templates/index.html 這個 HTML 檔案
    # render_template 會到 templates/ 資料夾找同名檔案，並把它回傳給瀏覽器
    return render_template("index.html")
－

@app.route("/hello/")
@app.route("/hello/<name>")
def hello(name=None):
    # 這個路由示範「動態網址參數」。
    # 不過現在使用純 HTML，所以不管網址是什麼都回傳同樣的頁面
    # 你可以試試 /hello/ 或 /hello/小明，都會看到一樣的內容
    return render_template("hello.html")


@app.route("/form", methods=["GET", "POST"])
def form():
    # 這個路由同時接受 GET 與 POST
    # - GET：第一次打開表單頁面
    # - POST：按下送出後，瀏覽器把表單資料送過來
    if request.method == "POST":
        # 表單送出後，我們直接重導向到結果頁
        # 因為現在是純 HTML，所以不處理表單資料
        return redirect("/result")
    # 若是 GET 請求，就回傳表單頁面
    return render_template("form.html")


@app.route("/result")
def result():
    # 結果頁現在是純 HTML，所以不需要處理任何資料
    # 直接回傳 result.html 檔案
    return render_template("result.html")


@app.route("/example")
def example():
    # iPhone 範例頁面 - 展示進階的 CSS 動畫效果
    return render_template("example.html")


@app.route("/introduction")
def introduction():
    # 從 iPhone 頁面點擊按鈕後跳轉到這裡
    # 可以建立一個介紹頁面，或重導向到首頁
    return redirect("/")


if __name__ == "__main__":
    # debug=True 代表開發模式，修改程式後會自動重啟伺服器，並顯示錯誤訊息
    app.run(debug=True)


