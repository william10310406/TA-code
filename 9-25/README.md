# 9-25 Flask + HTML 教學範例

此資料夾示範如何用 Flask 串接 HTML（Jinja2 模板）與靜態資源。

## 如何執行
1. 建議建立虛擬環境並啟用：python3 -m venv .venv / source .venv/bin/activate
2. 安裝套件：pip install flask
3. 啟動伺服器：python app.py（或設定環境變數 FLASK_APP=app.py 後 flask run）
4. 瀏覽器開啟：http://127.0.0.1:5000/

## 路由與模板
- `/` → `templates/index.html`
- `/hello/` 與 `/hello/<name>` → `templates/hello.html`
- `/form`（GET/POST）→ `templates/form.html`
- `/result`（顯示表單送出的資料）→ `templates/result.html`

## 重點觀念
- 使用 `render_template(...)` 從 `templates/` 讀取 HTML
- 模板繼承：子頁 `{% extends 'base.html' %}`，共用導覽與版型
- 靜態資源：`url_for('static', filename='style.css')`
- 表單送出（POST）後使用 `redirect(url_for(...))` 導向結果頁並帶參數

## 檔案結構
- app.py
- templates/
  - base.html, index.html, hello.html, form.html, result.html
- static/
  - style.css
