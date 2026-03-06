from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os

app = Flask(__name__, 
            template_folder='website/templates',
            static_folder='website/static')
app.secret_key = 'xjtlubasketball2024'

DATA_FILE = os.environ.get('DATA_FILE', 'website/data.json')

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default_data()

def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def default_data():
    return {
        "team_name": "西交利物浦大学篮球队",
        "team_name_en": "XJTLU Basketball Team",
        "slogan": "团结 · 拼搏 · 进取",
        "legends": [
            {"name": "待添加", "position": "后卫 | 10", "years": "2020-2024", "photo": ""}
        ],
        "stats": {
            "games": "150+",
            "win_rate": "65%",
            "championships": "8",
            "players": "200+"
        },
        "schedule": [
            {"date": "2026.03.15", "home": "XJTLU", "away": "对手", "location": "主场"},
            {"date": "2026.03.22", "home": "XJTLU", "away": "对手", "location": "客场"}
        ],
        "player_stats": [
            {"name": "球员 A", "points": "18.5", "rebounds": "7.2", "assists": "5.3", "fg": "52%"}
        ],
        "honors": [
            {"year": "2023", "title": "校级联赛冠军", "desc": "获得2023年西浦杯篮球赛冠军"}
        ],
        "history": [
            {"year": "2015", "title": "球队成立", "desc": "西交利物浦大学篮球队正式成立"}
        ],
        "players": [
            {"name": "球员姓名", "number": "10", "position": "后卫", "height": "180cm", "weight": "70kg", "bio": "球员简介", "photo": ""}
        ],
        "gallery": [
            {"title": "训练照片", "url": "", "desc": ""}
        ],
        "messages": []
    }

@app.route('/')
def index():
    data = load_data()
    lang = request.args.get('lang', 'zh')
    return render_template('index.html', data=data, lang=lang)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'xjtlubasketball':
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            error = '用户名或密码错误'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

@app.route('/admin')
def admin():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    data = load_data()
    return render_template('admin.html', data=data)

@app.route('/api/update', methods=['POST'])
def update_data():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    new_data = request.json
    save_data(new_data)
    return jsonify({'success': True})

@app.route('/api/message', methods=['POST'])
def add_message():
    data = request.json
    name = data.get('name', '匿名')
    content = data.get('content', '')
    if content:
        current_data = load_data()
        current_data['messages'].insert(0, {
            'name': name,
            'content': content,
            'time': data.get('time', '')
        })
        save_data(current_data)
        return jsonify({'success': True})
    return jsonify({'success': False})

if __name__ == '__main__':
    import sys
    if getattr(sys, 'frozen', False):
        DATA_FILE = os.path.join(os.path.dirname(sys.executable), 'data.json')
    if not os.path.exists(DATA_FILE):
        save_data(default_data())
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
