from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os
import base64
import uuid
import hashlib
import cloudinary
import cloudinary.uploader

app = Flask(__name__, 
            template_folder='website/templates',
            static_folder='website/static')
app.secret_key = 'xjtlubasketball2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', ''),
    api_key=os.environ.get('CLOUDINARY_API_KEY', ''),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', ''),
    secure=True
)

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
        "team_photo": "",
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
        "messages": [],
        "rosters": [
            {"year": "2025", "players": [{"name": "球员姓名", "number": "10", "position": "后卫"}]}
        ],
        "coaches": [
            {"name": "教练姓名", "role": "主教练", "bio": "教练简介", "photo": ""}
        ],
        "videos": [
            {"title": "比赛集锦", "url": "", "desc": ""}
        ],
        "discussions": [],
        "users": []
    }

@app.route('/')
def index():
    data = load_data()
    lang = request.args.get('lang', 'zh')
    return render_template('index.html', data=data, lang=lang)

@app.route('/user')
def user():
    data = load_data()
    lang = request.args.get('lang', 'zh')
    return render_template('user.html', data=data, lang=lang)

@app.route('/player/<name>')
def player(name):
    data = load_data()
    lang = request.args.get('lang', 'zh')
    player_data = None
    for p in data.get('players', []):
        if p.get('name') == name:
            player_data = p
            break
    if not player_data:
        return redirect(url_for('index'))
    return render_template('player.html', player=player_data, data=data, lang=lang)

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

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'success': False, 'message': '请填写用户名和密码'}), 400
    
    if len(password) < 6:
        return jsonify({'success': False, 'message': '密码至少6位'}), 400
    
    current_data = load_data()
    users = current_data.get('users', [])
    
    for user in users:
        if user.get('username') == username:
            return jsonify({'success': False, 'message': '用户名已存在'}), 400
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    users.append({
        'username': username,
        'password': hashed_password
    })
    current_data['users'] = users
    save_data(current_data)
    return jsonify({'success': True})

@app.route('/api/login', methods=['POST'])
def user_login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'success': False, 'message': '请填写用户名和密码'}), 400
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    current_data = load_data()
    users = current_data.get('users', [])
    
    for user in users:
        if user.get('username') == username and user.get('password') == hashed_password:
            session['user'] = username
            return jsonify({'success': True, 'username': username})
    
    return jsonify({'success': False, 'message': '用户名或密码错误'}), 400

@app.route('/api/logout')
def user_logout():
    session.pop('user', None)
    return jsonify({'success': True})

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image'}), 400
    
    if file:
        ext = file.filename.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            return jsonify({'success': False, 'message': 'Invalid type'}), 400
        
        cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME', '')
        if cloud_name:
            try:
                result = cloudinary.uploader.upload(file, folder='xjtlubasketball')
                return jsonify({'success': True, 'url': result['secure_url']})
            except Exception as e:
                print(f"Cloudinary error: {e}")
        
        img_data = base64.b64encode(file.read()).decode('utf-8')
        data_url = f"data:image/{ext};base64,{img_data}"
        return jsonify({'success': True, 'url': data_url})
    return jsonify({'success': False})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No video'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No video'}), 400
    
    if file:
        ext = file.filename.split('.')[-1].lower()
        if ext not in ['mp4', 'webm', 'ogg']:
            return jsonify({'success': False, 'message': 'Invalid type'}), 400
        
        cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME', '')
        if cloud_name:
            try:
                result = cloudinary.uploader.upload(file, folder='xjtlubasketball/videos', resource_type='video')
                return jsonify({'success': True, 'url': result['secure_url']})
            except Exception as e:
                print(f"Cloudinary error: {e}")
        
        video_data = base64.b64encode(file.read()).decode('utf-8')
        data_url = f"data:video/{ext};base64,{video_data}"
        return jsonify({'success': True, 'url': data_url})
    return jsonify({'success': False})

@app.route('/api/discussion', methods=['POST'])
def add_discussion():
    data = request.json
    name = data.get('name', '匿名')
    content = data.get('content', '')
    if content:
        current_data = load_data()
        current_data['discussions'].insert(0, {
            'name': name,
            'content': content,
            'time': data.get('time', ''),
            'replies': []
        })
        save_data(current_data)
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/api/discussion/reply', methods=['POST'])
def reply_discussion():
    data = request.json
    index = data.get('index', -1)
    name = data.get('name', '匿名')
    content = data.get('content', '')
    if content and index >= 0:
        current_data = load_data()
        if index < len(current_data['discussions']):
            current_data['discussions'][index].setdefault('replies', []).insert(0, {
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
