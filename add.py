from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import json
import os
import re
from datetime import datetime, timedelta
import hashlib
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
DB_PATH = 'ammanalam.db'

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        baby_dob TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        trust_level INTEGER DEFAULT 0
    )''')

    # Daily checkins table
    c.execute('''CREATE TABLE IF NOT EXISTS checkins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        checkin_date TEXT NOT NULL,
        mood_text TEXT,
        sentiment_score REAL,
        voice_stress REAL,
        baby_week INTEGER,
        week_multiplier REAL,
        raw_risk REAL,
        risk_level TEXT,
        missed INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')

    # EPDS responses table
    c.execute('''CREATE TABLE IF NOT EXISTS epds_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        taken_at TEXT DEFAULT CURRENT_TIMESTAMP,
        q1 INTEGER, q2 INTEGER, q3 INTEGER, q4 INTEGER, q5 INTEGER,
        q6 INTEGER, q7 INTEGER, q8 INTEGER, q9 INTEGER, q10 INTEGER,
        total_score INTEGER,
        risk_level TEXT
    )''')

    conn.commit()
    conn.close()

init_db()


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS (keyword-based NLP)
# ─────────────────────────────────────────────

SAD_WORDS = {
    'exhausted': 0.8, 'empty': 0.9, 'tired': 0.6, 'crying': 0.8,
    'sad': 0.75, 'hopeless': 0.95, 'worthless': 0.95, 'alone': 0.7,
    'scared': 0.7, 'afraid': 0.7, 'anxious': 0.65, 'overwhelmed': 0.75,
    'horrible': 0.85, 'terrible': 0.8, 'awful': 0.8, 'broken': 0.85,
    'lost': 0.7, 'failing': 0.8, 'useless': 0.9, 'miserable': 0.85,
    'depressed': 0.9, 'numb': 0.8, 'trapped': 0.9, 'desperate': 0.9,
    'helpless': 0.85, 'painful': 0.75, 'hate': 0.7, 'dying': 0.95,
    'disappear': 0.95, 'nothing': 0.6, 'never': 0.5, 'worst': 0.7,
    'dark': 0.65, 'heavy': 0.6, 'suffocating': 0.9, 'drowning': 0.85,
    'dead': 0.95, 'hurt': 0.7, 'pain': 0.7, 'suffer': 0.8,
    'cannot': 0.5, "can't": 0.5, 'no energy': 0.7, 'no sleep': 0.65,
    'baby wont stop': 0.7, 'regret': 0.75, 'mistake': 0.65
}

HAPPY_WORDS = {
    'happy': 0.8, 'good': 0.6, 'great': 0.75, 'wonderful': 0.85,
    'amazing': 0.85, 'love': 0.7, 'joy': 0.8, 'smile': 0.7,
    'better': 0.6, 'okay': 0.4, 'fine': 0.4, 'grateful': 0.8,
    'thankful': 0.75, 'blessed': 0.8, 'excited': 0.75, 'calm': 0.7,
    'peaceful': 0.75, 'relaxed': 0.7, 'enjoying': 0.75, 'beautiful': 0.7,
    'proud': 0.75, 'strong': 0.7, 'hopeful': 0.75, 'positive': 0.7,
    'managing': 0.4, 'coping': 0.5, 'rested': 0.65, 'slept': 0.55
}

def analyze_sentiment(text):
    if not text or not text.strip():
        return 0.0
    text_lower = text.lower()
    sad_score = 0
    happy_score = 0
    sad_count = 0
    happy_count = 0
    for word, weight in SAD_WORDS.items():
        if word in text_lower:
            sad_score += weight
            sad_count += 1
    for word, weight in HAPPY_WORDS.items():
        if word in text_lower:
            happy_score += weight
            happy_count += 1
    if sad_count == 0 and happy_count == 0:
        return 0.1  # slightly positive neutral
    net = happy_score - sad_score
    # Normalize to -1 to +1
    total = sad_score + happy_score if (sad_score + happy_score) > 0 else 1
    normalized = net / (total * 0.8)
    return round(max(-1.0, min(1.0, normalized)), 2)


# ─────────────────────────────────────────────
# WEEK MULTIPLIER
# ─────────────────────────────────────────────

WEEK_MULTIPLIERS = {
    1: 1.2, 2: 1.5, 3: 1.8, 4: 1.6,
    5: 1.4, 6: 1.2, 7: 1.0, 8: 1.0
}

def get_baby_week(baby_dob_str):
    try:
        dob = datetime.strptime(baby_dob_str, '%Y-%m-%d')
        delta = datetime.now() - dob
        week = max(1, min(8, (delta.days // 7) + 1))
        return week
    except:
        return 1

def get_multiplier(week):
    return WEEK_MULTIPLIERS.get(min(week, 8), 1.0)


# ─────────────────────────────────────────────
# RISK CALCULATION
# ─────────────────────────────────────────────

def calculate_risk(sentiment, multiplier, voice_stress):
    raw = (sentiment * multiplier) + voice_stress
    return round(raw, 3)

def classify_risk(raw_score, sentiment):
    # Critical override for very negative sentiment
    if sentiment <= -0.8:
        return 'critical'
    if raw_score >= 1.2 or sentiment <= -0.7:
        return 'critical'
    elif raw_score >= 0.6 or sentiment <= -0.3:
        return 'high'
    elif raw_score >= 0.2 or sentiment < 0:
        return 'moderate'
    else:
        return 'low'

def get_risk_message(level):
    messages = {
        'low': {
            'title': 'You seem to be doing okay 💛',
            'message': "That's wonderful to hear. Remember, it's completely normal to have tough moments. Keep checking in daily — even small changes matter.",
            'action': 'Keep up your daily check-ins. You are doing great, Amma!'
        },
        'moderate': {
            'title': 'We noticed you might need some support 🌸',
            'message': "It sounds like things have been a little hard lately. That's okay — you don't have to carry this alone. Many moms feel this way.",
            'action': 'Try talking to someone you trust today. A short walk or 5 minutes of quiet time can also help.'
        },
        'high': {
            'title': 'You deserve more support right now 💙',
            'message': "We hear you. What you're feeling is real and valid. Postpartum emotions can be overwhelming — but you are not alone and this is not your fault.",
            'action': 'Please speak to a doctor or call iCall: 9152987821. You deserve care too, not just your baby.'
        },
        'critical': {
            'title': 'Please reach out for help right now 🆘',
            'message': "We are very concerned about how you are feeling. What you are going through is serious and you deserve immediate, professional support.",
            'action': 'Call now: iCall 9152987821 | Vandrevala Foundation: 1860-2662-345 | NIMHANS: 080-46110007'
        }
    }
    return messages.get(level, messages['moderate'])


# ─────────────────────────────────────────────
# EPDS SCORING
# ─────────────────────────────────────────────

def score_epds(answers):
    # answers = list of 10 integers (0-3)
    total = sum(answers)
    # Question 10 override (self-harm)
    if answers[9] > 0:
        return total, 'critical'
    if total >= 17:
        return total, 'critical'
    elif total >= 13:
        return total, 'high'
    elif total >= 10:
        return total, 'moderate'
    else:
        return total, 'low'


# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user = get_user(session['user_id'])
    conn = get_db()
    # Get last 7 checkins for chart
    checkins = conn.execute(
        'SELECT * FROM checkins WHERE user_id = ? ORDER BY checkin_date DESC LIMIT 7',
        (session['user_id'],)
    ).fetchall()
    # Check if already checked in today
    today = datetime.now().strftime('%Y-%m-%d')
    today_checkin = conn.execute(
        'SELECT * FROM checkins WHERE user_id = ? AND checkin_date = ?',
        (session['user_id'], today)
    ).fetchone()
    # Get trust level and checkin count
    checkin_count = conn.execute(
        'SELECT COUNT(*) as cnt FROM checkins WHERE user_id = ? AND missed = 0',
        (session['user_id'],)
    ).fetchone()['cnt']
    # EPDS eligible after 5 checkins
    epds_eligible = checkin_count >= 5
    last_epds = conn.execute(
        'SELECT * FROM epds_responses WHERE user_id = ? ORDER BY taken_at DESC LIMIT 1',
        (session['user_id'],)
    ).fetchone()
    conn.close()

    baby_week = get_baby_week(user['baby_dob'])

    return render_template('dashboard.html',
        user=user,
        checkins=checkins,
        today_checkin=today_checkin,
        baby_week=baby_week,
        checkin_count=checkin_count,
        epds_eligible=epds_eligible,
        last_epds=last_epds
    )

@app.route('/checkin')
@login_required
def checkin_page():
    user = get_user(session['user_id'])
    baby_week = get_baby_week(user['baby_dob'])
    multiplier = get_multiplier(baby_week)
    return render_template('checkin.html', user=user, baby_week=baby_week, multiplier=multiplier)

@app.route('/epds')
@login_required
def epds_page():
    conn = get_db()
    checkin_count = conn.execute(
        'SELECT COUNT(*) as cnt FROM checkins WHERE user_id = ? AND missed = 0',
        (session['user_id'],)
    ).fetchone()['cnt']
    conn.close()
    if checkin_count < 5:
        return redirect(url_for('dashboard'))
    return render_template('epds.html')

@app.route('/history')
@login_required
def history_page():
    conn = get_db()
    checkins = conn.execute(
        'SELECT * FROM checkins WHERE user_id = ? ORDER BY checkin_date DESC LIMIT 30',
        (session['user_id'],)
    ).fetchall()
    epds_list = conn.execute(
        'SELECT * FROM epds_responses WHERE user_id = ? ORDER BY taken_at DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return render_template('history.html', checkins=checkins, epds_list=epds_list)


# ─────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    baby_dob = data.get('baby_dob', '').strip()

    if not username or not password or not baby_dob:
        return jsonify({'error': 'All fields required'}), 400

    conn = get_db()
    existing = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': 'Username already exists'}), 400

    conn.execute(
        'INSERT INTO users (username, password_hash, baby_dob) VALUES (?, ?, ?)',
        (username, hash_password(password), baby_dob)
    )
    conn.commit()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()

    session['user_id'] = user['id']
    return jsonify({'success': True, 'redirect': '/dashboard'})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    conn = get_db()
    user = conn.execute(
        'SELECT * FROM users WHERE username = ? AND password_hash = ?',
        (username, hash_password(password))
    ).fetchone()
    conn.close()

    if not user:
        return jsonify({'error': 'Invalid username or password'}), 401

    session['user_id'] = user['id']
    return jsonify({'success': True, 'redirect': '/dashboard'})


@app.route('/api/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/api/analyze_text', methods=['POST'])
@login_required
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    score = analyze_sentiment(text)
    return jsonify({'sentiment': score})


@app.route('/api/checkin', methods=['POST'])
@login_required
def submit_checkin():
    data = request.get_json()
    mood_text = data.get('mood_text', '')
    voice_stress = float(data.get('voice_stress', 0.1))
    user = get_user(session['user_id'])

    sentiment = analyze_sentiment(mood_text)
    baby_week = get_baby_week(user['baby_dob'])
    multiplier = get_multiplier(baby_week)
    raw_risk = calculate_risk(sentiment, multiplier, voice_stress)
    risk_level = classify_risk(raw_risk, sentiment)

    today = datetime.now().strftime('%Y-%m-%d')
    conn = get_db()

    # Check if already checked in today
    existing = conn.execute(
        'SELECT id FROM checkins WHERE user_id = ? AND checkin_date = ?',
        (session['user_id'], today)
    ).fetchone()

    if existing:
        conn.execute('''UPDATE checkins SET mood_text=?, sentiment_score=?, voice_stress=?,
            baby_week=?, week_multiplier=?, raw_risk=?, risk_level=?
            WHERE user_id=? AND checkin_date=?''',
            (mood_text, sentiment, voice_stress, baby_week, multiplier,
             raw_risk, risk_level, session['user_id'], today))
    else:
        conn.execute('''INSERT INTO checkins
            (user_id, checkin_date, mood_text, sentiment_score, voice_stress,
             baby_week, week_multiplier, raw_risk, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (session['user_id'], today, mood_text, sentiment, voice_stress,
             baby_week, multiplier, raw_risk, risk_level))

    conn.commit()
    conn.close()

    risk_info = get_risk_message(risk_level)

    return jsonify({
        'success': True,
        'sentiment': sentiment,
        'baby_week': baby_week,
        'multiplier': multiplier,
        'voice_stress': voice_stress,
        'raw_risk': raw_risk,
        'risk_level': risk_level,
        'risk_info': risk_info,
        'formula': f"({sentiment} × {multiplier}) + {voice_stress} = {raw_risk}"
    })


@app.route('/api/chart_data')
@login_required
def chart_data():
    conn = get_db()
    checkins = conn.execute(
        '''SELECT checkin_date, raw_risk, risk_level, sentiment_score
           FROM checkins WHERE user_id = ? ORDER BY checkin_date ASC LIMIT 14''',
        (session['user_id'],)
    ).fetchall()
    conn.close()

    labels = []
    risks = []
    colors = []
    color_map = {'low': '#7BAE7F', 'moderate': '#E0A030', 'high': '#D4574A', 'critical': '#8B1A1A'}

    for c in checkins:
        labels.append(c['checkin_date'])
        risks.append(round(c['raw_risk'], 2))
        colors.append(color_map.get(c['risk_level'], '#C4B5A5'))

    return jsonify({'labels': labels, 'risks': risks, 'colors': colors})


@app.route('/api/epds', methods=['POST'])
@login_required
def submit_epds():
    data = request.get_json()
    answers = data.get('answers', [])

    if len(answers) != 10:
        return jsonify({'error': 'All 10 questions required'}), 400

    answers = [int(a) for a in answers]
    total, risk_level = score_epds(answers)

    conn = get_db()
    conn.execute('''INSERT INTO epds_responses
        (user_id, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, total_score, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (session['user_id'], *answers, total, risk_level))
    conn.commit()
    conn.close()

    risk_info = get_risk_message(risk_level)
    return jsonify({
        'success': True,
        'total_score': total,
        'risk_level': risk_level,
        'risk_info': risk_info,
        'max_score': 30
    })


@app.route('/api/missed_checkins')
@login_required
def check_missed():
    """Mark missed checkins and detect silent risk signals"""
    conn = get_db()
    # Find last 3 days
    missed_days = []
    for i in range(1, 4):
        day = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        existing = conn.execute(
            'SELECT id FROM checkins WHERE user_id = ? AND checkin_date = ?',
            (session['user_id'], day)
        ).fetchone()
        if not existing:
            missed_days.append(day)
            conn.execute('''INSERT OR IGNORE INTO checkins
                (user_id, checkin_date, missed, risk_level) VALUES (?, ?, 1, 'unknown')''',
                (session['user_id'], day))
    conn.commit()
    conn.close()
    return jsonify({'missed_days': len(missed_days), 'silent_risk': len(missed_days) >= 3})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
