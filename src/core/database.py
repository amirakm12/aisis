import sqlite3
from pathlib import Path

def init_db():
    db_dir = Path('storage')
    db_dir.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_dir / 'users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  preferences TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS bugs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  description TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  status TEXT DEFAULT 'open',
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

def report_bug(user_id: int, description: str):
    conn = sqlite3.connect(Path('storage') / 'users.db')
    c = conn.cursor()
    c.execute("INSERT INTO bugs (user_id, description) VALUES (?, ?)", (user_id, description))
    conn.commit()
    conn.close()
