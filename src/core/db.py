import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from src.core.config import config
import os

Base = declarative_base()

engine = sa.create_engine(config.get('database.url', 'sqlite:///aisis.db'))
Session = scoped_session(sessionmaker(bind=engine))

class Preference(Base):
    __tablename__ = 'preferences'
    key = sa.Column(sa.String, primary_key=True)
    value = sa.Column(sa.String)

class History(Base):
    __tablename__ = 'history'
    id = sa.Column(sa.Integer, primary_key=True)
    timestamp = sa.Column(sa.DateTime, default=sa.func.now())
    task_type = sa.Column(sa.String)
    input_data = sa.Column(sa.Text)
    output_data = sa.Column(sa.Text)
    status = sa.Column(sa.String)

Base.metadata.create_all(engine)

def get_preference(key, default=None):
    session = Session()
    pref = session.query(Preference).get(key)
    Session.remove()
    return pref.value if pref else default

def set_preference(key, value):
    session = Session()
    pref = session.query(Preference).get(key)
    if not pref:
        pref = Preference(key=key, value=value)
        session.add(pref)
    else:
        pref.value = value
    session.commit()
    Session.remove()

def add_history(task_type, input_data, output_data, status):
    session = Session()
    history = History(task_type=task_type, input_data=input_data, output_data=output_data, status=status)
    session.add(history)
    session.commit()
    Session.remove()

def backup_db():
    import time
    import zipfile
    backup_file = f'backups/backup_{int(time.time())}.zip'
    with zipfile.ZipFile(backup_file, 'w') as zipf:
        zipf.write('aisis.db')

def restore_db(backup_file):
    import zipfile
    with zipfile.ZipFile(backup_file, 'r') as zipf:
        zipf.extract('aisis.db')
