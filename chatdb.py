from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class ChatThread(Base):
    __tablename__ = 'chat_threads'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship('ChatMessage', back_populates='thread', cascade='all, delete-orphan')

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey('chat_threads.id'))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    thread = relationship('ChatThread', back_populates='messages')

# SQLite engine and session
engine = create_engine('sqlite:///chat.sqlite3', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Create tables
Base.metadata.create_all(engine)
