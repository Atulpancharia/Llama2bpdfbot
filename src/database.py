from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    "postgresql://postgres:root@localhost:5432/llm_chatbot",
    pool_size=20,
    max_overflow=20,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        db.commit()
        yield db
    except:
        db.rollback()
        raise
    finally:
        db.close()
