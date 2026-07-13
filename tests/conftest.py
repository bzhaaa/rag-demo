import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TEST_DATABASE = os.path.join(tempfile.gettempdir(), "enterprise_rag_tests.db")
os.environ["DATABASE_URL"] = (
    f"sqlite+pysqlite:///{TEST_DATABASE.replace(chr(92), '/')}"
)

@pytest.fixture()
def db_session():
    from app import models
    from app.db import Base, SessionLocal, engine

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        yield session, models
    Base.metadata.drop_all(engine)
