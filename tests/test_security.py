from app.config import get_settings
from app.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hash_and_verify():
    hashed = hash_password("Correct-Horse-42")
    assert hashed != "Correct-Horse-42"
    assert verify_password("Correct-Horse-42", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_round_trip(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-with-enough-entropy")
    get_settings.cache_clear()
    token = create_access_token("user-uuid", {"role": "viewer"})
    payload = decode_access_token(token)
    assert payload["sub"] == "user-uuid"
    assert payload["role"] == "viewer"
    get_settings.cache_clear()
