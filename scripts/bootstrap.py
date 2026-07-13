import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db import SessionLocal
from app.models import Department, DepartmentMembership, Role, User
from app.security import hash_password
from app.storage import ObjectStorage


def main() -> None:
    settings = get_settings()
    with SessionLocal() as db:
        department = db.scalar(
            select(Department).where(
                Department.name == settings.bootstrap_department_name
            )
        )
        if department is None:
            department = Department(name=settings.bootstrap_department_name)
            db.add(department)
            db.flush()
        user = db.scalar(
            select(User).where(
                User.username == settings.bootstrap_admin_username
            )
        )
        if user is None:
            user = User(
                username=settings.bootstrap_admin_username,
                email=settings.bootstrap_admin_email,
                password_hash=hash_password(
                    settings.bootstrap_admin_password
                ),
                role=Role.admin.value,
                is_active=True,
            )
            db.add(user)
            db.flush()
        membership = db.scalar(
            select(DepartmentMembership).where(
                DepartmentMembership.user_id == user.id,
                DepartmentMembership.department_id == department.id,
            )
        )
        if membership is None:
            db.add(
                DepartmentMembership(
                    user_id=user.id, department_id=department.id
                )
            )
        db.commit()
    ObjectStorage().ensure_bucket()
    print(f"Bootstrap complete. Admin username: {settings.bootstrap_admin_username}")


if __name__ == "__main__":
    main()
