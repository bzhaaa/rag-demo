from app.repositories import list_accessible_documents


def test_department_and_explicit_user_acl(db_session):
    db, models = db_session
    department_a = models.Department(name="A")
    department_b = models.Department(name="B")
    owner = models.User(
        username="owner",
        email="owner@example.com",
        password_hash="x",
        role=models.Role.editor.value,
    )
    same_department = models.User(
        username="same",
        email="same@example.com",
        password_hash="x",
        role=models.Role.viewer.value,
    )
    explicit_user = models.User(
        username="explicit",
        email="explicit@example.com",
        password_hash="x",
        role=models.Role.viewer.value,
    )
    outsider = models.User(
        username="outsider",
        email="outsider@example.com",
        password_hash="x",
        role=models.Role.viewer.value,
    )
    db.add_all(
        [department_a, department_b, owner, same_department, explicit_user, outsider]
    )
    db.flush()
    db.add_all(
        [
            models.DepartmentMembership(
                user_id=owner.id, department_id=department_a.id
            ),
            models.DepartmentMembership(
                user_id=same_department.id, department_id=department_a.id
            ),
            models.DepartmentMembership(
                user_id=explicit_user.id, department_id=department_b.id
            ),
            models.DepartmentMembership(
                user_id=outsider.id, department_id=department_b.id
            ),
        ]
    )
    department_document = models.Document(
        title="Department",
        owner_id=owner.id,
        department_id=department_a.id,
        visibility=models.Visibility.department.value,
    )
    restricted_document = models.Document(
        title="Restricted",
        owner_id=owner.id,
        department_id=department_a.id,
        visibility=models.Visibility.restricted.value,
    )
    db.add_all([department_document, restricted_document])
    db.flush()
    db.add(
        models.DocumentACL(
            document_id=restricted_document.id, user_id=explicit_user.id
        )
    )
    db.commit()
    for user in (same_department, explicit_user, outsider):
        db.refresh(user)

    assert {item.title for item in list_accessible_documents(db, same_department)} == {
        "Department"
    }
    assert {item.title for item in list_accessible_documents(db, explicit_user)} == {
        "Restricted"
    }
    assert list(list_accessible_documents(db, outsider)) == []
