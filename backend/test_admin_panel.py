import os
import sys
import tempfile
import pytest
from datetime import datetime, timedelta, timezone
import jwt

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.models import User, Case, ImageRecord, PredictionRecord, ClinicalReview, UserRole, ActivityLog
from app.auth import hash_password
from app.config import settings
from app.main import app

# Create isolated in-memory or temp SQLite database for tests
temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
TEST_DATABASE_URL = f"sqlite:///{temp_db.name}"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def override_db():
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists(temp_db.name):
        try:
            os.remove(temp_db.name)
        except Exception:
            pass

def create_test_token(user: User):
    # Backdate iat by 10 minutes to avoid any WSL/Windows clock drift issues
    now_utc = datetime.now(timezone.utc)
    expire = now_utc + timedelta(minutes=60)
    to_encode = {
        "sub": user.username,
        "role": user.role.value if hasattr(user.role, "value") else str(user.role),
        "exp": expire,
        "iat": now_utc - timedelta(minutes=10)
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def seed_users(db_session):
    # Clear tables
    for tbl in [ActivityLog, ClinicalReview, PredictionRecord, ImageRecord, Case, User]:
        db_session.query(tbl).delete()
    db_session.commit()

    admin = User(
        username="admin_user",
        email="admin@dermai.org",
        hashed_password=hash_password("AdminPass123!"),
        full_name="Administrator Jane",
        role=UserRole.ADMIN,
        is_verified=True
    )
    doctor = User(
        username="dr_derm",
        email="doctor@hospital.org",
        hashed_password=hash_password("DoctorPass123!"),
        full_name="Dr. Gregory House",
        institution="Princeton Plainsboro",
        role=UserRole.CLINICIAN,
        is_verified=True
    )
    unverified_doctor = User(
        username="dr_intern",
        email="intern@hospital.org",
        hashed_password=hash_password("InternPass123!"),
        full_name="Dr. Allison Cameron",
        institution="Mercy Hospital",
        role=UserRole.CLINICIAN,
        is_verified=False
    )
    patient = User(
        username="john_doe",
        email="john@patient.org",
        hashed_password=hash_password("PatientPass123!"),
        full_name="John Doe",
        role=UserRole.PATIENT,
        is_verified=True
    )

    db_session.add_all([admin, doctor, unverified_doctor, patient])
    db_session.commit()
    db_session.refresh(admin)
    db_session.refresh(doctor)
    db_session.refresh(unverified_doctor)
    db_session.refresh(patient)

    return {
        "admin": admin,
        "doctor": doctor,
        "unverified_doctor": unverified_doctor,
        "patient": patient
    }


def test_admin_stats_authorization(seed_users):
    admin = seed_users["admin"]
    patient = seed_users["patient"]
    doctor = seed_users["doctor"]

    admin_token = create_test_token(admin)
    patient_token = create_test_token(patient)
    doctor_token = create_test_token(doctor)

    # 1. Unauthenticated -> 401
    resp = client.get("/api/v1/admin/stats")
    assert resp.status_code == 401

    # 2. Patient -> 403 Forbidden
    resp = client.get("/api/v1/admin/stats", headers={"Authorization": f"Bearer {patient_token}"})
    assert resp.status_code == 403

    # 3. Doctor -> 403 Forbidden
    resp = client.get("/api/v1/admin/stats", headers={"Authorization": f"Bearer {doctor_token}"})
    assert resp.status_code == 403

    # 4. Admin -> 200 OK
    resp = client.get("/api/v1/admin/stats", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_users"] == 4
    assert data["total_patients"] == 1
    assert data["total_doctors"] == 2
    assert "recent_activity" in data


def test_admin_patients_and_doctors_endpoints(seed_users):
    admin = seed_users["admin"]
    admin_token = create_test_token(admin)

    # Get Patients
    resp = client.get("/api/v1/admin/patients", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    patients = resp.json()
    assert len(patients) == 1
    assert patients[0]["email"] == "john@patient.org"

    # Search Patients
    resp = client.get("/api/v1/admin/patients?q=John", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = client.get("/api/v1/admin/patients?q=NonExistent", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert len(resp.json()) == 0

    # Get Doctors
    resp = client.get("/api/v1/admin/doctors", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    doctors = resp.json()
    assert len(doctors) == 2

    # Search Doctors
    resp = client.get("/api/v1/admin/doctors?q=Cameron", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["full_name"] == "Dr. Allison Cameron"


def test_admin_update_user(seed_users, db_session):
    admin = seed_users["admin"]
    patient = seed_users["patient"]
    admin_token = create_test_token(admin)

    # Update patient name and email
    update_payload = {
        "full_name": "Jonathan Doe",
        "email": "jonathan.doe@updated.org",
        "password": "NewSecretPassword123!"
    }
    resp = client.put(
        f"/api/v1/admin/users/{patient.id}",
        json=update_payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert resp.status_code == 200
    updated_data = resp.json()
    assert updated_data["full_name"] == "Jonathan Doe"
    assert updated_data["email"] == "jonathan.doe@updated.org"

    # Verify activity log
    log = db_session.query(ActivityLog).filter(
        ActivityLog.action == "USER_UPDATED",
        ActivityLog.target_id == patient.id
    ).first()
    assert log is not None
    assert "Jonathan Doe" in log.details


def test_admin_delete_user_safety(seed_users, db_session):
    admin = seed_users["admin"]
    doctor = seed_users["doctor"]
    patient = seed_users["patient"]
    admin_token = create_test_token(admin)

    # 1. Self-deletion blocked
    resp = client.delete(f"/api/v1/admin/users/{admin.id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 400
    assert "Cannot delete your own active administrator account" in resp.json()["detail"]

    # 2. Setup a case assigned to doctor with a review
    case = Case(
        case_code="CASE-ADMIN-001",
        user_id=patient.id,
        body_location="back",
        assigned_doctor_id=doctor.id,
        review_status="under_review"
    )
    db_session.add(case)
    db_session.commit()
    db_session.refresh(case)

    review = ClinicalReview(
        case_id=case.id,
        doctor_id=doctor.id,
        comment="Clinical observation notes",
        recommendation="Biopsy recommended"
    )
    db_session.add(review)
    db_session.commit()

    # 3. Delete doctor -> Safe unassignment
    doctor_id = doctor.id
    resp = client.delete(f"/api/v1/admin/users/{doctor_id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200

    # Verify case was safely returned to triage
    db_session.expire_all()
    updated_case = db_session.query(Case).filter(Case.id == case.id).first()
    assert updated_case is not None
    assert updated_case.assigned_doctor_id is None
    assert updated_case.review_status == "pending_review"

    # Verify activity log
    del_log = db_session.query(ActivityLog).filter(
        ActivityLog.action == "USER_DELETED",
        ActivityLog.target_id == str(doctor_id)
    ).first()
    assert del_log is not None


def test_admin_analyses_logs_settings(seed_users):
    admin = seed_users["admin"]
    admin_token = create_test_token(admin)

    # Analyses endpoint
    resp = client.get("/api/v1/admin/analyses", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # Activity Logs endpoint
    resp = client.get("/api/v1/admin/activity-logs", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # Settings endpoint
    resp = client.get("/api/v1/admin/settings", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    settings_data = resp.json()
    assert "environment" in settings_data
    assert settings_data["admin_user"]["username"] == admin.username
