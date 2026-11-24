import io
import os
import cv2
import numpy as np
import tempfile
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.models import User, UserRole
from app.auth import hash_password
from app.main import app

client = TestClient(app)


def generate_test_image(blur=False):
    img = np.full((300, 300, 3), (180, 200, 240), dtype=np.uint8)
    cv2.circle(img, (150, 150), 40, (40, 60, 100), -1)
    if blur:
        img = cv2.GaussianBlur(img, (25, 25), 0)
    success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer.tobytes())


def test_combined_workflow_and_rbac():
    print("\n" + "=" * 70)
    print("VERIFYING DERMAI COMBINED WORKFLOW, RETRY BEHAVIOR & RBAC")
    print("=" * 70)

    # Isolated test database fixture
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix="_comb.db")
    os.close(temp_db_fd)
    test_db_url = f"sqlite:///{temp_db_path}"
    test_engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    from datetime import timedelta, timezone, datetime
    import jwt
    from app.config import settings
    import app.auth as auth_mod
    import app.main as main_mod
    orig_auth_token_func = auth_mod.create_access_token
    orig_main_token_func = main_mod.create_access_token

    def safe_create_access_token(data: dict, expires_delta=None):
        to_encode = data.copy()
        now_utc = datetime.now(timezone.utc)
        if expires_delta:
            expire = now_utc + expires_delta
        else:
            expire = now_utc + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "iat": now_utc - timedelta(minutes=10)})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    auth_mod.create_access_token = safe_create_access_token
    main_mod.create_access_token = safe_create_access_token

    ts = os.urandom(4).hex()
    try:
        # 1. Provision Test Administrator directly in DB (public admin registration is forbidden)
        admin_username = f"admin_comb_{ts}"
        admin_password = "AdminSecurePassword123!"
        db = TestingSessionLocal()
        try:
            test_admin = User(
                username=admin_username,
                email=f"admin_{ts}@dermaitest.org",
                full_name="Chief Medical Administrator",
                role=UserRole.ADMIN,
                institution="DermAI Test Center",
                hashed_password=hash_password(admin_password),
                is_active=True,
                is_verified=True
            )
            db.add(test_admin)
            db.commit()
        finally:
            db.close()

        res_adm_login = client.post("/api/v1/auth/login", json={
            "username": admin_username,
            "password": admin_password
        })
        assert res_adm_login.status_code == 200
        token_admin = res_adm_login.json()["access_token"]
        print("[PASS] 1a. Administrator provisioned and authenticated via login.")

        # 2. Register Clinician (starts unverified)
        res_clinician = client.post("/api/v1/auth/register", json={
            "username": f"clinician_{ts}",
            "email": f"clinician_{ts}@test.org",
            "full_name": "Dr. Sarah Clinician MD",
            "password": "Password123!",
            "role": "clinician"
        })
        assert res_clinician.status_code == 200
        clinician_data = res_clinician.json()
        assert clinician_data.get("is_verified") is False
        token_unverified_clinician = clinician_data["access_token"]
        clinician_id = clinician_data["user_id"]
        print("[PASS] 1b. New clinician registered; account awaits administrative verification.")

        # 2b. Unverified clinician blocked from clinical triage (HTTP 403)
        res_unverified_block = client.get(
            "/api/v1/cases/?assignment_filter=unassigned",
            headers={"Authorization": f"Bearer {token_unverified_clinician}"}
        )
        assert res_unverified_block.status_code == 403
        print("[PASS] 1c. Unverified clinician blocked from triage queue with HTTP 403.")

        # 2c. Admin verifies clinician
        res_verify = client.patch(
            f"/api/v1/admin/clinicians/{clinician_id}/verify",
            headers={"Authorization": f"Bearer {token_admin}"}
        )
        assert res_verify.status_code == 200
        assert res_verify.json()["is_verified"] is True
        print("[PASS] 1d. Admin approved and verified clinician.")

        # Verified clinician authenticates
        res_c_login = client.post("/api/v1/auth/login", json={
            "username": f"clinician_{ts}",
            "password": "Password123!"
        })
        assert res_c_login.status_code == 200
        token_clinician = res_c_login.json()["access_token"]
        print("[PASS] 1e. Verified clinician authenticated successfully.")

        # 3. Security Boundary: Public registration rejects unsupported roles (Researcher, ML Engineer, Admin)
        res_researcher = client.post("/api/v1/auth/register", json={
            "username": f"researcher_{ts}",
            "email": f"researcher_{ts}@test.org",
            "full_name": "Elena Vance PhD",
            "password": "Password123!",
            "role": "researcher"
        })
        assert res_researcher.status_code == 400
        assert "not permitted" in res_researcher.text.lower()
        print("[PASS] 2. Public registration correctly rejects 'researcher' with HTTP 400 Bad Request.")

        res_rej_admin = client.post("/api/v1/auth/register", json={
            "username": f"rogue_admin_{ts}",
            "email": f"rogue_{ts}@test.org",
            "full_name": "Rogue Admin",
            "password": "Password123!",
            "role": "admin"
        })
        assert res_rej_admin.status_code == 400
        print("[PASS] 2b. Public registration correctly rejects 'admin' with HTTP 400 Bad Request.")

        # 4. Register Patient
        res_patient = client.post("/api/v1/auth/register", json={
            "username": f"patient_{ts}",
            "email": f"patient_{ts}@test.org",
            "full_name": "Alex Patient",
            "password": "Password123!",
            "role": "patient"
        })
        assert res_patient.status_code == 200
        token_patient = res_patient.json()["access_token"]
        print("[PASS] 3. Patient registered and authenticated.")

        # 5. Combined Case Creation & Image Upload (Happy Path)
        case_payload = {
            "patient_age": 42,
            "patient_gender": "male",
            "body_location": "back",
            "known_allergies": "None",
            "medical_history": "Normal",
            "clinical_notes": "Uniform border melanocytic lesion on upper back",
            "image_source": "mobile_upload"
        }
        res_case = client.post("/api/v1/cases/", headers={"Authorization": f"Bearer {token_clinician}"}, json=case_payload)
        assert res_case.status_code == 200, f"Case creation failed: {res_case.text}"
        case_data = res_case.json()
        case_id = case_data["id"]
        case_code = case_data["case_code"]
        print(f"[PASS] 4a. Case record created: {case_code} (ID: {case_id})")

        # Upload valid image
        img_bytes = generate_test_image()
        res_upload = client.post(
            f"/api/v1/cases/{case_id}/upload-image",
            headers={"Authorization": f"Bearer {token_clinician}"},
            files={"file": ("lesion_back.jpg", img_bytes, "image/jpeg")}
        )
        assert res_upload.status_code == 200, f"Upload failed: {res_upload.text}"
        processed_case = res_upload.json()
        assert processed_case["status"] == "preprocessed"
        assert len(processed_case["images"]) == 1
        img_rec = processed_case["images"][0]
        assert img_rec["blur_score"] is not None
        assert img_rec["brightness_score"] is not None
        assert img_rec["contrast_score"] is not None
        assert img_rec["preprocessed_url"] is not None
        print(f"[PASS] 4b. Image uploaded & preprocessed: blur={img_rec['blur_score']:.1f}, brightness={img_rec['brightness_score']:.1f}, status={processed_case['status']}")

        # 6. Combined Case Creation & Failure + Retry Behavior
        res_case2 = client.post("/api/v1/cases/", headers={"Authorization": f"Bearer {token_clinician}"}, json=case_payload)
        assert res_case2.status_code == 200
        case2_data = res_case2.json()
        case2_id = case2_data["id"]
        print(f"[PASS] 5a. Created case {case2_data['case_code']} for failure simulation.")

        # Simulate upload failure (corrupted file)
        res_fail_upload = client.post(
            f"/api/v1/cases/{case2_id}/upload-image",
            headers={"Authorization": f"Bearer {token_clinician}"},
            files={"file": ("corrupted.jpg", io.BytesIO(b"not a real jpeg binary"), "image/jpeg")}
        )
        assert res_fail_upload.status_code == 400
        print(f"[PASS] 5b. Corrupted upload correctly rejected with HTTP 400: '{res_fail_upload.json()['detail']}'")

        # Case still exists in 'created' state
        res_check = client.get(f"/api/v1/cases/{case2_id}", headers={"Authorization": f"Bearer {token_clinician}"})
        assert res_check.status_code == 200
        assert res_check.json()["status"] == "created"
        assert len(res_check.json()["images"]) == 0
        print(f"[PASS] 5c. Verified case {case2_data['case_code']} remains intact awaiting retry.")

        # Retry upload with valid image (WITHOUT re-creating case!)
        res_retry = client.post(
            f"/api/v1/cases/{case2_id}/upload-image",
            headers={"Authorization": f"Bearer {token_clinician}"},
            files={"file": ("retry_lesion.jpg", generate_test_image(), "image/jpeg")}
        )
        assert res_retry.status_code == 200
        assert res_retry.json()["status"] == "preprocessed"
        print(f"[PASS] 5d. Retry upload succeeded for case {case2_data['case_code']}. Status: preprocessed.")

        # 7. Verify Dataset & Research RBAC Protection
        # 7a. Unauthenticated -> 401
        res_unauth = client.get("/api/v1/dataset/eda-summary")
        assert res_unauth.status_code == 401
        print("[PASS] 6a. Unauthenticated access to /api/v1/dataset/eda-summary rejected with HTTP 401.")

        # 7b. Patient -> 403
        res_patient_ds = client.get("/api/v1/dataset/eda-summary", headers={"Authorization": f"Bearer {token_patient}"})
        assert res_patient_ds.status_code == 403
        print("[PASS] 6b. Patient access to /api/v1/dataset/eda-summary rejected with HTTP 403 Forbidden.")

        # 7c. Clinician -> 403
        res_clinician_ds = client.get("/api/v1/dataset/eda-summary", headers={"Authorization": f"Bearer {token_clinician}"})
        assert res_clinician_ds.status_code == 403
        print("[PASS] 6c. Clinician access to /api/v1/dataset/eda-summary rejected with HTTP 403 Forbidden.")

        # 7d. Admin (authorized system governance role) -> 200 OK
        res_admin_ds = client.get("/api/v1/dataset/eda-summary", headers={"Authorization": f"Bearer {token_admin}"})
        assert res_admin_ds.status_code == 200
        eda_json = res_admin_ds.json()
        assert eda_json["total_records"] == 10015
        assert len(eda_json["class_summary"]) == 7
        print(f"[PASS] 6d. Admin authorized dataset inspection: HTTP 200 OK, {eda_json['total_records']} verified records, 7 classes.")

        print("\n" + "=" * 70)
        print("ALL COMBINED WORKFLOW & RBAC VERIFICATION SCENARIOS PASSED!")
        print("=" * 70 + "\n")

    finally:
        auth_mod.create_access_token = orig_auth_token_func
        main_mod.create_access_token = orig_main_token_func
        app.dependency_overrides.clear()
        test_engine.dispose()
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except Exception:
                pass

        # Clean any uploaded test images
        upload_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        if os.path.isdir(upload_base):
            for root, dirs, files in os.walk(upload_base):
                for f in files:
                    if f != ".gitkeep":
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception:
                            pass


if __name__ == "__main__":
    test_combined_workflow_and_rbac()
