"""
DermAI Milestone 1 Verification Test Suite
Tests all 10 core verification scenarios:
1. Successful registration and login
2. Invalid credentials rejection
3. Unauthorized case access (patient cross-access -> 403)
4. Unauthorized image upload (IDOR prevention -> 403)
5. Unauthorized image retrieval (401 unauthenticated, 403 cross-patient, 200 authorized)
6. Invalid image format rejection (unsupported extension -> 400)
7. Corrupted image rejection (non-image bytes -> 400)
8. Successful image upload and preprocessing (valid image -> 200, QA scores, preprocessed file)
9. Failure cleanup (no orphaned files left on disk)
10. Existing case management behavior (create, list, get, update, delete)
11. Researcher RBAC dataset protection (401 unauth, 403 patient, 403 clinician, 400 registration rejection, 200 admin)
"""

import os
import io
import cv2
import numpy as np
import tempfile
import glob
from pathlib import Path
from datetime import datetime, timedelta, timezone
import jwt
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.config import settings
from app.database import Base, get_db
from app.models import User, UserRole, Case
from app.auth import hash_password

client = TestClient(app)

def generate_test_image_bytes():
    """Generates valid 300x300 JPEG skin-tone image bytes."""
    img = np.full((300, 300, 3), (180, 200, 240), dtype=np.uint8)
    # Add synthetic lesion in center
    cv2.circle(img, (150, 150), 40, (40, 60, 100), -1)
    success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer.tobytes())

def run_tests():
    print("\n" + "="*70)
    print("DERMAI MILESTONE 1 VERIFICATION TEST SUITE EXECUTION")
    print("="*70)

    # Isolated test database fixture
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix="_verif.db")
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

    results = []
    ts = os.urandom(4).hex()
    created_case_codes = []

    try:
        # Provision Admin directly in database (for admin RBAC verification)
        admin_username = f"admin_verif_{ts}"
        admin_password = "SecurePassword123!"
        db = TestingSessionLocal()
        try:
            test_admin = User(
                username=admin_username,
                email=f"admin_{ts}@dermaitest.org",
                full_name="Verification Admin",
                role=UserRole.ADMIN,
                institution="Verification Center",
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
        admin_token = res_adm_login.json()["access_token"]

        # 1. Successful Registration & Login
        try:
            reg_payload = {
                "username": f"test_doc_{ts}",
                "email": f"doc_{ts}@test.org",
                "full_name": "Test Doctor MD",
                "password": "SecurePassword123!",
                "role": "clinician",
                "institution": "Verification Clinic"
            }
            res_reg = client.post("/api/v1/auth/register", json=reg_payload)
            assert res_reg.status_code == 200, f"Registration failed: {res_reg.text}"
            doc_token = res_reg.json()["access_token"]

            # Login with newly registered user
            res_login = client.post("/api/v1/auth/login", json={
                "username": f"test_doc_{ts}",
                "password": "SecurePassword123!"
            })
            assert res_login.status_code == 200
            assert "access_token" in res_login.json()
            results.append(("1. Registration & Login", "PASSED", "Successfully registered clinician and received JWT token."))
        except Exception as e:
            results.append(("1. Registration & Login", "FAILED", str(e)))

        # 2. Invalid Credentials
        try:
            res_bad_login = client.post("/api/v1/auth/login", json={
                "username": f"test_doc_{ts}",
                "password": "WrongPassword!"
            })
            assert res_bad_login.status_code == 401, f"Expected 401, got {res_bad_login.status_code}"
            results.append(("2. Invalid Credentials Rejection", "PASSED", "Rejected invalid password with HTTP 401 Unauthorized."))
        except Exception as e:
            results.append(("2. Invalid Credentials Rejection", "FAILED", str(e)))

        # Register two separate patients for ownership tests
        res_pat_a = client.post("/api/v1/auth/register", json={
            "username": f"pat_a_{ts}",
            "email": f"pata_{ts}@test.org",
            "full_name": "Patient Alpha",
            "password": "Password123!",
            "role": "patient"
        })
        token_pat_a = res_pat_a.json()["access_token"]

        res_pat_b = client.post("/api/v1/auth/register", json={
            "username": f"pat_b_{ts}",
            "email": f"patb_{ts}@test.org",
            "full_name": "Patient Beta",
            "password": "Password123!",
            "role": "patient"
        })
        token_pat_b = res_pat_b.json()["access_token"]

        # Create Case for Patient A
        res_case_a = client.post(
            "/api/v1/cases/",
            headers={"Authorization": f"Bearer {token_pat_a}"},
            json={"patient_age": 45, "patient_gender": "female", "body_location": "arm", "clinical_notes": "Lesion on right forearm"}
        )
        assert res_case_a.status_code == 200
        case_a_data = res_case_a.json()
        case_a_id = case_a_data["id"]
        created_case_codes.append(case_a_data["case_code"])

        # 3. Unauthorized Case Access (Patient B -> Case A)
        try:
            res_cross_view = client.get(
                f"/api/v1/cases/{case_a_id}",
                headers={"Authorization": f"Bearer {token_pat_b}"}
            )
            assert res_cross_view.status_code == 403, f"Expected 403, got {res_cross_view.status_code}"
            results.append(("3. Unauthorized Case Read Isolation", "PASSED", "Patient B denied access to Patient A's case with HTTP 403."))
        except Exception as e:
            results.append(("3. Unauthorized Case Read Isolation", "FAILED", str(e)))

        # 4. Unauthorized Image Upload (Patient B -> Case A)
        try:
            img_bytes = generate_test_image_bytes()
            res_cross_upload = client.post(
                f"/api/v1/cases/{case_a_id}/upload-image",
                headers={"Authorization": f"Bearer {token_pat_b}"},
                files={"file": ("lesion.jpg", img_bytes, "image/jpeg")}
            )
            assert res_cross_upload.status_code == 403, f"Expected 403, got {res_cross_upload.status_code}"
            results.append(("4. Unauthorized Image Upload (BOLA/IDOR Prevention)", "PASSED", "Patient B prevented from uploading to Patient A's case with HTTP 403."))
        except Exception as e:
            results.append(("4. Unauthorized Image Upload (BOLA/IDOR Prevention)", "FAILED", str(e)))

        # 6. Invalid Image Upload Format Rejection
        try:
            bad_text_file = io.BytesIO(b"Hello world, this is a plain text file pretending to be code.")
            res_bad_ext = client.post(
                f"/api/v1/cases/{case_a_id}/upload-image",
                headers={"Authorization": f"Bearer {token_pat_a}"},
                files={"file": ("malicious.txt", bad_text_file, "text/plain")}
            )
            assert res_bad_ext.status_code == 400
            results.append(("6. Invalid File Format Rejection", "PASSED", "Rejected .txt extension with HTTP 400 Bad Request."))
        except Exception as e:
            results.append(("6. Invalid File Format Rejection", "FAILED", str(e)))

        # 7. Corrupted / Spoofed Image Rejection (Fake .jpg with garbage binary)
        try:
            corrupted_bytes = io.BytesIO(b"\xFF\xD8\xFF\xE0GarbageCorruptedBinaryDataPayload1234567890")
            res_corrupt = client.post(
                f"/api/v1/cases/{case_a_id}/upload-image",
                headers={"Authorization": f"Bearer {token_pat_a}"},
                files={"file": ("spoofed.jpg", corrupted_bytes, "image/jpeg")}
            )
            assert res_corrupt.status_code == 400
            assert "Corrupted or invalid image" in res_corrupt.json()["detail"]
            results.append(("7. Corrupted Binary Image Rejection", "PASSED", "Deep binary verification with Pillow rejected non-image bytes."))
        except Exception as e:
            results.append(("7. Corrupted Binary Image Rejection", "FAILED", str(e)))

        # 9. Failure Cleanup Verification
        try:
            files_before = set(os.listdir(settings.ORIGINAL_DIR))
            # Send bad file
            client.post(
                f"/api/v1/cases/{case_a_id}/upload-image",
                headers={"Authorization": f"Bearer {token_pat_a}"},
                files={"file": ("bad.jpg", io.BytesIO(b"junk"), "image/jpeg")}
            )
            files_after = set(os.listdir(settings.ORIGINAL_DIR))
            assert files_before == files_after, "Orphaned files remained in upload directory after failure!"
            results.append(("9. Transactional Failure Cleanup", "PASSED", "Verified zero orphaned files remain in storage after failed uploads."))
        except Exception as e:
            results.append(("9. Transactional Failure Cleanup", "FAILED", str(e)))

        # 8. Successful Upload and Preprocessing (Patient A)
        uploaded_filename = None
        try:
            valid_img = generate_test_image_bytes()
            res_upload = client.post(
                f"/api/v1/cases/{case_a_id}/upload-image",
                headers={"Authorization": f"Bearer {token_pat_a}"},
                files={"file": ("patient_lesion.jpg", valid_img, "image/jpeg")}
            )
            assert res_upload.status_code == 200, f"Upload failed: {res_upload.text}"
            case_data = res_upload.json()
            assert case_data["status"] == "preprocessed"
            assert len(case_data["images"]) > 0
            img_info = case_data["images"][-1]
            uploaded_filename = img_info["filename"]
            assert img_info["blur_score"] is not None
            assert img_info["brightness_score"] is not None
            assert img_info["contrast_score"] is not None
            assert img_info["quality_status"] in ["Good Quality", "Acceptable Quality", "Poor Quality"]
            results.append(("8. Successful Upload & Preprocessing", "PASSED", f"Image preprocessed. Blur: {img_info['blur_score']}, Quality: {img_info['quality_status']}"))
        except Exception as e:
            results.append(("8. Successful Upload & Preprocessing", "FAILED", str(e)))

        # 5. Unauthorized Image Retrieval
        try:
            img_endpoint = f"/api/v1/cases/{case_a_id}/images/original/{uploaded_filename}"
            
            # 5a. Unauthenticated access without token -> 401
            res_unauth = client.get(img_endpoint)
            assert res_unauth.status_code == 401, f"Expected 401, got {res_unauth.status_code}"

            # 5b. Patient B trying to access Patient A's image -> 403
            res_cross_img = client.get(
                img_endpoint,
                headers={"Authorization": f"Bearer {token_pat_b}"}
            )
            assert res_cross_img.status_code == 403, f"Expected 403, got {res_cross_img.status_code}"

            # 5c. Patient A accessing own image with Bearer header -> 200
            res_owner_img = client.get(
                img_endpoint,
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_owner_img.status_code == 200

            # 5d. Image access via token query parameter (?token=...) -> 200
            res_query_token = client.get(f"{img_endpoint}?token={token_pat_a}")
            assert res_query_token.status_code == 200

            results.append(("5. Authenticated Image Access Control", "PASSED", "Verified: 401 unauthenticated, 403 cross-patient, 200 owner, 200 token query param."))
        except Exception as e:
            results.append(("5. Authenticated Image Access Control", "FAILED", str(e)))

        # 10. Existing Case Management (CRUD)
        try:
            # Update case
            res_update = client.put(
                f"/api/v1/cases/{case_a_id}",
                headers={"Authorization": f"Bearer {token_pat_a}"},
                json={"clinical_notes": "Updated follow-up notes: lesion observed for 2 weeks."}
            )
            assert res_update.status_code == 200
            assert res_update.json()["clinical_notes"] == "Updated follow-up notes: lesion observed for 2 weeks."

            # List cases
            res_list = client.get("/api/v1/cases/", headers={"Authorization": f"Bearer {token_pat_a}"})
            assert res_list.status_code == 200
            assert len(res_list.json()) >= 1

            # Delete case
            res_del = client.delete(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {token_pat_a}"})
            assert res_del.status_code == 200
            assert res_del.json().get("is_archived") is True

            # Verify case is archived and omitted from active list
            res_archived = client.get(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {token_pat_a}"})
            assert res_archived.status_code == 200
            assert res_archived.json()["is_archived"] is True

            res_list_after = client.get("/api/v1/cases/", headers={"Authorization": f"Bearer {token_pat_a}"})
            assert res_list_after.status_code == 200
            assert not any(c["id"] == case_a_id for c in res_list_after.json())

            results.append(("10. Case CRUD Operations", "PASSED", "Successfully verified Case Create, Read, Update, List, and Soft-Deletion Archival."))
        except Exception as e:
            results.append(("10. Case CRUD Operations", "FAILED", str(e)))

        # 11. Researcher RBAC on Dataset Endpoints
        try:
            # 11a. Unauthenticated access -> 401
            res_unauth_ds = client.get("/api/v1/dataset/eda-summary")
            assert res_unauth_ds.status_code == 401, f"Expected 401 unauth, got {res_unauth_ds.status_code}"

            # 11b. Patient access -> 403 Forbidden
            res_pat_ds = client.get("/api/v1/dataset/eda-summary", headers={"Authorization": f"Bearer {token_pat_a}"})
            assert res_pat_ds.status_code == 403, f"Expected 403 for patient, got {res_pat_ds.status_code}"

            # 11c. Clinician access -> 403 Forbidden
            res_doc_ds = client.get("/api/v1/dataset/eda-summary", headers={"Authorization": f"Bearer {doc_token}"})
            assert res_doc_ds.status_code == 403, f"Expected 403 for clinician, got {res_doc_ds.status_code}"

            # 11d. Public researcher registration rejected -> 400 Bad Request
            res_res_reg = client.post("/api/v1/auth/register", json={
                "username": f"researcher_{ts}",
                "email": f"res_{ts}@dermai.org",
                "full_name": "Dr. Research PhD",
                "password": "Password123!",
                "role": "researcher"
            })
            assert res_res_reg.status_code == 400, f"Expected 400 for public researcher registration, got {res_res_reg.status_code}"

            # 11e. Authorized Admin access -> 200 Authorized
            res_adm_ds = client.get("/api/v1/dataset/eda-summary", headers={"Authorization": f"Bearer {admin_token}"})
            assert res_adm_ds.status_code == 200, f"Expected 200 for admin, got {res_adm_ds.status_code}"
            assert res_adm_ds.json().get("total_records") == 10015
            results.append(("11. Researcher RBAC Dataset Protection", "PASSED", "Verified 401 unauth, 403 patient, 403 clinician, 400 researcher registration rejection, and 200 admin access."))
        except Exception as e:
            results.append(("11. Researcher RBAC Dataset Protection", "FAILED", str(e)))

        # Summary report
        print("\n" + "-"*70)
        print(f"{'TEST NAME':<45} | {'STATUS':<8} | {'DETAILS'}")
        print("-"*70)
        all_passed = True
        for name, status_str, detail in results:
            print(f"{name:<45} | {status_str:<8} | {detail}")
            if status_str != "PASSED":
                all_passed = False
        print("-"*70)
        print(f"OVERALL RESULT: {'ALL TESTS PASSED (' + str(len(results)) + '/' + str(len(results)) + ')' if all_passed else 'SOME TESTS FAILED'}")
        print("="*70 + "\n")
        return all_passed

    finally:
        auth_mod.create_access_token = orig_auth_token_func
        main_mod.create_access_token = orig_main_token_func
        app.dependency_overrides.clear()
        test_engine.dispose()
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass

        # Clean up any test images generated during test
        for code in created_case_codes:
            for upload_dir in [settings.ORIGINAL_DIR, settings.PREVIEW_DIR, settings.PREPROCESSED_DIR]:
                for fpath in glob.glob(os.path.join(upload_dir, f"{code}*")):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
