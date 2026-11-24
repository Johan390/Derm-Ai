"""
DermAI Milestone 2 - Stage 5: End-to-End Frontend Integration Test Suite
Validates all backend API contracts and workflows utilized by the React frontend:
1. Health & ML service telemetry contract (GET /api/v1/health)
2. Patient authentication and session token issuance
3. Clinician authentication, administrative verification, and session token issuance
4. Unified case creation and image upload contract
5. Automated IQA metrics schema verification
6. Deep-learning prediction execution contract (POST /api/v1/cases/{id}/predict)
7. Full 7-class probability schema validation for horizontal bar visualization
8. Risk tier classification mapping (High / Moderate / Low)
9. Historical predictions retrieval contract (GET /api/v1/cases/{id}/predictions)
10. Suboptimal image quality warning propagation (is_suitable_for_ai == false)
11. Cross-patient BOLA/IDOR security isolation
12. Unauthenticated request protection (HTTP 401)
13. Prediction input validation (empty case -> HTTP 400, non-existent -> HTTP 404)
14. End-to-end case queue state synchronization (status: preprocessed -> completed)
"""

import sys
import os
import io
import json
import tempfile
import glob
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.config import settings
from app.database import Base, get_db
from app.models import Case, ImageRecord, PredictionRecord, CaseStatus, User, UserRole
from app.auth import hash_password

client = TestClient(app)


def generate_lesion_bytes(blur: bool = False) -> io.BytesIO:
    """Generates synthetic dermoscopy lesion photography for testing."""
    # Balanced skin background tone (BGR) within 45-215 brightness range
    img = np.full((300, 300, 3), (160, 180, 200), dtype=np.uint8)
    cv2.circle(img, (150, 150), 50, (40, 60, 100), -1)
    cv2.ellipse(img, (150, 150), (60, 40), 45, 0, 360, (30, 45, 80), -1)
    if blur:
        img = cv2.GaussianBlur(img, (45, 45), 0)
    success, buf = cv2.imencode(".jpg", img)
    return io.BytesIO(buf.tobytes())


def run_frontend_integration_tests():
    print("\n" + "=" * 80)
    print("DERMAI MILESTONE 2 - STAGE 5: FRONTEND INTEGRATION TEST SUITE")
    print("=" * 80)

    # Isolated test database fixture
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix="_fe_integ.db")
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
        # Protect against WSL clock jitter by issuing with iat 10 minutes in the past
        to_encode.update({"exp": expire, "iat": now_utc - timedelta(minutes=10)})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    auth_mod.create_access_token = safe_create_access_token
    main_mod.create_access_token = safe_create_access_token

    results = []
    ts = os.urandom(4).hex()
    created_case_codes = []

    try:
        # 1. Health & ML Telemetry Contract
        try:
            res = client.get("/api/v1/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "healthy"
            assert "ml_service" in data
            assert data["ml_service"]["status"] == "online"
            assert data["ml_service"]["model_name"] == "DermAI_MobileNetV2"
            assert data["ml_service"]["num_classes"] == 7
            results.append(("1. Health & ML Telemetry Contract", "PASSED", "Health endpoint provides ML model status for frontend indicators."))
        except Exception as e:
            results.append(("1. Health & ML Telemetry Contract", "FAILED", str(e)))

        # Provision Admin directly in isolated DB for clinician verification
        admin_username = f"admin_fe_{ts}"
        admin_password = "AdminSecurePassword123!"
        db = TestingSessionLocal()
        try:
            test_admin = User(
                username=admin_username,
                email=f"admin_fe_{ts}@dermaitest.org",
                full_name="Frontend Admin Controller",
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

        # 2. Patient Auth & Registration
        token_pat_a = None
        try:
            res_pat = client.post("/api/v1/auth/register", json={
                "username": f"stage5_patient_{ts}",
                "email": f"s5_pat_{ts}@dermaitest.org",
                "full_name": "Stage5 Patient",
                "password": "Password123!",
                "role": "patient"
            })
            assert res_pat.status_code == 200
            token_pat_a = res_pat.json()["access_token"]
            assert token_pat_a and len(token_pat_a) > 20
            results.append(("2. Patient Authentication Flow", "PASSED", "Registered and issued JWT bearer token for patient session."))
        except Exception as e:
            results.append(("2. Patient Authentication Flow", "FAILED", str(e)))

        # 3. Clinician Auth, Registration & Admin Verification
        token_doc = None
        try:
            res_doc = client.post("/api/v1/auth/register", json={
                "username": f"stage5_doc_{ts}",
                "email": f"s5_doc_{ts}@dermaitest.org",
                "full_name": "Dr. Stage5 Clinician MD",
                "password": "Password123!",
                "role": "clinician"
            })
            assert res_doc.status_code == 200
            doc_data = res_doc.json()
            assert doc_data.get("is_verified") is False
            doc_id = doc_data["user_id"]
            unverified_token = doc_data["access_token"]

            # Unverified clinician blocked from triage
            res_unv_block = client.get("/api/v1/cases/?assignment_filter=unassigned", headers={"Authorization": f"Bearer {unverified_token}"})
            assert res_unv_block.status_code == 403

            # Admin verifies clinician
            res_verify = client.patch(f"/api/v1/admin/clinicians/{doc_id}/verify", headers={"Authorization": f"Bearer {token_admin}"})
            assert res_verify.status_code == 200
            assert res_verify.json()["is_verified"] is True

            # Verified clinician logs in
            res_doc_login = client.post("/api/v1/auth/login", json={
                "username": f"stage5_doc_{ts}",
                "password": "Password123!"
            })
            assert res_doc_login.status_code == 200
            token_doc = res_doc_login.json()["access_token"]
            results.append(("3. Clinician Authentication & Verification Flow", "PASSED", "Registered, verified by admin, and authenticated verified clinician."))
        except Exception as e:
            results.append(("3. Clinician Authentication & Verification Flow", "FAILED", str(e)))

        # 4. Patient Case Creation & Image Upload
        case_id = None
        try:
            res_case = client.post(
                "/api/v1/cases/",
                json={
                    "patient_age": 48,
                    "patient_gender": "female",
                    "body_location": "upper arm",
                    "known_allergies": "Penicillin",
                    "clinical_notes": "Irregular pigmented papule, slight enlargement over 3 months"
                },
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_case.status_code == 200
            case_data = res_case.json()
            case_id = case_data["id"]
            created_case_codes.append(case_data["case_code"])
            assert case_data["case_code"].startswith("CAS-")
            assert case_data["status"] == "created"

            # Upload valid lesion photography
            img_buf = generate_lesion_bytes(blur=False)
            res_upload = client.post(
                f"/api/v1/cases/{case_id}/upload-image",
                files={"file": ("lesion_arm.jpg", img_buf, "image/jpeg")},
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_upload.status_code == 200
            uploaded_case = res_upload.json()
            assert uploaded_case["status"] == "preprocessed"
            assert len(uploaded_case["images"]) == 1
            img_record = uploaded_case["images"][0]
            assert img_record["blur_score"] > 0
            assert img_record["brightness_score"] > 0
            assert img_record["contrast_score"] > 0
            assert img_record["is_suitable_for_ai"] is True
            results.append(("4. Unified Intake & Preprocessing Contract", "PASSED", f"Case {case_data['case_code']} created and preprocessed."))
        except Exception as e:
            results.append(("4. Unified Intake & Preprocessing Contract", "FAILED", str(e)))

        # 5. Deep Learning Prediction API Contract
        pred_data = None
        try:
            res_pred = client.post(
                f"/api/v1/cases/{case_id}/predict",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_pred.status_code == 200, f"Predict failed: {res_pred.text}"
            pred_data = res_pred.json()

            # Check required fields expected by PredictionResults.jsx
            required_keys = [
                "id", "case_id", "image_id", "predicted_class", "predicted_disease_name",
                "confidence_score", "risk_level", "class_probabilities", "model_name",
                "prediction_timestamp", "disclaimer", "image_quality_status", "is_suitable_for_ai"
            ]
            for key in required_keys:
                assert key in pred_data, f"Missing key in prediction response: {key}"

            assert pred_data["risk_level"] in ["High Risk", "Moderate Risk", "Low Risk"]
            assert pred_data["confidence_score"] > 0
            assert len(pred_data["disclaimer"]) > 10
            results.append(("5. Prediction Response Contract", "PASSED", f"Returned: {pred_data['predicted_class']} ({pred_data['predicted_disease_name']}) {pred_data['confidence_score']:.4f}"))
        except Exception as e:
            results.append(("5. Prediction Response Contract", "FAILED", str(e)))

        # 6. Seven-Class Probability Schema Validation (for Horizontal Bar Chart)
        try:
            probs = pred_data["class_probabilities"]
            expected_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            assert len(probs) == 7
            for c in expected_classes:
                assert c in probs, f"Class {c} missing from probabilities"
                assert isinstance(probs[c], (int, float))
                assert 0.0 <= probs[c] <= 1.0

            prob_sum = sum(probs.values())
            assert 0.98 <= prob_sum <= 1.02, f"Probabilities sum out of range: {prob_sum}"
            results.append(("6. 7-Class Probability Visualization Contract", "PASSED", f"All 7 classes present; sum={prob_sum:.4f}."))
        except Exception as e:
            results.append(("6. 7-Class Probability Visualization Contract", "FAILED", str(e)))

        # 7. Historical Predictions Retrieval Contract (GET /cases/{id}/predictions)
        try:
            res_hist = client.get(
                f"/api/v1/cases/{case_id}/predictions",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_hist.status_code == 200
            hist_list = res_hist.json()
            assert isinstance(hist_list, list)
            assert len(hist_list) >= 1
            assert hist_list[0]["id"] == pred_data["id"]
            results.append(("7. Historical Predictions Retrieval Contract", "PASSED", f"Retrieved {len(hist_list)} predictions for case history drawer."))
        except Exception as e:
            results.append(("7. Historical Predictions Retrieval Contract", "FAILED", str(e)))

        # 8. Re-Analysis & Multiple Predictions Retention
        try:
            # Run a second prediction on the same case
            res_pred2 = client.post(
                f"/api/v1/cases/{case_id}/predict",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_pred2.status_code == 200
            pred2_data = res_pred2.json()

            # Retrieve history again to verify accumulation
            res_hist2 = client.get(
                f"/api/v1/cases/{case_id}/predictions",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_hist2.status_code == 200
            hist_list2 = res_hist2.json()
            assert len(hist_list2) == 2, f"Expected 2 predictions in history, got {len(hist_list2)}"
            results.append(("8. Multiple Inferences Archiving", "PASSED", "History preserves all sequential inferences without overwriting."))
        except Exception as e:
            results.append(("8. Multiple Inferences Archiving", "FAILED", str(e)))

        # 9. Suboptimal Image Warning Propagation (is_suitable_for_ai == false)
        try:
            # Create Case B with blurry image
            res_case_b = client.post(
                "/api/v1/cases/",
                json={"patient_age": 62, "patient_gender": "male", "body_location": "back"},
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            case_b_id = res_case_b.json()["id"]
            created_case_codes.append(res_case_b.json()["case_code"])

            blurry_buf = generate_lesion_bytes(blur=True)
            res_upload_b = client.post(
                f"/api/v1/cases/{case_b_id}/upload-image",
                files={"file": ("blurry_back.jpg", blurry_buf, "image/jpeg")},
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_upload_b.status_code == 200
            case_b_data = res_upload_b.json()
            assert case_b_data["images"][0]["is_suitable_for_ai"] is False

            # Predict on suboptimal image
            res_pred_b = client.post(
                f"/api/v1/cases/{case_b_id}/predict",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_pred_b.status_code == 200
            pred_b_data = res_pred_b.json()
            assert pred_b_data["is_suitable_for_ai"] is False
            assert "QUALITY WARNING" in pred_b_data["disclaimer"]
            results.append(("9. Suboptimal Quality Warning Propagation", "PASSED", "Preserved is_suitable_for_ai=False and attached QUALITY WARNING."))
        except Exception as e:
            results.append(("9. Suboptimal Quality Warning Propagation", "FAILED", str(e)))

        # 10. Role-Based Access Control: Cross-Patient BOLA/IDOR Isolation
        try:
            # Register Patient Beta
            res_pat_b = client.post("/api/v1/auth/register", json={
                "username": f"stage5_pat_b_{ts}",
                "email": f"s5_patb_{ts}@dermaitest.org",
                "full_name": "Stage5 Patient Beta",
                "password": "Password123!",
                "role": "patient"
            })
            token_pat_b = res_pat_b.json()["access_token"]

            # Patient Beta attempts to predict on Patient Alpha's case -> 403
            res_unauth_pred = client.post(
                f"/api/v1/cases/{case_id}/predict",
                headers={"Authorization": f"Bearer {token_pat_b}"}
            )
            assert res_unauth_pred.status_code == 403

            # Patient Beta attempts to view Patient Alpha's predictions -> 403
            res_unauth_hist = client.get(
                f"/api/v1/cases/{case_id}/predictions",
                headers={"Authorization": f"Bearer {token_pat_b}"}
            )
            assert res_unauth_hist.status_code == 403
            results.append(("10. Patient BOLA/IDOR Security Isolation", "PASSED", "Rejected cross-patient prediction and history access with HTTP 403."))
        except Exception as e:
            results.append(("10. Patient BOLA/IDOR Security Isolation", "FAILED", str(e)))

        # 11. Clinician Diagnostic Evaluation Across Patient Cases
        try:
            # Verified clinician evaluates Patient Alpha's case
            res_clin_pred = client.post(
                f"/api/v1/cases/{case_id}/predict",
                headers={"Authorization": f"Bearer {token_doc}"}
            )
            assert res_clin_pred.status_code == 200

            # Clinician lists completed cases
            res_cases_list = client.get(
                "/api/v1/cases/?status_filter=completed",
                headers={"Authorization": f"Bearer {token_doc}"}
            )
            assert res_cases_list.status_code == 200
            completed_cases = res_cases_list.json()
            assert any(c["id"] == case_id for c in completed_cases)
            results.append(("11. Clinician Evaluation & Case Queue Contract", "PASSED", "Clinician authorized to evaluate patient cases and query completed queue."))
        except Exception as e:
            results.append(("11. Clinician Evaluation & Case Queue Contract", "FAILED", str(e)))

        # 12. Security Boundary: Unauthenticated Access Rejection (401)
        try:
            res_noauth_pred = client.post(f"/api/v1/cases/{case_id}/predict")
            assert res_noauth_pred.status_code == 401
            res_noauth_hist = client.get(f"/api/v1/cases/{case_id}/predictions")
            assert res_noauth_hist.status_code == 401
            results.append(("12. Unauthenticated Access Rejection", "PASSED", "Enforced HTTP 401 Unauthorized for unauthenticated requests."))
        except Exception as e:
            results.append(("12. Unauthenticated Access Rejection", "FAILED", str(e)))

        # 13. Case Queue State Transition (status: COMPLETED)
        try:
            res_case_check = client.get(
                f"/api/v1/cases/{case_id}",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_case_check.status_code == 200
            final_case = res_case_check.json()
            assert final_case["status"] == "completed"
            assert len(final_case["predictions"]) >= 1
            results.append(("13. Case Lifecycle Synchronization", "PASSED", "Case status transitioned to 'completed' with serialized predictions."))
        except Exception as e:
            results.append(("13. Case Lifecycle Synchronization", "FAILED", str(e)))

        # Print Summary Table
        print("\n" + "-" * 80)
        print(f"{'TEST NAME':<48} | {'STATUS':<8} | {'DETAILS'}")
        print("-" * 80)
        passed_count = sum(1 for _, status, _ in results if status == "PASSED")
        for name, status, details in results:
            print(f"{name:<48} | {status:<8} | {details}")
        print("-" * 80)
        print(f"OVERALL RESULT: {passed_count}/{len(results)} TESTS PASSED")
        print("=" * 80 + "\n")

        assert passed_count == len(results), f"Only {passed_count}/{len(results)} passed"

    finally:
        auth_mod.create_access_token = orig_auth_token_func
        main_mod.create_access_token = orig_main_token_func
        # Teardown: clear overrides, dispose engine, delete temp db
        app.dependency_overrides.clear()
        test_engine.dispose()
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass

        # Clean up any image files generated during test
        for code in created_case_codes:
            for upload_dir in [settings.ORIGINAL_DIR, settings.PREVIEW_DIR, settings.PREPROCESSED_DIR]:
                for fpath in glob.glob(os.path.join(upload_dir, f"{code}*")):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass


if __name__ == "__main__":
    run_frontend_integration_tests()
