"""
DermAI Milestone 2 - Stage 4: FastAPI Prediction API Test Suite
Automated validation of MobileNetV2 deep learning inference integration into FastAPI backend:
1. Health endpoint ML service status & metadata reporting
2. Unauthenticated prediction rejection (HTTP 401)
3. Non-existent case prediction rejection (HTTP 404)
4. Case without image prediction rejection (HTTP 400)
5. Cross-patient unauthorized prediction rejection (HTTP 403)
6. Successful image upload & prediction execution (HTTP 200)
7. Prediction response schema & field validation
8. All 7 HAM10000 class probabilities validity & summation (~1.0)
9. Database persistence verification (PredictionRecord record check)
10. Case status lifecycle transition (PREPROCESSED -> COMPLETED)
11. Dedicated case predictions list endpoint verification (GET /cases/{case_id}/predictions)
12. Comprehensive case response serialization (GET /cases/{case_id} contains predictions)
13. Suboptimal image quality warning attachment
14. Clinician cross-case diagnostic evaluation authorization
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
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend directory is in path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.config import settings
from app.database import Base, get_db
from app.models import Case, ImageRecord, PredictionRecord, CaseStatus, User, UserRole
from app.auth import hash_password

client = TestClient(app)


def generate_synthetic_skin_lesion_bytes(blur: bool = False) -> io.BytesIO:
    """Generates a synthetic dermoscopic lesion image in JPEG format."""
    # Skin background tone (BGR)
    img = np.full((300, 300, 3), (180, 200, 240), dtype=np.uint8)
    # Synthetic pigmented lesion (dark brown/irregular)
    cv2.circle(img, (150, 150), 55, (30, 45, 80), -1)
    cv2.ellipse(img, (150, 150), (60, 45), 30, 0, 360, (20, 30, 60), -1)
    # Add subtle texture variations
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    if blur:
        # Aggressive blur to trigger poor quality / is_suitable_for_ai = False
        img = cv2.GaussianBlur(img, (35, 35), 0)

    success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer.tobytes())


def run_prediction_tests():
    print("\n" + "=" * 75)
    print("DERMAI MILESTONE 2 - STAGE 4: PREDICTION API TEST SUITE")
    print("=" * 75)

    # Isolated test database fixture
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix="_pred_api.db")
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
        to_encode.update({"exp": expire, "iat": now_utc - timedelta(minutes=10)})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    auth_mod.create_access_token = safe_create_access_token
    main_mod.create_access_token = safe_create_access_token

    results = []
    ts = os.urandom(4).hex()
    created_case_codes = []

    try:
        # 1. Health Endpoint ML Status
        try:
            res = client.get("/api/v1/health")
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            assert "ml_service" in data, "ml_service key missing in health response"
            ml_info = data["ml_service"]
            assert ml_info["status"] == "online", f"Expected ml_service status 'online', got {ml_info['status']}"
            assert ml_info["model_name"] == "DermAI_MobileNetV2"
            assert ml_info["num_classes"] == 7
            expected_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            assert set(ml_info["classes"]) == set(expected_classes)
            results.append(("1. Health Endpoint ML Status", "PASSED", "ML service reported online with 7 classes and MobileNetV2."))
        except Exception as e:
            results.append(("1. Health Endpoint ML Status", "FAILED", str(e)))

        # Provision Admin directly in isolated DB
        admin_username = f"admin_pred_{ts}"
        admin_password = "AdminSecurePassword123!"
        db = TestingSessionLocal()
        try:
            test_admin = User(
                username=admin_username,
                email=f"admin_pred_{ts}@dermaitest.org",
                full_name="Prediction Admin Controller",
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

        # Setup Test Users
        # Register Patient Alpha
        res_pat_a = client.post("/api/v1/auth/register", json={
            "username": f"patient_alpha_{ts}",
            "email": f"alpha_{ts}@dermaitest.org",
            "full_name": "Patient Alpha",
            "password": "Password123!",
            "role": "patient"
        })
        assert res_pat_a.status_code == 200
        token_pat_a = res_pat_a.json()["access_token"]

        # Register Patient Beta
        res_pat_b = client.post("/api/v1/auth/register", json={
            "username": f"patient_beta_{ts}",
            "email": f"beta_{ts}@dermaitest.org",
            "full_name": "Patient Beta",
            "password": "Password123!",
            "role": "patient"
        })
        assert res_pat_b.status_code == 200
        token_pat_b = res_pat_b.json()["access_token"]

        # Register Clinician and verify via Admin
        res_doc = client.post("/api/v1/auth/register", json={
            "username": f"doc_eval_{ts}",
            "email": f"doc_eval_{ts}@dermaitest.org",
            "full_name": "Dr. Dermatologist",
            "password": "Password123!",
            "role": "clinician"
        })
        assert res_doc.status_code == 200
        doc_data = res_doc.json()
        doc_id = doc_data["user_id"]
        assert doc_data.get("is_verified") is False

        # Verify clinician via Admin
        res_verify = client.patch(
            f"/api/v1/admin/clinicians/{doc_id}/verify",
            headers={"Authorization": f"Bearer {token_admin}"}
        )
        assert res_verify.status_code == 200
        assert res_verify.json()["is_verified"] is True

        # Verified clinician authenticates
        res_doc_login = client.post("/api/v1/auth/login", json={
            "username": f"doc_eval_{ts}",
            "password": "Password123!"
        })
        assert res_doc_login.status_code == 200
        token_doc = res_doc_login.json()["access_token"]

        # Create Case for Patient Alpha
        res_case_a = client.post(
            "/api/v1/cases/",
            json={"patient_age": 42, "patient_gender": "female", "body_location": "back"},
            headers={"Authorization": f"Bearer {token_pat_a}"}
        )
        assert res_case_a.status_code == 200
        case_a_data = res_case_a.json()
        case_a_id = case_a_data["id"]
        created_case_codes.append(case_a_data["case_code"])

        # 2. Prediction Fails Without Authentication (401)
        try:
            res = client.post(f"/api/v1/cases/{case_a_id}/predict")
            assert res.status_code == 401, f"Expected 401, got {res.status_code}"
            results.append(("2. Unauthenticated Prediction Rejection", "PASSED", "Rejected unauthenticated request with HTTP 401."))
        except Exception as e:
            results.append(("2. Unauthenticated Prediction Rejection", "FAILED", str(e)))

        # 3. Prediction Fails for Non-Existent Case (404)
        try:
            res = client.post(
                "/api/v1/cases/999999/predict",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res.status_code == 404, f"Expected 404, got {res.status_code}"
            results.append(("3. Non-Existent Case Prediction Rejection", "PASSED", "Rejected non-existent case with HTTP 404 Not Found."))
        except Exception as e:
            results.append(("3. Non-Existent Case Prediction Rejection", "FAILED", str(e)))

        # 4. Prediction Fails for Case Without an Image (400)
        try:
            res = client.post(
                f"/api/v1/cases/{case_a_id}/predict",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.text}"
            assert "no uploaded images" in res.json()["detail"].lower()
            results.append(("4. Case Without Image Rejection", "PASSED", "Rejected prediction on case without images with HTTP 400."))
        except Exception as e:
            results.append(("4. Case Without Image Rejection", "FAILED", str(e)))

        # Upload Image to Patient Alpha's Case
        img_bytes = generate_synthetic_skin_lesion_bytes(blur=False)
        upload_res = client.post(
            f"/api/v1/cases/{case_a_id}/upload-image",
            files={"file": ("lesion_test.jpg", img_bytes, "image/jpeg")},
            headers={"Authorization": f"Bearer {token_pat_a}"}
        )
        assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"

        # 5. Prediction Fails for Cross-Patient Unauthorized Access (403)
        try:
            res = client.post(
                f"/api/v1/cases/{case_a_id}/predict",
                headers={"Authorization": f"Bearer {token_pat_b}"}
            )
            assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"
            results.append(("5. Cross-Patient Unauthorized Prediction Rejection", "PASSED", "Patient Beta denied inference on Patient Alpha's case (HTTP 403)."))
        except Exception as e:
            results.append(("5. Cross-Patient Unauthorized Prediction Rejection", "FAILED", str(e)))

        # 6. Successful Prediction Execution on Valid Case Image (200)
        pred_data = None
        try:
            res = client.post(
                f"/api/v1/cases/{case_a_id}/predict",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res.status_code == 200, f"Prediction failed: {res.status_code}: {res.text}"
            pred_data = res.json()
            results.append(("6. Successful Prediction Execution", "PASSED", f"Predicted: {pred_data['predicted_class']} ({pred_data['predicted_disease_name']}) with score {pred_data['confidence_score']:.4f}"))
        except Exception as e:
            results.append(("6. Successful Prediction Execution", "FAILED", str(e)))

        # 7. Prediction Response Schema Validation
        try:
            assert pred_data is not None, "Previous prediction failed"
            required_fields = [
                "id", "case_id", "image_id", "predicted_class", "predicted_disease_name",
                "confidence_score", "risk_level", "class_probabilities", "model_name",
                "prediction_timestamp", "disclaimer", "image_quality_status", "is_suitable_for_ai"
            ]
            for field in required_fields:
                assert field in pred_data, f"Required field '{field}' missing from prediction response"

            assert pred_data["case_id"] == case_a_id
            assert pred_data["model_name"] == "DermAI_MobileNetV2"
            assert pred_data["risk_level"] in ["High Risk", "Moderate Risk", "Low Risk"]
            assert len(pred_data["disclaimer"]) > 20
            results.append(("7. Prediction Response Schema Validation", "PASSED", "All 13 required fields validated against Pydantic schema."))
        except Exception as e:
            results.append(("7. Prediction Response Schema Validation", "FAILED", str(e)))

        # 8. All 7 Class Probabilities Summation & Validity
        try:
            probs = pred_data["class_probabilities"]
            assert len(probs) == 7, f"Expected 7 classes, got {len(probs)}"
            expected_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            for cls in expected_classes:
                assert cls in probs, f"Class '{cls}' missing from probabilities"
                assert isinstance(probs[cls], (int, float)), f"Probability for '{cls}' is not float"
                assert 0.0 <= probs[cls] <= 1.0, f"Probability for '{cls}' out of range: {probs[cls]}"

            total_prob = sum(probs.values())
            # Softmax probabilities should sum to approximately 1.0 (allowing for 4-decimal rounding)
            assert 0.98 <= total_prob <= 1.02, f"Probabilities sum to {total_prob}, expected ~1.0"
            results.append(("8. 7-Class Probabilities Validity & Sum", "PASSED", f"Validated 7 classes. Sum: {total_prob:.4f} (~1.0)."))
        except Exception as e:
            results.append(("8. 7-Class Probabilities Validity & Sum", "FAILED", str(e)))

        # 9. Database Persistence Verification
        try:
            db = TestingSessionLocal()
            try:
                db_pred = db.query(PredictionRecord).filter(
                    PredictionRecord.case_id == case_a_id,
                    PredictionRecord.id == pred_data["id"]
                ).first()
                assert db_pred is not None, "PredictionRecord not found in database"
                assert db_pred.predicted_class == pred_data["predicted_class"]
                assert db_pred.confidence_score == pred_data["confidence_score"]
                parsed_probs = json.loads(db_pred.class_probabilities_json)
                assert len(parsed_probs) == 7
                results.append(("9. Database Persistence Verification", "PASSED", f"Confirmed PredictionRecord id={db_pred.id} persisted in database."))
            finally:
                db.close()
        except Exception as e:
            results.append(("9. Database Persistence Verification", "FAILED", str(e)))

        # 10. Case Status Lifecycle Transition
        try:
            db = TestingSessionLocal()
            try:
                db_case = db.query(Case).filter(Case.id == case_a_id).first()
                assert db_case.status == CaseStatus.COMPLETED, f"Expected CaseStatus.COMPLETED, got {db_case.status}"
                results.append(("10. Case Status Lifecycle Transition", "PASSED", "Case status successfully progressed to CaseStatus.COMPLETED."))
            finally:
                db.close()
        except Exception as e:
            results.append(("10. Case Status Lifecycle Transition", "FAILED", str(e)))

        # 11. Dedicated Case Predictions Endpoint (GET /cases/{case_id}/predictions)
        try:
            res = client.get(
                f"/api/v1/cases/{case_a_id}/predictions",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            preds_list = res.json()
            assert len(preds_list) >= 1, "Expected at least 1 prediction in list"
            assert preds_list[0]["id"] == pred_data["id"]
            results.append(("11. Case Predictions Endpoint", "PASSED", f"GET /cases/{case_a_id}/predictions returned {len(preds_list)} records."))
        except Exception as e:
            results.append(("11. Case Predictions Endpoint", "FAILED", str(e)))

        # 12. Case Response Serialization Includes Predictions (GET /cases/{case_id})
        try:
            res = client.get(
                f"/api/v1/cases/{case_a_id}",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            case_data = res.json()
            assert "predictions" in case_data, "predictions key missing in case response"
            assert len(case_data["predictions"]) >= 1
            assert case_data["predictions"][0]["id"] == pred_data["id"]
            results.append(("12. Case Response Serialization", "PASSED", "Verified GET /cases/{case_id} contains serialized predictions list."))
        except Exception as e:
            results.append(("12. Case Response Serialization", "FAILED", str(e)))

        # 13. Suboptimal Image Quality Warning Attachment
        try:
            # Create Case for Suboptimal Image Test
            res_case_sub = client.post(
                "/api/v1/cases/",
                json={"patient_age": 55, "patient_gender": "male", "body_location": "arm"},
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_case_sub.status_code == 200
            case_sub_data = res_case_sub.json()
            case_sub_id = case_sub_data["id"]
            created_case_codes.append(case_sub_data["case_code"])

            # Upload heavily blurred image to trigger poor quality
            blurry_bytes = generate_synthetic_skin_lesion_bytes(blur=True)
            upload_sub = client.post(
                f"/api/v1/cases/{case_sub_id}/upload-image",
                files={"file": ("blurry_lesion.jpg", blurry_bytes, "image/jpeg")},
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert upload_sub.status_code == 200

            # Run prediction
            res_sub = client.post(
                f"/api/v1/cases/{case_sub_id}/predict",
                headers={"Authorization": f"Bearer {token_pat_a}"}
            )
            assert res_sub.status_code == 200
            sub_pred = res_sub.json()
            assert sub_pred["is_suitable_for_ai"] is False
            assert "QUALITY WARNING" in sub_pred["disclaimer"]
            results.append(("13. Suboptimal Image Warning", "PASSED", "Preserved is_suitable_for_ai=False and attached QUALITY WARNING to disclaimer."))
        except Exception as e:
            results.append(("13. Suboptimal Image Warning", "FAILED", str(e)))

        # 14. Clinician Cross-Case Diagnostic Authorization
        try:
            # Clinician evaluates Patient Alpha's case
            res_clin = client.post(
                f"/api/v1/cases/{case_a_id}/predict",
                headers={"Authorization": f"Bearer {token_doc}"}
            )
            assert res_clin.status_code == 200, f"Clinician access denied: {res_clin.status_code}: {res_clin.text}"
            results.append(("14. Clinician Diagnostic Authorization", "PASSED", "Clinician authorized to run predictions on patient cases."))
        except Exception as e:
            results.append(("14. Clinician Diagnostic Authorization", "FAILED", str(e)))

        # Print Summary Table
        print("\n" + "-" * 75)
        print(f"{'TEST NAME':<45} | {'STATUS':<8} | {'DETAILS'}")
        print("-" * 75)
        passed_count = sum(1 for _, status, _ in results if status == "PASSED")
        for name, status, details in results:
            print(f"{name:<45} | {status:<8} | {details}")
        print("-" * 75)
        print(f"OVERALL RESULT: {passed_count}/{len(results)} TESTS PASSED")
        print("=" * 75 + "\n")

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
    run_prediction_tests()
