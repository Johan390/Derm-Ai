"""
DermAI Realistic End-to-End User Workflow & Journey Test Suite

Executes realistic end-to-end user journeys representing actual clinical workflows
under the current DermAI 3-role architecture (Patient, Clinician, Administrator):

- Journey D: Administrator Journey (Provisioned Admin -> Health Check -> Pending Clinicians -> Model Governance)
- Journey A: Patient Journey (Registration -> Login -> Create Case -> Upload Real Image -> IQA -> Predict -> Notify)
- Journey B: Clinician Journey (Registration -> Pending Triage Guard 403 -> Admin Approval -> Claim -> Grad-CAM -> Review -> Reminder -> PDF)
- Journey C: Role Security & Registration Boundaries (Reject Admin/Researcher/ML Engineer/Invalid Public Registration with HTTP 400)
- Journey E: Privilege Separation & RBAC Enforcement (Patient/Clinician Barriers, Non-Admin Model Blocks, Unauthenticated 401)
- Journey F: Clinical Edge Cases & Security Robustness (Corrupt Upload 400 -> Status Preservation -> Retry Upload -> Cross-Patient BOLA 403 -> Post-Review Freeze 403)

Outputs structured test artifacts to:
- `END_TO_END_TEST_RESULTS.json`
"""

import os
import sys
import io
import json
import time
import tempfile
from datetime import datetime, timezone
import numpy as np
import cv2
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, "backend") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from app.database import Base, get_db
from app.models import User, UserRole
from app.auth import hash_password
from app.main import app

client = TestClient(app)
OUTPUT_JSON = os.path.join(BASE_DIR, "END_TO_END_TEST_RESULTS.json")
REAL_IMAGE_PATH = os.path.join(BASE_DIR, "ml_dataset", "HAM10000", "images", "ISIC_0024306.jpg")
REGISTRY_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "model_registry.json")


def load_real_image_bytes():
    if os.path.isfile(REAL_IMAGE_PATH):
        with open(REAL_IMAGE_PATH, "rb") as f:
            return f.read()
    img = np.full((300, 300, 3), (180, 200, 240), dtype=np.uint8)
    cv2.circle(img, (150, 150), 40, (40, 60, 100), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def run_e2e_tests():
    print("=" * 80)
    print("DERMAI: REALISTIC END-TO-END CLINICAL & USER JOURNEY TEST SUITE")
    print("=" * 80)

    # Isolated test database fixture to avoid modifying production or persistent state
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix="_e2e.db")
    os.close(temp_db_fd)
    test_db_url = f"sqlite:///{temp_db_path}"
    test_engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Initialize isolated schema
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Backup pristine model registry content to prevent test pollution
    original_registry_content = None
    if os.path.isfile(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as rf:
                original_registry_content = rf.read()
        except Exception:
            pass

    ts = os.urandom(4).hex()
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "journeys": {},
        "summary": {}
    }

    passed_count = 0
    failed_count = 0

    admin_username = f"admin_e2e_{ts}"
    admin_password = "AdminSecurePassword123!"
    patient_username = f"patient_e2e_{ts}"
    patient_password = "Password123!"
    clinician_username = f"dr_curie_{ts}"
    clinician_password = "Password123!"

    admin_token = None
    patient_token = None
    patient_case_id = None
    clinician_token = None

    def get_admin_auth_header():
        nonlocal admin_token
        res = client.post("/api/v1/auth/login", json={
            "username": admin_username,
            "password": admin_password
        })
        if res.status_code == 200:
            admin_token = res.json()["access_token"]
        return {"Authorization": f"Bearer {admin_token}"}

    def get_patient_auth_header():
        nonlocal patient_token
        res = client.post("/api/v1/auth/login", json={
            "username": patient_username,
            "password": patient_password
        })
        if res.status_code == 200:
            patient_token = res.json()["access_token"]
        return {"Authorization": f"Bearer {patient_token}"}

    def get_clinician_auth_header():
        nonlocal clinician_token
        res = client.post("/api/v1/auth/login", json={
            "username": clinician_username,
            "password": clinician_password
        })
        if res.status_code == 200:
            clinician_token = res.json()["access_token"]
        return {"Authorization": f"Bearer {clinician_token}"}

    try:
        # -------------------------------------------------------------
        # JOURNEY D: System Administrator Setup & Health Check
        # -------------------------------------------------------------
        print("\n--- [Journey D] Administrator Journey ---")
        j_admin = {"steps": [], "passed": True}
        try:
            # D1. Provision test-only administrator directly in isolated DB (not via public registration)
            db = TestingSessionLocal()
            try:
                test_admin = User(
                    username=admin_username,
                    email=f"admin_{ts}@dermai.org",
                    full_name="Chief Administrator",
                    role=UserRole.ADMIN,
                    institution="DermAI Clinical Administration",
                    hashed_password=hash_password(admin_password),
                    is_active=True,
                    is_verified=True
                )
                db.add(test_admin)
                db.commit()
            finally:
                db.close()

            # D2. Admin Authenticates via Login API Endpoint
            res_adm_login = client.post("/api/v1/auth/login", json={
                "username": admin_username,
                "password": admin_password
            })
            assert res_adm_login.status_code == 200, f"Admin login failed: {res_adm_login.text}"
            admin_token = res_adm_login.json()["access_token"]
            j_admin["steps"].append({"step": "Admin Authentication (Provisioned Admin Login)", "status": "PASS"})

            # D3. System Health Verification
            res_health = client.get("/api/v1/health")
            assert res_health.status_code == 200
            assert res_health.json()["status"] == "healthy"
            j_admin["steps"].append({"step": "System Health Verification", "status": "PASS"})

            # D4. Admin Inspects Pending Clinicians Queue
            res_pending = client.get(
                "/api/v1/admin/clinicians/pending",
                headers=get_admin_auth_header()
            )
            assert res_pending.status_code == 200
            j_admin["steps"].append({"step": "Admin Pending Clinicians Queue Inspection", "status": "PASS"})

            # D5. Model Registry Inspection (All 4 Checkpoint Architectures)
            res_reg = client.get(
                "/api/v1/models/registry",
                headers=get_admin_auth_header()
            )
            assert res_reg.status_code == 200
            models = res_reg.json()
            assert len(models) >= 4
            j_admin["steps"].append({"step": "Model Registry Inspection (4 architectures)", "status": "PASS"})

            # D6. Training History & Lineage Governance
            res_hist = client.get(
                "/api/v1/models/training-history",
                headers=get_admin_auth_header()
            )
            assert res_hist.status_code == 200
            hist_data = res_hist.json()
            assert len(hist_data.get("training_runs", [])) >= 1
            j_admin["steps"].append({"step": "Training Lineage & Hyperparameter Audit", "status": "PASS"})

            # D7. Safe Model Activation & Rollback
            res_act = client.post(
                "/api/v1/models/active",
                headers=get_admin_auth_header(),
                json={"target_version": "v1.1", "reason": "E2E active switch to EfficientNetB0"}
            )
            assert res_act.status_code == 200
            assert res_act.json()["active_version"] == "v1.1"

            res_rev = client.post(
                "/api/v1/models/active",
                headers=get_admin_auth_header(),
                json={"target_version": "v1.0", "reason": "E2E rollback to MobileNetV2"}
            )
            assert res_rev.status_code == 200
            assert res_rev.json()["active_version"] == "v1.0"
            j_admin["steps"].append({"step": "Safe Model Activation & Rollback", "status": "PASS"})

            print("  [PASS] Journey D (Administrator) Completed Successfully.")
            passed_count += 1
        except Exception as e:
            j_admin["passed"] = False
            j_admin["error"] = str(e)
            print(f"  [FAIL] Journey D Failed: {e}")
            failed_count += 1
        results["journeys"]["journey_d_administrator"] = j_admin

        # -------------------------------------------------------------
        # JOURNEY A: Patient Journey
        # -------------------------------------------------------------
        print("\n--- [Journey A] Patient Journey ---")
        j_patient = {"steps": [], "passed": True}
        try:
            # A1. Registration & Token Issuance
            res_p_reg = client.post("/api/v1/auth/register", json={
                "username": patient_username, "email": f"patient_{ts}@e2e.org",
                "full_name": "Eleanor Vance", "password": patient_password, "role": "patient"
            })
            assert res_p_reg.status_code == 200
            patient_token = res_p_reg.json()["access_token"]
            patient_id = res_p_reg.json()["user_id"]
            j_patient["steps"].append({"step": "Patient Registration & Token Issuance", "status": "PASS"})

            # A2. Create Case
            res_case = client.post(
                "/api/v1/cases/",
                headers=get_patient_auth_header(),
                json={
                    "patient_age": 47, "patient_gender": "female",
                    "body_location": "upper extremity", "clinical_notes": "Slightly raised pigmented lesion on forearm",
                    "known_allergies": "Penicillin", "medical_history": "No prior melanoma history"
                }
            )
            assert res_case.status_code == 200
            case_data = res_case.json()
            patient_case_id = case_data["id"]
            case_code = case_data["case_code"]
            j_patient["steps"].append({"step": f"Case Creation ({case_code})", "status": "PASS", "case_id": patient_case_id})

            # A3. Upload Real Image
            real_img = load_real_image_bytes()
            res_upload = client.post(
                f"/api/v1/cases/{patient_case_id}/upload-image",
                headers=get_patient_auth_header(),
                files={"file": ("lesion_forearm.jpg", real_img, "image/jpeg")}
            )
            assert res_upload.status_code == 200
            upload_data = res_upload.json()
            assert upload_data["status"] == "preprocessed"
            img_info = upload_data["images"][0]
            assert img_info["blur_score"] is not None
            assert img_info["brightness_score"] is not None
            j_patient["steps"].append({
                "step": "Real Dermatoscopy Image Upload & IQA Evaluation",
                "status": "PASS",
                "blur_score": img_info["blur_score"],
                "brightness_score": img_info["brightness_score"],
                "contrast_score": img_info["contrast_score"]
            })

            # A4. Run AI Prediction (MobileNetV2 Baseline)
            res_pred = client.post(
                f"/api/v1/cases/{patient_case_id}/predict?model=mobilenet",
                headers=get_patient_auth_header()
            )
            assert res_pred.status_code == 200
            pred_data = res_pred.json()
            assert pred_data["predicted_class"] in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            assert "confidence_score" in pred_data
            assert "top_three_predictions" in pred_data
            assert "top_five_predictions" in pred_data
            j_patient["steps"].append({
                "step": "AI Prediction & Top-N Differential Ranking",
                "status": "PASS",
                "predicted_class": pred_data["predicted_class"],
                "confidence": pred_data["confidence_score"]
            })

            # A5. Check Notifications (follow redirects)
            res_notif = client.get(
                "/api/v1/notifications",
                headers=get_patient_auth_header(),
                follow_redirects=True
            )
            assert res_notif.status_code == 200
            notifs = res_notif.json()
            assert len(notifs) >= 1
            j_patient["steps"].append({"step": "In-App Notification Delivery", "status": "PASS", "notification_count": len(notifs)})

            print("  [PASS] Journey A (Patient) Completed Successfully.")
            passed_count += 1
        except Exception as e:
            j_patient["passed"] = False
            j_patient["error"] = str(e)
            print(f"  [FAIL] Journey A Failed: {e}")
            failed_count += 1
        results["journeys"]["journey_a_patient"] = j_patient

        # -------------------------------------------------------------
        # JOURNEY B: Clinician Journey
        # -------------------------------------------------------------
        print("\n--- [Journey B] Clinician Journey ---")
        j_clinician = {"steps": [], "passed": True}
        try:
            # B1. Register New Clinician (defaults to is_verified = False)
            res_c_reg = client.post("/api/v1/auth/register", json={
                "username": clinician_username, "email": f"dr_curie_{ts}@hospital.org",
                "full_name": "Dr. Marie Curie MD", "password": clinician_password, "role": "clinician",
                "institution": "Marie Curie Dermatology Center"
            })
            assert res_c_reg.status_code == 200
            new_c_token = res_c_reg.json()["access_token"]
            new_c_id = res_c_reg.json()["user_id"]
            j_clinician["steps"].append({"step": "Clinician Registration", "status": "PASS"})

            # B2. Verify Unapproved Clinician is Blocked from Case Management (403)
            res_block = client.get(
                "/api/v1/cases/?assignment_filter=unassigned",
                headers={"Authorization": f"Bearer {new_c_token}"}
            )
            assert res_block.status_code == 403
            j_clinician["steps"].append({"step": "Unverified Clinician Triage Guard (403)", "status": "PASS"})

            # B3. Admin Lists Pending Clinicians & Approves (using provisioned admin_token)
            res_pending = client.get(
                "/api/v1/admin/clinicians/pending",
                headers=get_admin_auth_header()
            )
            assert res_pending.status_code == 200
            pending_ids = [c["id"] for c in res_pending.json()]
            assert new_c_id in pending_ids

            res_approve = client.patch(
                f"/api/v1/admin/clinicians/{new_c_id}/verify",
                headers=get_admin_auth_header()
            )
            assert res_approve.status_code == 200
            assert res_approve.json()["is_verified"] is True
            j_clinician["steps"].append({"step": "Admin Approval & Verification", "status": "PASS"})

            # B4. Clinician Re-authenticates & Accesses Triage Queue
            res_c_login = client.post("/api/v1/auth/login", json={
                "username": clinician_username,
                "password": clinician_password
            })
            assert res_c_login.status_code == 200
            clinician_token = res_c_login.json()["access_token"]

            res_triage = client.get(
                "/api/v1/cases/?assignment_filter=unassigned",
                headers=get_clinician_auth_header()
            )
            assert res_triage.status_code == 200
            j_clinician["steps"].append({"step": "Verified Clinician Triage Access (200)", "status": "PASS"})

            # B5. Clinician Claims Patient Case
            res_claim = client.put(
                f"/api/v1/cases/{patient_case_id}/assign",
                headers=get_clinician_auth_header(),
                json={}
            )
            assert res_claim.status_code == 200
            j_clinician["steps"].append({"step": "Case Triage Claiming", "status": "PASS"})

            # B6. Inspect Explainability (Grad-CAM Overlay)
            res_cam = client.get(
                f"/api/v1/cases/{patient_case_id}/gradcam/overlay?model=mobilenet",
                headers=get_clinician_auth_header()
            )
            assert res_cam.status_code == 200
            assert res_cam.headers["content-type"] in ["image/jpeg", "image/png"]
            j_clinician["steps"].append({"step": "Grad-CAM Explainability Review (image/jpeg)", "status": "PASS"})

            # B7. Clinician Submits Clinical Review
            review_payload = {
                "comment": "Uniform regular pigment network. Clinical appearance consistent with benign junctional nevus. Routine monitoring recommended.",
                "recommendation": "Routine self-examination; schedule re-evaluation in 6 months if changes occur.",
                "review_status": "reviewed"
            }
            res_review = client.post(
                f"/api/v1/cases/{patient_case_id}/reviews",
                headers=get_clinician_auth_header(),
                json=review_payload
            )
            assert res_review.status_code == 200
            j_clinician["steps"].append({"step": "Clinical Review & Diagnostic Authorization", "status": "PASS"})

            # B8. Schedule Longitudinal Reminder
            res_rem = client.post(
                f"/api/v1/cases/{patient_case_id}/reminders",
                headers=get_clinician_auth_header(),
                json={
                    "scheduled_date": "2027-05-21T09:00:00Z",
                    "priority": "routine",
                    "notes": "6-month routine dermatoscopic follow-up exam."
                }
            )
            assert res_rem.status_code == 200
            j_clinician["steps"].append({"step": "Longitudinal Follow-up Reminder Scheduling", "status": "PASS"})

            # B9. Download Clinical PDF Report
            res_pdf = client.get(
                f"/api/v1/cases/{patient_case_id}/report/pdf",
                headers=get_clinician_auth_header()
            )
            assert res_pdf.status_code == 200
            assert res_pdf.content.startswith(b"%PDF-")
            assert len(res_pdf.content) > 1000
            j_clinician["steps"].append({
                "step": "Clinical PDF Report Generation",
                "status": "PASS",
                "pdf_size_bytes": len(res_pdf.content)
            })

            print("  [PASS] Journey B (Clinician) Completed Successfully.")
            passed_count += 1
        except Exception as e:
            j_clinician["passed"] = False
            j_clinician["error"] = str(e)
            print(f"  [FAIL] Journey B Failed: {e}")
            failed_count += 1
        results["journeys"]["journey_b_clinician"] = j_clinician

        # -------------------------------------------------------------
        # JOURNEY C: Role Security & Registration Boundaries
        # -------------------------------------------------------------
        print("\n--- [Journey C] Role Security & Public Registration Boundaries ---")
        j_rolesec = {"steps": [], "passed": True}
        try:
            # C1. Public Registration Rejects 'admin' Role (HTTP 400 Bad Request)
            res_rej_admin = client.post("/api/v1/auth/register", json={
                "username": f"bad_admin_{ts}", "email": f"bad_adm_{ts}@evil.org",
                "full_name": "Rogue Admin", "password": "Password123!", "role": "admin"
            })
            assert res_rej_admin.status_code == 400
            assert "not permitted" in res_rej_admin.text.lower()
            j_rolesec["steps"].append({"step": "Reject Public Registration for 'admin' (400)", "status": "PASS"})

            # C2. Public Registration Rejects 'researcher' Role (HTTP 400 Bad Request)
            res_rej_res = client.post("/api/v1/auth/register", json={
                "username": f"bad_res_{ts}", "email": f"bad_res_{ts}@evil.org",
                "full_name": "Rogue Researcher", "password": "Password123!", "role": "researcher"
            })
            assert res_rej_res.status_code == 400
            assert "not permitted" in res_rej_res.text.lower()
            j_rolesec["steps"].append({"step": "Reject Public Registration for 'researcher' (400)", "status": "PASS"})

            # C3. Public Registration Rejects 'ml_engineer' Role (HTTP 400 Bad Request)
            res_rej_mle = client.post("/api/v1/auth/register", json={
                "username": f"bad_mle_{ts}", "email": f"bad_mle_{ts}@evil.org",
                "full_name": "Rogue ML Engineer", "password": "Password123!", "role": "ml_engineer"
            })
            assert res_rej_mle.status_code == 400
            assert "not permitted" in res_rej_mle.text.lower()
            j_rolesec["steps"].append({"step": "Reject Public Registration for 'ml_engineer' (400)", "status": "PASS"})

            # C4. Public Registration Rejects Unsupported/Unknown Role (HTTP 400 or 422)
            res_rej_unk = client.post("/api/v1/auth/register", json={
                "username": f"bad_unk_{ts}", "email": f"bad_unk_{ts}@evil.org",
                "full_name": "Rogue Unknown", "password": "Password123!", "role": "superuser"
            })
            assert res_rej_unk.status_code in [400, 422]
            j_rolesec["steps"].append({"step": "Reject Unsupported Role 'superuser' (400/422)", "status": "PASS"})

            # C5. Confirm Only 'patient' and 'clinician' Roles Are Supported for Public Registration
            res_ok_patient = client.post("/api/v1/auth/register", json={
                "username": f"valid_pat_{ts}", "email": f"valid_p_{ts}@hospital.org",
                "full_name": "Valid Patient", "password": "Password123!", "role": "patient"
            })
            assert res_ok_patient.status_code == 200

            res_ok_clinician = client.post("/api/v1/auth/register", json={
                "username": f"valid_doc_{ts}", "email": f"valid_d_{ts}@hospital.org",
                "full_name": "Valid Clinician", "password": "Password123!", "role": "clinician"
            })
            assert res_ok_clinician.status_code == 200
            j_rolesec["steps"].append({"step": "Verified Public Registration Restricted to Patient & Clinician", "status": "PASS"})

            print("  [PASS] Journey C (Role Security & Boundaries) Completed Successfully.")
            passed_count += 1
        except Exception as e:
            j_rolesec["passed"] = False
            j_rolesec["error"] = str(e)
            print(f"  [FAIL] Journey C Failed: {e}")
            failed_count += 1
        results["journeys"]["journey_c_role_security"] = j_rolesec

        # -------------------------------------------------------------
        # JOURNEY E: Role-Based Access Control (RBAC) & Privilege Separation
        # -------------------------------------------------------------
        print("\n--- [Journey E] RBAC & Privilege Separation Enforcement ---")
        j_rbac = {"steps": [], "passed": True}
        try:
            patient_auth = get_patient_auth_header()
            clinician_auth = get_clinician_auth_header()
            admin_auth = get_admin_auth_header()

            # E1. Patient cannot claim cases (HTTP 403 - Clinician/Admin only)
            res_p_claim = client.put(
                f"/api/v1/cases/{patient_case_id}/assign",
                headers=patient_auth,
                json={}
            )
            assert res_p_claim.status_code == 403
            j_rbac["steps"].append({"step": "Patient Blocked from Case Assignment Claiming (403)", "status": "PASS"})

            # E2. Patient cannot submit clinical reviews (HTTP 403 - Clinician/Admin only)
            res_p_review = client.post(
                f"/api/v1/cases/{patient_case_id}/reviews",
                headers=patient_auth,
                json={"comment": "Unauthorized review attempt", "recommendation": "None", "review_status": "reviewed"}
            )
            assert res_p_review.status_code == 403
            j_rbac["steps"].append({"step": "Patient Blocked from Submitting Clinical Reviews (403)", "status": "PASS"})

            # E3. Patient cannot access admin clinician verification (HTTP 403 - Admin only)
            res_p_verify = client.patch(
                "/api/v1/admin/clinicians/1/verify",
                headers=patient_auth
            )
            assert res_p_verify.status_code == 403
            j_rbac["steps"].append({"step": "Patient Blocked from Admin Clinician Verification (403)", "status": "PASS"})

            # E4. Patient cannot access admin pending clinicians list (HTTP 403 - Admin only)
            res_p_pending = client.get(
                "/api/v1/admin/clinicians/pending",
                headers=patient_auth
            )
            assert res_p_pending.status_code == 403
            j_rbac["steps"].append({"step": "Patient Blocked from Admin Pending Clinicians (403)", "status": "PASS"})

            # E5. Clinician cannot access admin clinician approval (HTTP 403 - Admin only)
            res_c_verify = client.patch(
                "/api/v1/admin/clinicians/1/verify",
                headers=clinician_auth
            )
            assert res_c_verify.status_code == 403
            j_rbac["steps"].append({"step": "Clinician Blocked from Admin Verification Endpoints (403)", "status": "PASS"})

            # E6. Clinician cannot access admin pending clinicians list (HTTP 403 - Admin only)
            res_c_pending = client.get(
                "/api/v1/admin/clinicians/pending",
                headers=clinician_auth
            )
            assert res_c_pending.status_code == 403
            j_rbac["steps"].append({"step": "Clinician Blocked from Admin Pending Clinicians List (403)", "status": "PASS"})

            # E7. Non-Admin (Patient & Clinician) cannot switch active model (HTTP 403 - ML Engineer/Admin only)
            res_p_model = client.post(
                "/api/v1/models/active",
                headers=patient_auth,
                json={"target_version": "v1.1", "reason": "Unauthorized model switch"}
            )
            assert res_p_model.status_code == 403

            res_c_model = client.post(
                "/api/v1/models/active",
                headers=clinician_auth,
                json={"target_version": "v1.1", "reason": "Unauthorized model switch"}
            )
            assert res_c_model.status_code == 403
            j_rbac["steps"].append({"step": "Non-Admins Blocked from Model Activation / Rollback (403)", "status": "PASS"})

            # E8. Non-Admins (Patient & Clinician) cannot view training history (HTTP 403)
            res_p_hist = client.get(
                "/api/v1/models/training-history",
                headers=patient_auth
            )
            assert res_p_hist.status_code == 403

            res_c_hist = client.get(
                "/api/v1/models/training-history",
                headers=clinician_auth
            )
            assert res_c_hist.status_code == 403
            j_rbac["steps"].append({"step": "Non-Admins Blocked from Training Lineage Governance (403)", "status": "PASS"})

            # E9. Unauthenticated requests to protected endpoints rejected (HTTP 401)
            res_unauth_cases = client.get("/api/v1/cases/")
            assert res_unauth_cases.status_code == 401

            res_unauth_admin = client.get("/api/v1/admin/clinicians/pending")
            assert res_unauth_admin.status_code == 401
            j_rbac["steps"].append({"step": "Unauthenticated Access Rejection (401)", "status": "PASS"})

            print("  [PASS] Journey E (RBAC Enforcement) Completed Successfully.")
            passed_count += 1
        except Exception as e:
            j_rbac["passed"] = False
            j_rbac["error"] = str(e)
            print(f"  [FAIL] Journey E Failed: {e}")
            failed_count += 1
        results["journeys"]["journey_e_rbac_enforcement"] = j_rbac

        # -------------------------------------------------------------
        # JOURNEY F: Clinical Edge Cases & Security Robustness
        # -------------------------------------------------------------
        print("\n--- [Journey F] Clinical Edge Cases & Security Boundaries ---")
        j_edge = {"steps": [], "passed": True}
        try:
            # F1. Corrupt Upload Rejection & Case Preservation
            res_bad_case = client.post(
                "/api/v1/cases/",
                headers=get_patient_auth_header(),
                json={"body_location": "abdomen", "clinical_notes": "Failure simulation case"}
            )
            bad_case_id = res_bad_case.json()["id"]

            res_bad_upload = client.post(
                f"/api/v1/cases/{bad_case_id}/upload-image",
                headers=get_patient_auth_header(),
                files={"file": ("fake.jpg", b"corrupted non-jpeg byte string", "image/jpeg")}
            )
            assert res_bad_upload.status_code == 400
            # Verify case is preserved in 'created' state
            res_case_check = client.get(f"/api/v1/cases/{bad_case_id}", headers=get_patient_auth_header())
            assert res_case_check.json()["status"] == "created"
            j_edge["steps"].append({"step": "Corrupt Upload Rejection & Case Retention (400)", "status": "PASS"})

            # F2. Retry Upload Without Case Re-creation
            res_retry = client.post(
                f"/api/v1/cases/{bad_case_id}/upload-image",
                headers=get_patient_auth_header(),
                files={"file": ("real.jpg", load_real_image_bytes(), "image/jpeg")}
            )
            assert res_retry.status_code == 200
            assert res_retry.json()["status"] == "preprocessed"
            j_edge["steps"].append({"step": "Upload Retry Without Case Recreation (200)", "status": "PASS"})

            # F3. Cross-Patient BOLA/IDOR Access Barrier
            res_other_p = client.post("/api/v1/auth/register", json={
                "username": f"patient_beta_{ts}", "email": f"beta_{ts}@e2e.org",
                "full_name": "Patient Beta", "password": "Password123!", "role": "patient"
            })
            token_beta = res_other_p.json()["access_token"]
            res_bola = client.get(f"/api/v1/cases/{patient_case_id}", headers={"Authorization": f"Bearer {token_beta}"})
            assert res_bola.status_code == 403
            j_edge["steps"].append({"step": "Cross-Patient BOLA/IDOR Access Rejection (403)", "status": "PASS"})

            # F4. Post-Review Patient Edit Freeze
            # Attempting to edit patient_case_id (already reviewed in Journey B)
            res_freeze = client.put(
                f"/api/v1/cases/{patient_case_id}",
                headers=get_patient_auth_header(),
                json={"clinical_notes": "Attempting unauthorized post-review note alteration"}
            )
            assert res_freeze.status_code == 403
            assert "frozen" in res_freeze.text.lower()
            j_edge["steps"].append({"step": "Post-Review Metadata Freezing Enforcement (403)", "status": "PASS"})

            print("  [PASS] Journey F (Edge Cases & Security) Completed Successfully.")
            passed_count += 1
        except Exception as e:
            j_edge["passed"] = False
            j_edge["error"] = str(e)
            print(f"  [FAIL] Journey F Failed: {e}")
            failed_count += 1
        results["journeys"]["journey_f_edge_cases"] = j_edge

        # Summary
        total_journeys = passed_count + failed_count
        pass_rate = round((passed_count / total_journeys) * 100.0, 2)
        results["summary"] = {
            "total_journeys": total_journeys,
            "journeys_passed": passed_count,
            "journeys_failed": failed_count,
            "pass_rate_pct": pass_rate,
            "overall_status": "PASS" if failed_count == 0 else "FAIL"
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 80)
        print(f"END-TO-END WORKFLOW TESTING COMPLETE: {passed_count}/{total_journeys} Journeys Passed ({pass_rate}%)")
        print(f"Results written to: {OUTPUT_JSON}")
        print("=" * 80)

        assert failed_count == 0, f"End-to-End Test Suite had {failed_count} failures."

    finally:
        # 1. Tear down isolated test database fixture
        app.dependency_overrides.clear()
        test_engine.dispose()
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except Exception:
                pass

        # 2. Restore pristine model registry content if modified during test
        if original_registry_content and os.path.isfile(REGISTRY_PATH):
            try:
                with open(REGISTRY_PATH, "w", encoding="utf-8") as wf:
                    wf.write(original_registry_content)
            except Exception:
                pass

        # 3. Clean up test image uploads from backend/uploads
        upload_base = os.path.join(BASE_DIR, "backend", "uploads")
        if os.path.isdir(upload_base):
            for root, dirs, files in os.walk(upload_base):
                for file in files:
                    if file != ".gitkeep":
                        try:
                            os.remove(os.path.join(root, file))
                        except Exception:
                            pass


def test_e2e_workflows():
    run_e2e_tests()


if __name__ == "__main__":
    run_e2e_tests()
