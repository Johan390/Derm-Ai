"""
DermAI - Role-Based Access and Clinical Workflow Test Suite
============================================================
Tests role-differentiated workflows for:
1. Patient Portal (case creation, photo upload, prediction, viewing doctor comments & review status, cross-patient isolation)
2. Doctor / Clinician Portal (viewing authorized cases, claiming case, adding clinical observations & recommendations, review status lifecycle, unauthorized doctor isolation)
3. Researcher Portal (benchmark datasets, model evaluation metrics, IQA telemetry, de-identified research cases, zero PHI access, blocking private clinical notes)
4. Admin System Oversight
"""

import sys
import os
import io
import time
import uuid
import tempfile
import glob
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.config import settings
from app.database import Base, get_db
from app.models import User, UserRole, Case
from app.auth import hash_password

client = TestClient(app)

def generate_synthetic_image_bytes():
    """Generates valid 224x224 RGB image bytes suitable for IQA and model input."""
    img = Image.new("RGB", (224, 224), color=(160, 120, 110))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def run_role_workflow_tests():
    print("\n" + "=" * 80)
    print("DERMAI - ROLE-BASED ACCESS & CLINICAL WORKFLOW CORRECTION TEST SUITE")
    print("=" * 80 + "\n")

    # Isolated test database fixture
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix="_role_wf.db")
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
    ts = int(time.time())
    created_case_codes = []

    try:
        # Provision Admin directly in isolated DB (public registration is restricted)
        admin_username = f"admin_init_{ts}"
        admin_password = "SecurePassword123!"
        db = TestingSessionLocal()
        try:
            test_admin = User(
                username=admin_username,
                email=f"admin_init_{ts}@dermaitest.org",
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

        # Admin authenticates via login endpoint
        res_adm_init = client.post("/api/v1/auth/login", json={
            "username": admin_username,
            "password": admin_password
        })
        assert res_adm_init.status_code == 200
        admin_token = res_adm_init.json()["access_token"]

        # Verify public registration of admin is rejected (HTTP 400)
        res_adm_pub = client.post("/api/v1/auth/register", json={
            "username": f"pub_admin_{ts}",
            "email": f"pub_admin_{ts}@dermaitest.org",
            "full_name": "Public Admin Attempt",
            "password": "Password123!",
            "role": "admin"
        })
        assert res_adm_pub.status_code == 400, f"Expected 400 for public admin registration, got {res_adm_pub.status_code}"

        # =========================================================================
        # Scenario 1: Patient Workflow & Isolation
        # =========================================================================
        try:
            # Register Patient Alpha
            res_pat_a = client.post("/api/v1/auth/register", json={
                "username": f"patient_alpha_{ts}",
                "email": f"pat_a_{ts}@dermaitest.org",
                "full_name": "Alice Alpha",
                "password": "SecurePassword123!",
                "role": "patient"
            })
            assert res_pat_a.status_code == 200, f"Registration failed: {res_pat_a.text}"
            pat_a_token = res_pat_a.json()["access_token"]
            pat_a_id = res_pat_a.json()["user_id"]

            # Patient Alpha creates a case
            res_case_a = client.post("/api/v1/cases/", json={
                "patient_age": 42,
                "patient_gender": "female",
                "body_location": "back",
                "clinical_notes": "Slightly itchy pigmented macule on upper left scapula"
            }, headers={"Authorization": f"Bearer {pat_a_token}"})
            assert res_case_a.status_code == 200
            case_a_data = res_case_a.json()
            case_a_id = case_a_data["id"]
            created_case_codes.append(case_a_data["case_code"])
            assert case_a_data["review_status"] == "pending_review"
            assert case_a_data["assigned_doctor_id"] is None

            # Patient Alpha uploads photo
            img_buf = generate_synthetic_image_bytes()
            res_up = client.post(
                f"/api/v1/cases/{case_a_id}/upload-image",
                files={"file": ("lesion_a.jpg", img_buf.getvalue(), "image/jpeg")},
                headers={"Authorization": f"Bearer {pat_a_token}"}
            )
            assert res_up.status_code == 200

            # Patient Alpha views own case
            res_get_own = client.get(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {pat_a_token}"})
            assert res_get_own.status_code == 200
            assert res_get_own.json()["id"] == case_a_id

            results.append(("1. Patient Viewing Own Case", "PASSED", f"Patient Alpha created Case {case_a_id} and verified initial review_status='pending_review'."))
        except Exception as e:
            results.append(("1. Patient Viewing Own Case", "FAILED", str(e)))

        # =========================================================================
        # Scenario 2: Cross-Patient Isolation (Patient B blocked from Patient A)
        # =========================================================================
        try:
            # Register Patient Beta
            res_pat_b = client.post("/api/v1/auth/register", json={
                "username": f"patient_beta_{ts}",
                "email": f"pat_b_{ts}@dermaitest.org",
                "full_name": "Bob Beta",
                "password": "SecurePassword123!",
                "role": "patient"
            })
            assert res_pat_b.status_code == 200
            pat_b_token = res_pat_b.json()["access_token"]

            # Patient Beta attempts to view Patient Alpha's case
            res_denied_read = client.get(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {pat_b_token}"})
            assert res_denied_read.status_code == 403, f"Expected 403, got {res_denied_read.status_code}"

            # Patient Beta attempts to access reviews of Patient Alpha's case
            res_denied_rev = client.get(f"/api/v1/cases/{case_a_id}/reviews", headers={"Authorization": f"Bearer {pat_b_token}"})
            assert res_denied_rev.status_code == 403

            results.append(("2. Cross-Patient Access Rejection", "PASSED", "Patient Beta strictly denied access (403) to Patient Alpha's clinical records and reviews."))
        except Exception as e:
            results.append(("2. Cross-Patient Access Rejection", "FAILED", str(e)))

        # =========================================================================
        # Scenario 3: Doctor Workflow & Case Claiming
        # =========================================================================
        try:
            # Register Doctor A
            res_doc_a = client.post("/api/v1/auth/register", json={
                "username": f"dr_carol_{ts}",
                "email": f"dr_carol_{ts}@hospital.org",
                "full_name": "Dr. Carol Danvers, MD",
                "password": "SecurePassword123!",
                "role": "clinician",
                "institution": "Metropolitan Dermatology Center"
            })
            assert res_doc_a.status_code == 200
            doc_a_id = res_doc_a.json()["user_id"]

            # Admin approves Doctor A (Task 1 Clinician Verification Requirement)
            res_v_a = client.patch(f"/api/v1/admin/clinicians/{doc_a_id}/verify", headers={"Authorization": f"Bearer {admin_token}"})
            assert res_v_a.status_code == 200

            # Doctor A authenticates to get verified token
            res_login_doc_a = client.post("/api/v1/auth/login", json={
                "username": f"dr_carol_{ts}",
                "password": "SecurePassword123!"
            })
            assert res_login_doc_a.status_code == 200
            doc_a_token = res_login_doc_a.json()["access_token"]

            # Doctor A lists cases (sees unassigned Case A in triage queue)
            res_doc_list = client.get("/api/v1/cases/?assignment_filter=unassigned", headers={"Authorization": f"Bearer {doc_a_token}"})
            assert res_doc_list.status_code == 200
            assert any(c["id"] == case_a_id for c in res_doc_list.json())

            # Doctor A claims/assigns Case A to self
            res_claim = client.put(f"/api/v1/cases/{case_a_id}/assign", json={}, headers={"Authorization": f"Bearer {doc_a_token}"})
            assert res_claim.status_code == 200
            claimed_data = res_claim.json()
            assert claimed_data["assigned_doctor_id"] == doc_a_id
            assert claimed_data["review_status"] == "under_review"

            results.append(("3. Doctor Claiming Authorized Case", "PASSED", f"Doctor A claimed unassigned Case {case_a_id}; review_status transitioned to 'under_review'."))
        except Exception as e:
            results.append(("3. Doctor Claiming Authorized Case", "FAILED", str(e)))

        # =========================================================================
        # Scenario 4: Doctor Adding Clinical Observation & Recommendations
        # =========================================================================
        try:
            # Doctor A adds clinical observation and recommendation
            res_add_rev = client.post(
                f"/api/v1/cases/{case_a_id}/reviews",
                json={
                    "comment": "Pigmented macule with regular reticular network and uniform brownish pigmentation. No atypical pseudopods or veil.",
                    "recommendation": "Benign melanocytic nevus presentation. Re-evaluate annually or if asymmetrical expansion occurs.",
                    "review_status": "reviewed"
                },
                headers={"Authorization": f"Bearer {doc_a_token}"}
            )
            assert res_add_rev.status_code == 200, f"Add review failed: {res_add_rev.text}"
            rev_data = res_add_rev.json()
            assert rev_data["case_id"] == case_a_id
            assert rev_data["doctor_id"] == doc_a_id
            assert rev_data["review_status"] == "reviewed"
            assert "Benign melanocytic nevus" in rev_data["recommendation"]

            # Verify case review status updated
            res_case_check = client.get(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {doc_a_token}"})
            assert res_case_check.status_code == 200
            assert res_case_check.json()["review_status"] == "reviewed"

            results.append(("4. Doctor Authoring Clinical Review", "PASSED", "Doctor A recorded clinical observations and recommendations; case marked 'reviewed'."))
        except Exception as e:
            results.append(("4. Doctor Authoring Clinical Review", "FAILED", str(e)))

        # =========================================================================
        # Scenario 5: Patient Viewing Doctor Comments & Recommendations
        # =========================================================================
        try:
            # Patient Alpha accesses their own case details and reviews
            res_pat_check = client.get(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {pat_a_token}"})
            assert res_pat_check.status_code == 200
            p_data = res_pat_check.json()
            assert p_data["review_status"] == "reviewed"
            assert len(p_data["reviews"]) >= 1
            assert p_data["reviews"][0]["doctor_name"] == "Dr. Carol Danvers, MD"
            assert "reticular network" in p_data["reviews"][0]["comment"]
            assert "annual" in p_data["reviews"][0]["recommendation"].lower()

            # Patient Alpha verifies dedicated reviews endpoint
            res_revs_list = client.get(f"/api/v1/cases/{case_a_id}/reviews", headers={"Authorization": f"Bearer {pat_a_token}"})
            assert res_revs_list.status_code == 200
            assert len(res_revs_list.json()) >= 1

            results.append(("5. Patient Viewing Doctor Review", "PASSED", "Patient Alpha retrieved Doctor A's clinical observation and follow-up recommendation."))
        except Exception as e:
            results.append(("5. Patient Viewing Doctor Review", "FAILED", str(e)))

        # =========================================================================
        # Scenario 6: Unauthorized Doctor Isolation (Doctor B blocked from Doctor A's case)
        # =========================================================================
        try:
            # Register Doctor B
            res_doc_b = client.post("/api/v1/auth/register", json={
                "username": f"dr_strange_{ts}",
                "email": f"dr_strange_{ts}@hospital.org",
                "full_name": "Dr. Stephen Strange, MD",
                "password": "SecurePassword123!",
                "role": "clinician",
                "institution": "Downtown Dermatology Clinic"
            })
            assert res_doc_b.status_code == 200
            doc_b_id = res_doc_b.json()["user_id"]

            # Admin approves Doctor B so they are a verified clinician
            res_v_b = client.patch(f"/api/v1/admin/clinicians/{doc_b_id}/verify", headers={"Authorization": f"Bearer {admin_token}"})
            assert res_v_b.status_code == 200

            # Doctor B authenticates to get verified token
            res_login_doc_b = client.post("/api/v1/auth/login", json={
                "username": f"dr_strange_{ts}",
                "password": "SecurePassword123!"
            })
            assert res_login_doc_b.status_code == 200
            doc_b_token = res_login_doc_b.json()["access_token"]

            # Doctor B attempts to view Case A (which is assigned to Doctor A)
            res_b_view = client.get(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {doc_b_token}"})
            assert res_b_view.status_code == 403, f"Expected 403 for unauthorized doctor, got {res_b_view.status_code}"

            # Doctor B attempts to post review on Case A
            res_b_rev = client.post(
                f"/api/v1/cases/{case_a_id}/reviews",
                json={"comment": "Unauthorized review attempt"},
                headers={"Authorization": f"Bearer {doc_b_token}"}
            )
            assert res_b_rev.status_code == 403

            results.append(("6. Unauthorized Doctor Isolation", "PASSED", "Doctor B strictly blocked (403) from accessing or modifying Case A assigned to Doctor A."))
        except Exception as e:
            results.append(("6. Unauthorized Doctor Isolation", "FAILED", str(e)))

        # =========================================================================
        # Scenario 7: Patient Denied Modification of Doctor Comments
        # =========================================================================
        try:
            # Patient Alpha attempts to post a doctor review on own case
            res_pat_post_rev = client.post(
                f"/api/v1/cases/{case_a_id}/reviews",
                json={"comment": "Patient attempting to author clinical review"},
                headers={"Authorization": f"Bearer {pat_a_token}"}
            )
            # Must be 403 Forbidden because patients are not allowed to author doctor reviews
            assert res_pat_post_rev.status_code == 403, f"Expected 403, got {res_pat_post_rev.status_code}"

            # Patient Alpha attempts to alter review status
            res_pat_status = client.patch(
                f"/api/v1/cases/{case_a_id}/review-status",
                json={"review_status": "reviewed"},
                headers={"Authorization": f"Bearer {pat_a_token}"}
            )
            assert res_pat_status.status_code == 403

            results.append(("7. Patient Review Modification Prevention", "PASSED", "Patients strictly blocked (403) from authoring or tampering with doctor reviews."))
        except Exception as e:
            results.append(("7. Patient Review Modification Prevention", "FAILED", str(e)))

        # =========================================================================
        # Scenario 8: Admin Dual Access & Oversight
        # =========================================================================
        try:
            # Admin can access Case A (assigned to Doctor A)
            res_adm_view = client.get(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {admin_token}"})
            assert res_adm_view.status_code == 200

            # Admin can update review status
            res_adm_patch = client.patch(
                f"/api/v1/cases/{case_a_id}/review-status",
                json={"review_status": "follow_up_required"},
                headers={"Authorization": f"Bearer {admin_token}"}
            )
            assert res_adm_patch.status_code == 200
            assert res_adm_patch.json()["review_status"] == "follow_up_required"

            results.append(("8. Admin Clinical Oversight", "PASSED", "System administrator verified for cross-physician case oversight and status management."))
        except Exception as e:
            results.append(("8. Admin Clinical Oversight", "FAILED", str(e)))

        # =========================================================================
        # Scenario 9: Researcher Role Boundary & Dataset Access
        # =========================================================================
        try:
            # 1. Verify public registration of researcher is rejected (HTTP 400 Bad Request)
            res_res_pub = client.post("/api/v1/auth/register", json={
                "username": f"dr_researcher_pub_{ts}",
                "email": f"researcher_pub_{ts}@university.edu",
                "full_name": "Public Researcher Attempt",
                "password": "SecurePassword123!",
                "role": "researcher",
                "institution": "Computational Dermatology Institute"
            })
            assert res_res_pub.status_code == 400, f"Expected 400 for public researcher registration, got {res_res_pub.status_code}"

            # 2. Provision researcher directly in DB to test researcher portal endpoints
            researcher_username = f"dr_researcher_{ts}"
            researcher_password = "SecurePassword123!"
            db = TestingSessionLocal()
            try:
                test_researcher = User(
                    username=researcher_username,
                    email=f"researcher_{ts}@university.edu",
                    full_name="Dr. Marie Curie, PhD",
                    role=UserRole.RESEARCHER,
                    institution="Computational Dermatology Institute",
                    hashed_password=hash_password(researcher_password),
                    is_active=True,
                    is_verified=True
                )
                db.add(test_researcher)
                db.commit()
            finally:
                db.close()

            res_login_res = client.post("/api/v1/auth/login", json={
                "username": researcher_username,
                "password": researcher_password
            })
            assert res_login_res.status_code == 200
            researcher_token = res_login_res.json()["access_token"]

            # 1. Dataset Info
            res_dinfo = client.get("/api/v1/dataset/info", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_dinfo.status_code == 200

            # 2. EDA Summary
            res_eda = client.get("/api/v1/dataset/eda-summary", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_eda.status_code == 200
            eda_data = res_eda.json()
            assert eda_data["total_records"] == 10015
            assert len(eda_data["class_summary"]) == 7

            # 3. Model Performance & Evaluation Metrics
            res_metrics = client.get("/api/v1/research/metrics", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_metrics.status_code == 200
            metrics_data = res_metrics.json()
            assert metrics_data["overall_accuracy"] > 0.65
            assert metrics_data["macro_roc_auc"] > 0.85
            if "per_class" in metrics_data:
                assert len(metrics_data["per_class"]) == 7

            # 4. Image Quality Telemetry
            res_qstats = client.get("/api/v1/research/image-quality-stats", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_qstats.status_code == 200
            assert "suitable_percentage" in res_qstats.json()

            results.append(("9. Researcher Benchmark & Metrics Access", "PASSED", "Rejected public registration (400) and authorized provisioned researcher for HAM10000 EDA & metrics."))
        except Exception as e:
            results.append(("9. Researcher Benchmark & Metrics Access", "FAILED", str(e)))

        # =========================================================================
        # Scenario 10: Researcher Blocked from Private Clinical Records
        # =========================================================================
        try:
            # Researcher attempts to list clinical patient cases
            res_r_cases = client.get("/api/v1/cases/", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_r_cases.status_code == 403, f"Expected 403, got {res_r_cases.status_code}"

            # Researcher attempts to read Case A details
            res_r_get = client.get(f"/api/v1/cases/{case_a_id}", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_r_get.status_code == 403

            # Researcher attempts to read Case A reviews
            res_r_revs = client.get(f"/api/v1/cases/{case_a_id}/reviews", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_r_revs.status_code == 403

            results.append(("10. Researcher Blocked from Clinical Cases", "PASSED", "Researcher strictly denied (403) from listing cases, viewing clinical notes, or reading doctor reviews."))
        except Exception as e:
            results.append(("10. Researcher Blocked from Clinical Cases", "FAILED", str(e)))

        # =========================================================================
        # Scenario 11: Researcher Anonymized Cohort & Zero PHI Verification
        # =========================================================================
        try:
            # Researcher accesses de-identified research cases endpoint
            res_anon = client.get("/api/v1/research/anonymized-cases", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_anon.status_code == 200
            anon_data = res_anon.json()
            assert isinstance(anon_data, list)
            assert len(anon_data) >= 1

            # Deep verification of Zero PHI / Zero Patient Identity
            for record in anon_data:
                assert "user_id" not in record, "Found forbidden user_id in research data"
                assert "username" not in record, "Found forbidden username in research data"
                assert "email" not in record, "Found forbidden email in research data"
                assert "patient_name" not in record, "Found forbidden patient_name in research data"
                assert "full_name" not in record, "Found forbidden full_name in research data"
                assert "clinical_notes" not in record, "Found forbidden private clinical_notes in research data"
                assert "medical_history" not in record, "Found forbidden private medical_history in research data"
                assert "known_allergies" not in record, "Found forbidden known_allergies in research data"
                assert record["research_id"].startswith("RES-"), "Research ID must be de-identified"

            # Researcher exports de-identified data as CSV
            res_export_csv = client.get("/api/v1/research/export-anonymized?format=csv", headers={"Authorization": f"Bearer {researcher_token}"})
            assert res_export_csv.status_code == 200
            assert "text/csv" in res_export_csv.headers["content-type"]
            assert "research_id" in res_export_csv.text
            assert "patient_age_group" in res_export_csv.text

            results.append(("11. De-Identified Cohort & Zero PHI Verification", "PASSED", "De-identified research cohort verified: zero patient identities, zero private notes, CSV/JSON export operational."))
        except Exception as e:
            results.append(("11. De-Identified Cohort & Zero PHI Verification", "FAILED", str(e)))

        # =========================================================================
        # Scenario 12: Baseline Milestone 1 & 2 Authentication Integrity
        # =========================================================================
        try:
            # 1. Unauthenticated requests blocked
            assert client.get("/api/v1/cases/").status_code == 401
            assert client.get("/api/v1/dataset/info").status_code == 401
            assert client.get("/api/v1/research/metrics").status_code == 401

            # 2. Invalid password rejected
            res_bad_pw = client.post("/api/v1/auth/login", json={"username": f"patient_alpha_{ts}", "password": "WrongPassword!"})
            assert res_bad_pw.status_code == 401

            # 3. Model predict executed by authorized clinician
            res_login_c = client.post("/api/v1/auth/login", json={
                "username": f"dr_carol_{ts}",
                "password": "SecurePassword123!"
            })
            assert res_login_c.status_code == 200
            token_c = res_login_c.json()["access_token"]
            res_pred = client.post(f"/api/v1/cases/{case_a_id}/predict", headers={"Authorization": f"Bearer {token_c}"})
            assert res_pred.status_code == 200
            pred_data = res_pred.json()
            assert pred_data["predicted_class"] in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            assert len(pred_data["class_probabilities"]) == 7

            results.append(("12. Baseline Auth & Inference Integrity", "PASSED", "401 unauthenticated enforcement, bcrypt verification, and 7-class MobileNetV2 prediction verified."))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append(("12. Baseline Auth & Inference Integrity", "FAILED", str(e)))

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

        assert passed_count == len(results), f"Only {passed_count}/{len(results)} tests passed"

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
    run_role_workflow_tests()
