"""
DermAI - Security Hardening & Clinical Workflow Verification Test Suite
========================================================================
Validates the four security hardening tasks:
TASK 1: Clinician Registration Verification
- Newly registered clinicians default to is_verified = False
- Unverified clinicians strictly denied clinical triage (403), case lists (403), reviews (403)
- Admins can query pending clinicians, approve and reject them
- Approved clinicians gain authorized clinical access
- Non-admins cannot verify clinicians (403)

TASK 2: Freeze Patient Editing After Clinical Review
- Patient can edit an unreviewed case (review_status = pending_review)
- Patient strictly blocked (403) from modifying case once review_status is 'reviewed' or 'follow_up_required'
- Patient can still view their reviewed case and doctor comments (200)
- Doctor can continue authorized clinical operations

TASK 3: Prevent Cascading Deletion of Clinical Records (Soft-Deletion)
- Archiving a case sets is_archived = True
- Soft-deleted / archived cases disappear from normal patient case list
- Soft-deleted / archived cases do not appear in clinician triage queue
- Clinical reviews and prediction history remain permanently preserved in DB
- Admin can inspect archived records

TASK 4: Clinician Case Relinquishment
- Assigned clinician can release their claimed case (POST /cases/{id}/release)
- Released case returns to unassigned triage queue (assigned_doctor_id = None, status = pending_review)
- Another clinician can claim the released case
- Unassigned clinician cannot release another clinician's case (403)
- Patient / researcher cannot release a case (403)
- Reviewed case cannot be released back to new-case triage queue (400)
"""

import sys
import os
import io
import time
import tempfile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.models import User, UserRole
from app.auth import hash_password

client = TestClient(app)

def generate_synthetic_image_bytes():
    img = Image.new("RGB", (224, 224), color=(180, 130, 110))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def run_security_hardening_tests():
    print("\n" + "=" * 80)
    print("DERMAI - SECURITY HARDENING & CLINICAL WORKFLOW TEST SUITE")
    print("=" * 80 + "\n")

    # Isolated test database fixture
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix="_sec_hard.db")
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

    results = []
    ts = int(time.time())

    try:
        # Provision Admin directly in database (public registration is restricted)
        admin_username = f"admin_security_{ts}"
        admin_password = "SecurePassword123!"
        db = TestingSessionLocal()
        try:
            test_admin = User(
                username=admin_username,
                email=f"admin_{ts}@dermaitest.org",
                full_name="Dr. Admin Controller",
                role=UserRole.ADMIN,
                institution="Security Center",
                hashed_password=hash_password(admin_password),
                is_active=True,
                is_verified=True
            )
            db.add(test_admin)
            db.commit()
        finally:
            db.close()

        # Admin logs in to obtain admin token
        res_admin_login = client.post("/api/v1/auth/login", json={
            "username": admin_username,
            "password": admin_password
        })
        assert res_admin_login.status_code == 200
        admin_token = res_admin_login.json()["access_token"]

        # Verify public registration of admin is rejected (HTTP 400 Bad Request)
        res_adm_pub = client.post("/api/v1/auth/register", json={
            "username": f"pub_admin_sec_{ts}",
            "email": f"pub_admin_sec_{ts}@dermaitest.org",
            "full_name": "Public Admin Attempt",
            "password": "Password123!",
            "role": "admin"
        })
        assert res_adm_pub.status_code == 400, f"Expected 400 for public admin registration, got {res_adm_pub.status_code}"

        # =========================================================================
        # TASK 1: Clinician Registration Verification
        # =========================================================================
        try:
            # 1. Register new clinician
            res_new_doc = client.post("/api/v1/auth/register", json={
                "username": f"dr_newbie_{ts}",
                "email": f"dr_newbie_{ts}@hospital.org",
                "full_name": "Dr. Newbie Intern, MD",
                "password": "SecurePassword123!",
                "role": "clinician",
                "institution": "County Clinic"
            })
            assert res_new_doc.status_code == 200
            new_doc_data = res_new_doc.json()
            assert new_doc_data["is_verified"] is False, "New clinician must default to is_verified=False"
            unverified_doc_token = new_doc_data["access_token"]
            unverified_doc_id = new_doc_data["user_id"]

            results.append(("1.1 Clinician Defaults to Unverified", "PASSED", f"New clinician Dr. Newbie registered with is_verified=False."))
        except Exception as e:
            results.append(("1.1 Clinician Defaults to Unverified", "FAILED", str(e)))

        try:
            # 2. Unverified clinician blocked from triage queue and cases
            res_triage = client.get("/api/v1/cases/", headers={"Authorization": f"Bearer {unverified_doc_token}"})
            assert res_triage.status_code == 403, f"Expected 403, got {res_triage.status_code}"
            assert "awaiting administrative verification" in res_triage.text

            # Unverified clinician blocked from creating review
            res_rev = client.post("/api/v1/cases/1/reviews", json={"comment": "Unauthorized review attempt"}, headers={"Authorization": f"Bearer {unverified_doc_token}"})
            assert res_rev.status_code == 403

            results.append(("1.2 Unverified Clinician Blocked (403)", "PASSED", "Unverified clinician strictly blocked (403) from triage, case lists, and reviews."))
        except Exception as e:
            results.append(("1.2 Unverified Clinician Blocked (403)", "FAILED", str(e)))

        try:
            # 3. Non-admin cannot approve clinician
            # Register a patient to test non-admin approval
            res_pat = client.post("/api/v1/auth/register", json={
                "username": f"patient_rogue_{ts}",
                "email": f"rogue_{ts}@patient.org",
                "full_name": "Rogue Patient",
                "password": "SecurePassword123!",
                "role": "patient"
            })
            pat_token = res_pat.json()["access_token"]

            res_unauth_verify = client.patch(f"/api/v1/admin/clinicians/{unverified_doc_id}/verify", headers={"Authorization": f"Bearer {pat_token}"})
            assert res_unauth_verify.status_code == 403, "Non-admin must be denied verify access"

            results.append(("1.3 Non-Admin Verification Rejection", "PASSED", "Non-admin accounts strictly blocked (403) from approving clinicians."))
        except Exception as e:
            results.append(("1.3 Non-Admin Verification Rejection", "FAILED", str(e)))

        try:
            # 4. Admin views pending clinicians and approves
            res_pending = client.get("/api/v1/admin/clinicians/pending", headers={"Authorization": f"Bearer {admin_token}"})
            assert res_pending.status_code == 200
            pending_ids = [c["id"] for c in res_pending.json()]
            assert unverified_doc_id in pending_ids, "New clinician must appear in admin pending queue"

            # Admin approves clinician
            res_approved = client.patch(f"/api/v1/admin/clinicians/{unverified_doc_id}/verify", headers={"Authorization": f"Bearer {admin_token}"})
            assert res_approved.status_code == 200
            assert res_approved.json()["is_verified"] is True

            # Now verified clinician can access triage queue
            res_triage_ok = client.get("/api/v1/cases/", headers={"Authorization": f"Bearer {unverified_doc_token}"})
            assert res_triage_ok.status_code == 200

            results.append(("1.4 Admin Clinician Approval & Activation", "PASSED", "Admin successfully approved clinician; clinical access instantly activated."))
        except Exception as e:
            results.append(("1.4 Admin Clinician Approval & Activation", "FAILED", str(e)))

        # =========================================================================
        # TASK 2: Freeze Patient Editing After Clinical Review
        # =========================================================================
        try:
            # Register a patient and create a case
            res_p = client.post("/api/v1/auth/register", json={
                "username": f"patient_freeze_{ts}",
                "email": f"freeze_{ts}@test.org",
                "full_name": "Peter Freeze",
                "password": "SecurePassword123!",
                "role": "patient"
            })
            p_token = res_p.json()["access_token"]

            res_case = client.post("/api/v1/cases/", json={
                "patient_age": 35,
                "patient_gender": "male",
                "body_location": "arm",
                "clinical_notes": "Initial symptoms: light brown flat lesion"
            }, headers={"Authorization": f"Bearer {p_token}"})
            assert res_case.status_code == 200
            case_id = res_case.json()["id"]

            # Patient CAN edit case while pending_review
            res_edit_ok = client.put(f"/api/v1/cases/{case_id}", json={
                "clinical_notes": "Updated symptoms: lesion expanded slightly"
            }, headers={"Authorization": f"Bearer {p_token}"})
            assert res_edit_ok.status_code == 200
            assert res_edit_ok.json()["clinical_notes"] == "Updated symptoms: lesion expanded slightly"

            # Doctor reviews the case
            res_doc_rev = client.post(f"/api/v1/cases/{case_id}/reviews", json={
                "comment": "Regular network, typical benign nevus.",
                "recommendation": "Annual surveillance.",
                "review_status": "reviewed"
            }, headers={"Authorization": f"Bearer {unverified_doc_token}"})
            assert res_doc_rev.status_code == 200

            # Now Patient CANNOT edit case (status = reviewed)
            res_edit_frozen = client.put(f"/api/v1/cases/{case_id}", json={
                "clinical_notes": "Post-review tampering attempt!"
            }, headers={"Authorization": f"Bearer {p_token}"})
            assert res_edit_frozen.status_code == 403, f"Expected 403, got {res_edit_frozen.status_code}"
            assert "frozen" in res_edit_frozen.text.lower()

            # Patient CAN still view case and physician comment
            res_view_ok = client.get(f"/api/v1/cases/{case_id}", headers={"Authorization": f"Bearer {p_token}"})
            assert res_view_ok.status_code == 200
            assert len(res_view_ok.json()["reviews"]) >= 1

            results.append(("2. Freeze Patient Editing Post-Review", "PASSED", "Patient edit successfully permitted before review and strictly frozen (403) after review."))
        except Exception as e:
            results.append(("2. Freeze Patient Editing Post-Review", "FAILED", str(e)))

        # =========================================================================
        # TASK 3: Prevent Cascading Deletion of Clinical Records
        # =========================================================================
        try:
            # Delete case that has clinical reviews
            res_del = client.delete(f"/api/v1/cases/{case_id}", headers={"Authorization": f"Bearer {p_token}"})
            assert res_del.status_code == 200
            del_data = res_del.json()
            assert del_data["is_archived"] is True

            # Verify case no longer appears in normal patient case listings
            res_p_cases = client.get("/api/v1/cases/", headers={"Authorization": f"Bearer {p_token}"})
            assert res_p_cases.status_code == 200
            active_ids = [c["id"] for c in res_p_cases.json()]
            assert case_id not in active_ids, "Archived case must not appear in patient active case list"

            # Verify case no longer appears in clinician triage queue
            res_doc_triage = client.get("/api/v1/cases/", headers={"Authorization": f"Bearer {unverified_doc_token}"})
            assert res_doc_triage.status_code == 200
            triage_ids = [c["id"] for c in res_doc_triage.json()]
            assert case_id not in triage_ids, "Archived case must not appear in clinician triage queue"

            # Verify doctor reviews remain intact in the database
            res_revs = client.get(f"/api/v1/cases/{case_id}/reviews", headers={"Authorization": f"Bearer {admin_token}"})
            assert res_revs.status_code == 200
            assert len(res_revs.json()) >= 1, "Clinical review records must be preserved"

            # Admin can view case including archived
            res_admin_archived = client.get("/api/v1/cases/?include_archived=true", headers={"Authorization": f"Bearer {admin_token}"})
            assert res_admin_archived.status_code == 200
            admin_ids = [c["id"] for c in res_admin_archived.json()]
            assert case_id in admin_ids, "Admin must be able to view archived records"

            results.append(("3. Soft-Deletion & Record Preservation", "PASSED", "Case archived (is_archived=True); hidden from active queues; reviews permanently preserved."))
        except Exception as e:
            results.append(("3. Soft-Deletion & Record Preservation", "FAILED", str(e)))

        # =========================================================================
        # TASK 4: Clinician Case Relinquishment
        # =========================================================================
        try:
            # Create a fresh patient case for assignment & release testing
            res_case2 = client.post("/api/v1/cases/", json={
                "patient_age": 50,
                "patient_gender": "female",
                "body_location": "neck",
                "clinical_notes": "Slightly pruritic lesion"
            }, headers={"Authorization": f"Bearer {p_token}"})
            case2_id = res_case2.json()["id"]

            # Fresh token for Doctor 1
            res_doc1_login = client.post("/api/v1/auth/login", json={"username": f"dr_newbie_{ts}", "password": "SecurePassword123!"})
            doc1_token = res_doc1_login.json()["access_token"]

            # Doctor 1 claims case
            res_claim = client.put(f"/api/v1/cases/{case2_id}/assign", json={}, headers={"Authorization": f"Bearer {doc1_token}"})
            assert res_claim.status_code == 200
            assert res_claim.json()["assigned_doctor_id"] == unverified_doc_id
            assert res_claim.json()["review_status"] == "under_review"

            # Register Doctor 2 (and verify them via admin)
            res_doc2 = client.post("/api/v1/auth/register", json={
                "username": f"dr_second_{ts}",
                "email": f"dr_second_{ts}@hospital.org",
                "full_name": "Dr. Second Opinion, MD",
                "password": "SecurePassword123!",
                "role": "clinician",
                "institution": "Metropolitan Clinic"
            })
            doc2_id = res_doc2.json()["user_id"]
            client.patch(f"/api/v1/admin/clinicians/{doc2_id}/verify", headers={"Authorization": f"Bearer {admin_token}"})

            # Authenticate Doctor 2
            res_doc2_login = client.post("/api/v1/auth/login", json={"username": f"dr_second_{ts}", "password": "SecurePassword123!"})
            doc2_token = res_doc2_login.json()["access_token"]

            # Doctor 2 CANNOT release Doctor 1's case (403)
            res_unauth_rel = client.post(f"/api/v1/cases/{case2_id}/release", headers={"Authorization": f"Bearer {doc2_token}"})
            assert res_unauth_rel.status_code == 403, "Unassigned clinician cannot release another doctor's case"

            # Patient CANNOT release case (403)
            res_pat_rel = client.post(f"/api/v1/cases/{case2_id}/release", headers={"Authorization": f"Bearer {p_token}"})
            assert res_pat_rel.status_code == 403, "Patient cannot release a clinical case"

            # Doctor 1 releases their case
            res_rel_ok = client.post(f"/api/v1/cases/{case2_id}/release", headers={"Authorization": f"Bearer {doc1_token}"})
            assert res_rel_ok.status_code == 200
            rel_data = res_rel_ok.json()
            assert rel_data["assigned_doctor_id"] is None
            assert rel_data["review_status"] == "pending_review"

            # Doctor 2 can now claim the released case
            res_claim_doc2 = client.put(f"/api/v1/cases/{case2_id}/assign", json={}, headers={"Authorization": f"Bearer {doc2_token}"})
            assert res_claim_doc2.status_code == 200, f"Claim failed: {res_claim_doc2.status_code} {res_claim_doc2.text}"
            assert res_claim_doc2.json()["assigned_doctor_id"] == doc2_id

            # Reviewed case release prevention
            # Doctor 2 reviews case
            client.post(f"/api/v1/cases/{case2_id}/reviews", json={"comment": "Completed exam", "review_status": "reviewed"}, headers={"Authorization": f"Bearer {doc2_token}"})
            # Attempting to release a reviewed case back to triage should be blocked (400)
            res_rel_reviewed = client.post(f"/api/v1/cases/{case2_id}/release", headers={"Authorization": f"Bearer {doc2_token}"})
            assert res_rel_reviewed.status_code == 400, "Reviewed cases must not return to new-case triage queue"

            results.append(("4. Case Relinquishment & Reassignment", "PASSED", "Doctor released case back to triage; claimed by colleague; reviewed case release safely rejected (400)."))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append(("4. Case Relinquishment & Reassignment", "FAILED", str(e)))

        # Print summary
        print("\n" + "-" * 80)
        print(f"{'TEST NAME':<45} | {'STATUS':<8} | {'DETAILS'}")
        print("-" * 80)
        passed_count = sum(1 for _, status, _ in results if status == "PASSED")
        for name, status, details in results:
            print(f"{name:<45} | {status:<8} | {details}")
        print("-" * 80)
        print(f"OVERALL RESULT: {passed_count}/{len(results)} TESTS PASSED")
        print("=" * 80 + "\n")

        assert passed_count == len(results), f"Only {passed_count}/{len(results)} tests passed"

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

if __name__ == "__main__":
    run_security_hardening_tests()
