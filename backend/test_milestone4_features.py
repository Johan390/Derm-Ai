"""
DermAI Milestone 4 - Feature Verification Test Suite
=====================================================
Automated tests for:
1. Health endpoint database & ML telemetry contracts
2. In-App Notification System (dispatch, retrieval, read/unread, isolation)
3. Downloadable Clinical PDF Reports (structure, binary format, authorization)
4. RFC 4180 CSV Export (single case & batch cohort)
5. Structured JSON Export (single case & batch cohort)
6. Security boundaries on reports & notifications (BOLA/IDOR prevention)
"""

import io
import os
import json
import time
from datetime import datetime, timedelta, timezone
import jwt
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
import app.auth as auth_mod
import app.main as main_mod

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_auth():
    """
    Deterministic test-only authentication setup.
    Guards against WSL2 system clock jitter/skew causing JWT token 'iat' (issued-at)
    timestamps to be recorded in the future relative to rapid subsequent request execution,
    which triggers PyJWT ImmatureSignatureError (HTTP 401).
    Restores original functions and cleans up test uploads upon module completion.
    """
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

    yield

    auth_mod.create_access_token = orig_auth_token_func
    main_mod.create_access_token = orig_main_token_func

    # Clean up any test uploads created during test run
    upload_base = os.path.join(settings.BASE_DIR, "uploads")
    if os.path.isdir(upload_base):
        for root, dirs, files in os.walk(upload_base):
            for file in files:
                if file != ".gitkeep":
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception:
                        pass


def generate_test_image_bytes():
    """Generates simple valid RGB JPEG bytes for testing."""
    img = Image.new("RGB", (224, 224), color=(180, 140, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture(scope="module")
def auth_tokens():
    ts = int(time.time() * 1000)
    # Register Patient Alpha
    res_pa = client.post("/api/v1/auth/register", json={
        "username": f"m4_patient_a_{ts}",
        "email": f"m4_pa_{ts}@dermaitest.org",
        "full_name": "Alice Patient",
        "password": "Password123!",
        "role": "patient"
    })
    assert res_pa.status_code == 200
    token_pa = res_pa.json()["access_token"]
    user_id_pa = res_pa.json()["user_id"]

    # Register Patient Beta (for cross-user isolation tests)
    res_pb = client.post("/api/v1/auth/register", json={
        "username": f"m4_patient_b_{ts}",
        "email": f"m4_pb_{ts}@dermaitest.org",
        "full_name": "Bob Patient",
        "password": "Password123!",
        "role": "patient"
    })
    assert res_pb.status_code == 200
    token_pb = res_pb.json()["access_token"]
    user_id_pb = res_pb.json()["user_id"]

    # Pre-verified Clinician
    res_doc = client.post("/api/v1/auth/login", json={
        "username": "dr_smith",
        "password": "password123"
    })
    assert res_doc.status_code == 200
    token_doc = res_doc.json()["access_token"]
    user_id_doc = res_doc.json()["user_id"]

    return {
        "patient_a": {"token": token_pa, "user_id": user_id_pa},
        "patient_b": {"token": token_pb, "user_id": user_id_pb},
        "clinician": {"token": token_doc, "user_id": user_id_doc}
    }


def test_health_endpoint_contract():
    """Verify health endpoint reports DB connectivity and ML status."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "database" in data
    assert data["database"]["status"] == "connected"
    assert "ml_service" in data
    assert data["ml_service"]["status"] == "online"
    assert data["ml_service"]["model_name"] == "DermAI_MobileNetV2"
    assert data["ml_service"]["num_classes"] == 7
    assert "supported_models" in data["ml_service"]
    assert "modules_active" in data
    assert any("PDF" in m for m in data["modules_active"])


def test_notification_lifecycle_and_security(auth_tokens):
    """Test notification creation, retrieval, mark as read, and isolation."""
    tok_pa = auth_tokens["patient_a"]["token"]
    tok_pb = auth_tokens["patient_b"]["token"]

    # 1. Patient Alpha creates a case -> triggers notification
    res_case = client.post(
        "/api/v1/cases/",
        headers={"Authorization": f"Bearer {tok_pa}"},
        json={
            "patient_age": 35,
            "patient_gender": "female",
            "body_location": "forearm",
            "known_allergies": "None",
            "clinical_notes": "Milestone 4 Notification Test Case"
        }
    )
    assert res_case.status_code == 200
    case_data = res_case.json()
    case_id = case_data["id"]

    # 2. Check Patient Alpha's notifications
    res_notif = client.get(
        "/api/v1/notifications",
        headers={"Authorization": f"Bearer {tok_pa}"}
    )
    assert res_notif.status_code == 200
    notif_data = res_notif.json()
    assert notif_data["total_count"] >= 1
    assert notif_data["unread_count"] >= 1
    notifications = notif_data["notifications"]
    assert any(n["case_id"] == case_id for n in notifications)
    target_notif = [n for n in notifications if n["case_id"] == case_id][0]
    notif_id = target_notif["id"]
    assert target_notif["is_read"] is False

    # 3. Security Boundary: Patient Beta attempts to mark Patient Alpha's notification -> 403
    res_illegal_read = client.patch(
        f"/api/v1/notifications/{notif_id}/read",
        headers={"Authorization": f"Bearer {tok_pb}"}
    )
    assert res_illegal_read.status_code == 403

    # 4. Patient Alpha marks notification as read -> 200
    res_mark = client.patch(
        f"/api/v1/notifications/{notif_id}/read",
        headers={"Authorization": f"Bearer {tok_pa}"}
    )
    assert res_mark.status_code == 200
    assert res_mark.json()["is_read"] is True

    # 5. Mark all as read
    res_read_all = client.post(
        "/api/v1/notifications/read-all",
        headers={"Authorization": f"Bearer {tok_pa}"}
    )
    assert res_read_all.status_code == 200
    assert "marked_read" in res_read_all.json()


def test_clinical_pdf_report_generation(auth_tokens):
    """Test downloadable PDF report generation with ReportLab."""
    tok_pa = auth_tokens["patient_a"]["token"]
    tok_pb = auth_tokens["patient_b"]["token"]
    tok_doc = auth_tokens["clinician"]["token"]

    # Create case and upload image for Patient Alpha
    res_case = client.post(
        "/api/v1/cases/",
        headers={"Authorization": f"Bearer {tok_pa}"},
        json={
            "patient_age": 52,
            "patient_gender": "male",
            "body_location": "back",
            "known_allergies": "Sulfa",
            "medical_history": "Dysplastic nevi history",
            "clinical_notes": "Dark asymmetric macular lesion on mid-back"
        }
    )
    assert res_case.status_code == 200
    case_id = res_case.json()["id"]
    case_code = res_case.json()["case_code"]

    # Upload test image
    img_bytes = generate_test_image_bytes()
    res_upload = client.post(
        f"/api/v1/cases/{case_id}/upload-image",
        headers={"Authorization": f"Bearer {tok_pa}"},
        files={"file": ("lesion_test.jpg", img_bytes, "image/jpeg")}
    )
    assert res_upload.status_code == 200

    # Clinician claims case and adds review
    res_claim = client.put(
        f"/api/v1/cases/{case_id}/assign",
        headers={"Authorization": f"Bearer {tok_doc}"},
        json={}
    )
    assert res_claim.status_code == 200

    res_review = client.post(
        f"/api/v1/cases/{case_id}/reviews",
        headers={"Authorization": f"Bearer {tok_doc}"},
        json={
            "comment": "Pigment network atypical with focal architectural disorder. Biopsy advised.",
            "recommendation": "Perform 2mm punch or excisional biopsy for histopathological confirmation.",
            "review_status": "reviewed"
        }
    )
    assert res_review.status_code == 200

    # 1. Download PDF report as authorized owner (Patient Alpha)
    res_pdf = client.get(
        f"/api/v1/cases/{case_id}/report/pdf",
        headers={"Authorization": f"Bearer {tok_pa}"}
    )
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert "attachment" in res_pdf.headers["content-disposition"]
    assert case_code in res_pdf.headers["content-disposition"]
    # PDF binary signature check
    assert res_pdf.content.startswith(b"%PDF-")
    assert len(res_pdf.content) > 1000

    # 2. Download PDF report as authorized Clinician
    res_doc_pdf = client.get(
        f"/api/v1/cases/{case_id}/report/pdf",
        headers={"Authorization": f"Bearer {tok_doc}"}
    )
    assert res_doc_pdf.status_code == 200
    assert res_doc_pdf.content.startswith(b"%PDF-")

    # 3. Security Boundary: Unauthorized Patient Beta blocked (403)
    res_unauth_pdf = client.get(
        f"/api/v1/cases/{case_id}/report/pdf",
        headers={"Authorization": f"Bearer {tok_pb}"}
    )
    assert res_unauth_pdf.status_code == 403

    # 4. Security Boundary: Unauthenticated access blocked (401)
    res_anon_pdf = client.get(f"/api/v1/cases/{case_id}/report/pdf")
    assert res_anon_pdf.status_code == 401


def test_csv_and_json_exports(auth_tokens):
    """Test authorized CSV and JSON exports."""
    tok_pa = auth_tokens["patient_a"]["token"]
    tok_pb = auth_tokens["patient_b"]["token"]
    tok_doc = auth_tokens["clinician"]["token"]

    # Create a case for Patient Alpha
    res_case = client.post(
        "/api/v1/cases/",
        headers={"Authorization": f"Bearer {tok_pa}"},
        json={
            "patient_age": 60,
            "patient_gender": "female",
            "body_location": "neck",
            "clinical_notes": "Export test lesion"
        }
    )
    assert res_case.status_code == 200
    case_id = res_case.json()["id"]

    # 1. Single Case CSV Export
    res_csv = client.get(
        f"/api/v1/cases/{case_id}/export/csv",
        headers={"Authorization": f"Bearer {tok_pa}"}
    )
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    csv_text = res_csv.text
    assert "case_id" in csv_text
    assert "predicted_class" in csv_text
    assert str(case_id) in csv_text

    # Cross-patient single CSV blocked
    res_csv_block = client.get(
        f"/api/v1/cases/{case_id}/export/csv",
        headers={"Authorization": f"Bearer {tok_pb}"}
    )
    assert res_csv_block.status_code == 403

    # 2. Single Case JSON Export
    res_json = client.get(
        f"/api/v1/cases/{case_id}/export/json",
        headers={"Authorization": f"Bearer {tok_pa}"}
    )
    assert res_json.status_code == 200
    data = res_json.json()
    assert data["id"] == case_id
    assert "disclaimer" in data
    assert "predictions" in data

    # 3. Batch CSV Export
    res_batch_csv = client.get(
        "/api/v1/cases/export/csv",
        headers={"Authorization": f"Bearer {tok_doc}"}
    )
    assert res_batch_csv.status_code == 200
    assert "text/csv" in res_batch_csv.headers["content-type"]

    # 4. Batch JSON Export
    res_batch_json = client.get(
        "/api/v1/cases/export/json",
        headers={"Authorization": f"Bearer {tok_doc}"}
    )
    assert res_batch_json.status_code == 200
    assert isinstance(res_batch_json.json(), list)
