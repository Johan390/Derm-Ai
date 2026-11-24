# DermAI Security, End-to-End & Testing Verification Summary

**Project:** DermAI Skin Disease Detector from Mobile Images Using DL  
**Status:** 100% VERIFIED PASS (28/28 Regression Tests Passed)  
**Artifact Directory:** [`reports/testing/`](.)

---

## 1. Regression Test Suite Execution

The automated verification suite in [`reports/testing/regression_results.json`](regression_results.json) validates the end-to-end integrity of the entire DermAI application stack:

- **Total Tests Executed:** **28**
- **Passed:** **28 (100%)**
- **Failed:** **0**
- **Execution Date:** 2026-11-24

### Summary of Passed Verification Categories:
1. **Model Checkpoint Integrity:** Cryptographic verification of MD5 and SHA-256 hashes against original baseline checkpoints for `mobilenetv2`, `phase1`, `efficientnetb0`, and `vit`.
2. **Inference Endpoints:** Verification of real-time predictions for all 4 operational modes:
   - `mobilenet`: HTTP 200, valid class probabilities summing to 1.0.
   - `efficientnet`: HTTP 200, valid class probabilities summing to 1.0.
   - `vit`: HTTP 200, valid class probabilities summing to 1.0.
   - `ensemble`: HTTP 200, equal-weight consensus of MobileNetV2 and EfficientNetB0.
3. **Explainability Endpoints:**
   - Grad-CAM heatmap overlay: HTTP 200, non-empty JPEG stream.
   - Saliency map overlay: HTTP 200, non-empty JPEG stream.
   - ViT Attention Rollout (Overlay): HTTP 200, non-empty 224x224 RGB JPEG stream.
   - ViT Attention Rollout (Heatmap): HTTP 200, non-empty 224x224 RGB JPEG stream.
4. **Authentication & RBAC Security:** Role-based access control, token validation, and authorization boundaries.
5. **Database Preservation:** Verification that PostgreSQL clinical records and the SQLite historical archive remain unmodified.

---

## 2. End-to-End User Journey Verification

The system was audited against five realistic clinical user journeys:

### Journey A: Patient Journey
- Self-registration with valid email and password hashing (Bcrypt cost 12).
- Case creation and dermoscopic image upload.
- Automated Image Quality Assessment (IQA) gatekeeper execution.
- AI screening inference generation.
- Automated patient notification delivery.

### Journey B: Clinician Journey
- Account creation requiring explicit Administrator verification before patient access is granted.
- Unassigned triage queue discovery and explicit case assignment (`claimed_by`).
- Model inference execution with Grad-CAM visual interpretability review.
- Clinical review submission, diagnostic notes, follow-up scheduling, and PDF report export.
- **Assignment Guard:** Clinicians cannot review cases claimed by other clinicians (strict HTTP 403 Forbidden).

### Journey C: Researcher Journey
- Read-only access to dataset exploration and exploratory data analysis (EDA) endpoints.
- De-identified research cohort exports.
- **PHI Barrier:** Strict HTTP 403 Forbidden on patient identifying clinical endpoints.

### Journey D: Administrator Journey
- Unverified clinician queue management.
- Clinician approval/rejection workflows.
- System health auditing and service metric inspection.
- **Privilege Separation:** Non-admin accounts strictly blocked (HTTP 403) from approving clinicians.

### Journey E: ML Engineer Journey
- Model registry inspection and version management (`v1.0`, `v1.1-vit`, `v2.0-ensemble`).
- Lineage tracking and active model rollback testing.
- **Clinical Data Barrier:** Strict HTTP 403 Forbidden on clinical case records.

---

## 3. Security Hardening & Penetration Defense

| Security Control | Implementation Details | Verified Test Status |
| :--- | :--- | :---: |
| **Authentication** | JWT tokens signed with HS256 algorithm and 24-hour expiration; Bcrypt password hashing (cost factor 12). | ✅ PASSED |
| **BOLA / IDOR Protection** | Patients can access only their own cases; cross-patient case access attempts return HTTP 404/403. | ✅ PASSED |
| **Clinician Assignment Protection** | Only the assigned clinician or administrator can submit reviews or edit diagnoses. | ✅ PASSED |
| **Non-Admin Approval Defense** | Elevation of privilege attacks blocked; unverified users cannot bypass review. | ✅ PASSED |
| **Path Traversal Protection** | File uploads sanitized using UUID prefixes; absolute path escapes and `../` sequences rejected. | ✅ PASSED |
| **MIME / Content Sniffing Guard** | Strict magic-byte validation and PIL image header decoding; non-image files rejected. | ✅ PASSED |
| **Upload Size Guard** | Strict 15 MB payload ceiling enforced at both Nginx proxy and FastAPI middleware. | ✅ PASSED |
| **Database Connection Pooling** | Connection pooling with pre-ping validation to prevent stale socket leaks. | ✅ PASSED |

---

## 4. Image Quality Assessment (IQA) Heuristic Defense

DermAI implements automated pre-inference image quality screening calibrated on HAM10000 empirical distributions:

- **Laplacian Blur Filter:** Rejects blurry captures (variance $< 20.0$) preventing out-of-focus misdiagnoses.
- **Contrast Screening:** Detects low-contrast captures ($\text{RMS} < 12.0$) typically caused by poor ambient lighting.
- **Over/Under-Exposure Defense:** Rejects severe glare (mean brightness $> 225.0$) and underexposed captures ($< 40.0$).
- **Minimum Resolution Guard:** Enforces minimum dimension $\ge 150\text{ px}$ before 224x224 tensor interpolation.

---

## 5. ViT Attention Rollout Verification & Resolution

- **Root Cause of Historical 0-Byte Response:** The FastAPI `/predict` route defines `model` as a query parameter (`POST /api/v1/cases/{case_id}/predict?model=vit`). Legacy test harnesses erroneously passed `json={"model": "vit"}` in the request body, causing the endpoint to fall back to default models (`mobilenet` or `ensemble`). When `/vit-attention-rollout` was called, it threw HTTP 409 Conflict because the active prediction was not a ViT model.
- **Resolution:** The test harness was updated to supply `params={"model": "vit"}` and supply `prediction_id`.
- **Verified Output:**
  - `GET /api/v1/cases/{case_id}/vit-attention/overlay`: Returns HTTP 200 OK, **4,705 bytes**, RGB format, 224x224 dimensions.
  - `GET /api/v1/cases/{case_id}/vit-attention/heatmap`: Returns HTTP 200 OK, **5,508 bytes**, RGB format, 224x224 dimensions.
  - Checkpoint integrity: Production ViT checkpoint (`vit_skin_lesion_exp2_best.keras`) unmodified.
