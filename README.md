# DermAI — Clinical Skin Disease Screening & Quality Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react)](https://react.dev)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg?logo=tensorflow)](https://tensorflow.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1.svg?logo=postgresql)](https://www.postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

DermAI is a clinical decision-support web platform engineered to assist healthcare professionals and patients in screening dermoscopic images for skin lesions. The system pairs pre-inference **Image Quality Assessment (IQA)** with multi-architecture deep learning classification (**MobileNetV2**, **EfficientNetB0**, **Vision Transformer**, and **Soft Voting Ensemble**) and clinical explainability (**Grad-CAM**, **Saliency Maps**, and **ViT Multi-Head Attention Rollout**), secured by strict Role-Based Access Control (RBAC).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [Clone and Configure](#3-clone-and-configure)
4. [Recommended Method: Run with Docker Compose](#4-recommended-method-run-with-docker-compose)
5. [First-Time Application Usage](#5-first-time-application-usage)
6. [Manual Development Setup (Without Docker)](#6-manual-development-setup-without-docker)
7. [Machine Learning Models and Dataset](#7-machine-learning-models-and-dataset)
8. [Testing and Verification](#8-testing-and-verification)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Security Architecture & Data Handling](#10-security-architecture--data-handling)
11. [Project Structure](#11-project-structure)
12. [Important Limitations & AI Medical Disclaimer](#12-important-limitations--ai-medical-disclaimer)

---

## 1. Project Overview

### What DermAI Does
DermAI facilitates early dermatological risk identification by evaluating mobile dermoscopy images against seven diagnostic categories established by the canonical **HAM10000** ("Human Against Machine with 10000 training images") and International Skin Imaging Collaboration (ISIC) benchmarks:
- **MEL**: Melanoma *(High Risk / Urgent Clinical Review)*
- **BCC**: Basal Cell Carcinoma *(High Risk / Clinical Review)*
- **AKIEC**: Actinic Keratosis / Intraepithelial Carcinoma *(Moderate Risk)*
- **BKL**: Benign Keratosis *(Low Risk)*
- **NV**: Melanocytic Nevi *(Low Risk)*
- **DF**: Dermatofibroma *(Low Risk)*
- **VASC**: Vascular Lesions *(Low Risk)*

### System Architecture
The platform is built as a modular micro-service stack:
- **FastAPI Backend (Python 3.11):** High-performance asynchronous REST API handling authentication (JWT / Bcrypt), clinical case management, IQA validation, neural network inference, Grad-CAM/attention map synthesis, and PDF report compilation (ReportLab).
- **React Frontend (React 19 + Vite):** Responsive Single Page Application (SPA) offering dedicated portals for patients, clinicians, and administrators.
- **Relational Database:** PostgreSQL 16 (in Docker deployment) or SQLite (in local development fallback) orchestrated through SQLAlchemy 2.0 ORM.
- **Machine Learning Subsystem (TensorFlow 2.21 / Keras 3.15):** Standalone inference pipelines supporting MobileNetV2, EfficientNetB0, Vision Transformer (ViT-16), and Soft Voting consensus ensembles.
- **Reverse Proxy & Web Server (Nginx 1.27 Alpine):** Multi-stage production container serving static assets, enforcing HTTP security headers, and reverse-proxying `/api/` traffic directly to the backend.

### Supported User Roles
The application strictly enforces Role-Based Access Control (RBAC) across three active roles:
1. **Patient:** Register, authenticate, create clinical screening cases, upload dermoscopic images, track review statuses, and view approved clinician reports.
2. **Clinician:** Register (pending administrative verification), view unassigned triage queues, claim cases, inspect multi-model AI predictions and visual explainability maps, author clinical diagnostic reviews, set follow-up schedules, and download structured clinical PDF reports.
3. **Administrator:** System overseer role. Reviews pending clinician registrations, approves or rejects clinician credentials, reassigns cases across clinicians, and inspects operational telemetry and audit logs.

> [!IMPORTANT]
> **Clinical Decision-Support Software Notice:**  
> DermAI is an experimental clinical decision-support tool designed to assist healthcare professionals. **It is NOT an autonomous medical device or a replacement for professional clinical judgment, histopathological examination, or physician diagnosis.**

---

## 2. Prerequisites

### For Recommended Docker Execution (Primary)
- **Operating System:** Windows 10/11 (64-bit), macOS, or Linux.
- **Docker Desktop:** Docker Engine 24.0+ with Docker Compose v2.20+.
  - *On Windows:* Ensure Docker Desktop is installed with the **WSL 2 backend** enabled.
- **Git:** Git 2.30+ installed and available in PATH.
- **Hardware:** Minimum 4 CPU cores, 8 GB RAM (16 GB recommended for multi-model neural network inference), and 10 GB free disk space.

### For Manual Development Setup (Without Docker, Optional)
- **Python:** Python 3.11 or 3.12 (64-bit).
- **Node.js:** Node.js 20 LTS or higher and npm 10+.
- **Database (Optional):** PostgreSQL 16 server (or use the built-in SQLite development fallback).
- **C/C++ Build Tools:** Standard build tools (Visual Studio C++ Build Tools on Windows or `build-essential` on Linux) if compiling native dependencies like `psycopg2` or `opencv-python`.

---

## 3. Clone and Configure

Follow these exact steps in Windows PowerShell to clone and configure the repository:

### Step 1: Clone the Repository
```powershell
git clone https://github.com/your-username/dermai.git
cd dermai
```

### Step 2: Create Environment Configuration
Copy the provided environment template `.env.example` to create `.env`:
```powershell
Copy-Item .env.example .env
```

### Step 3: Configure Environment Variables
Generate a cryptographically secure 256-bit hexadecimal string for `SECRET_KEY`:
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Open `.env` in your preferred editor (e.g., `notepad .env` or `code .env`) and update the key parameters:

| Variable | Required / Default | Purpose & Description |
| :--- | :---: | :--- |
| `ENVIRONMENT` | `development` / `production` | Deployment mode. Production mode enforces strict secret and seeding checks. |
| `SECRET_KEY` | **Required** | Secret key used for signing HS256 JWT authentication tokens. |
| `POSTGRES_USER` | `dermai_admin` | Database username for the PostgreSQL container. |
| `POSTGRES_PASSWORD` | `change_this_strong_password_in_env` | Database password. Change to a secure password. |
| `POSTGRES_DB` | `dermai_production` | PostgreSQL database name. |
| `SEED_DEMO_USERS` | `true` (dev) / `false` (prod) | When `true`, automatically provisions offline demo accounts on startup. |
| `BACKEND_PORT` | `8000` | Host port mapped to FastAPI backend. |
| `FRONTEND_PORT` | `80` | Host port mapped to Nginx frontend. |
| `MAX_FILE_SIZE_MB` | `15` | Maximum dermoscopy image upload payload size in megabytes. |

> [!WARNING]
> Never commit `.env` or production secrets to Git. The `.env` file is excluded by `.gitignore`.

---

## 4. Recommended Method: Run with Docker Compose

Running via Docker Compose is the recommended and simplest method to deploy DermAI with all networking, database schemas, and dependencies pre-configured.

### Step-by-Step Launch Procedure

#### 1. Verify Docker Desktop is Running
Ensure the Docker Desktop engine is active before running commands:
```powershell
docker --version
docker compose version
docker info
```
*If `docker info` returns an engine connection error, launch Docker Desktop from the Windows Start menu and wait until the status indicates "Engine running".*

#### 2. Start the Development Stack
From the project root directory, build and launch all containers in detached mode:
```powershell
docker compose up -d --build
```

#### 3. Inspect Container Status
Check that all three services are running and report healthy status:
```powershell
docker compose ps
```
You should see:
- `dermai_postgres` — Port `5432/tcp` (healthy)
- `dermai_backend` — Port `0.0.0.0:8000->8000/tcp` (healthy)
- `dermai_frontend` — Port `0.0.0.0:80->80/tcp` (running)

#### 4. Stream Service Logs
To view aggregate logs or monitor a specific service:
```powershell
# Stream all logs:
docker compose logs -f

# Or stream a specific service:
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f postgres
```

#### 5. Access the Web Application and APIs
Once the containers are healthy, open your web browser:
- **Frontend Web Portal:** [`http://localhost`](http://localhost) (or [`http://localhost:80`](http://localhost:80))
- **Interactive Backend API Documentation (Swagger UI):** [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **Alternative ReDoc API Documentation:** [`http://localhost:8000/redoc`](http://localhost:8000/redoc)
- **Health Check & Telemetry Endpoint:** [`http://localhost:8000/api/v1/health`](http://localhost:8000/api/v1/health)

### Architectural Service Breakdown

```
                         Browser Client
                               │
                      [Port 80 (HTTP)]
                               ▼
        ┌──────────────────────────────────────────────┐
        │        dermai_frontend (Nginx Alpine)        │
        │  • Serves React 19 Production SPA Bundle     │
        │  • Reverse-proxies /api/ to backend:8000     │
        └──────────────────────┬───────────────────────┘
                               │ Internal Container Network
                      [Port 8000 (HTTP)]
                               ▼
        ┌──────────────────────────────────────────────┐
        │         dermai_backend (FastAPI / Uvicorn)   │
        │  • REST API & Authentication (JWT/Bcrypt)    │
        │  • IQA & ML Pipelines (Keras / TensorFlow)   │
        │  • Mounted Volume: persistent_uploads        │
        └──────────────────────┬───────────────────────┘
                               │ Internal Container Network
                      [Port 5432 (TCP)]
                               ▼
        ┌──────────────────────────────────────────────┐
        │       dermai_postgres (PostgreSQL 16)        │
        │  • Persistent relational storage             │
        │  • Mounted Volume: postgres_data             │
        └──────────────────────────────────────────────┘
```

### Named Volumes and Data Persistence
DermAI manages state through two dedicated Docker named volumes:
1. `postgres_data`: Preserves all PostgreSQL tables, user accounts, clinical cases, and review records.
2. `persistent_uploads`: Preserves all original uploaded dermoscopy images, preprocessed images, and thumbnail previews across container restarts.

### Stopping and Restarting Safely
To stop the application **without losing data**:
```powershell
docker compose stop
# or
docker compose down
```
To restart the application with your existing database and uploads intact:
```powershell
docker compose up -d
```

> [!CAUTION]
> **Data Loss Warning:** Running `docker compose down -v` deletes the named volumes (`postgres_data` and `persistent_uploads`). This will permanently erase all stored cases, users, and patient images. Never pass the `-v` flag unless you intend to perform an irreversible fresh reset.

### Hardened Production Deployment
To deploy using the hardened production configuration:
```powershell
docker compose -f docker-compose.production.yml up -d --build
```
*Key production guardrails:*
- Demo user seeding is strictly disabled (`SEED_DEMO_USERS=false`).
- Enforces an explicit, strong `SECRET_KEY` in `.env` (the compose engine will refuse to start if unset).
- Model checkpoints are mounted strictly read-only (`:ro`).

---

## 5. First-Time Application Usage

### Real User Journey & Supported Workflows

```
  PATIENT WORKFLOW              CLINICIAN WORKFLOW             ADMINISTRATOR WORKFLOW
 ──────────────────            ────────────────────           ────────────────────────
  1. Register / Login           1. Register Account            1. Login as Admin
        │                             │                              │
  2. Create Screening Case      2. Status: is_verified=False   2. Inspect Pending Clinicians
        │                             │                              │
  3. Upload Lesion Photo              │ (Awaiting Verification) 3. Approve Credentials
        │                             ▼                              │
  4. Automatic IQA Check       [Blocked from Clinical Queue]   4. Reassign Cases (Optional)
        │                             │                              │
  5. AI Inference & Heatmaps          │ (Approved by Admin)          ▼
        │                             ▼                        Review System Audit Logs
  6. Track Status               3. Access Triage Queue
        │                             │
  7. View Approved Report <──── 4. Claim Case & Review AI
                                      │
                                5. Submit Clinical Plan
                                      │
                                6. Export Clinical PDF
```

#### 1. Patient Screening Workflow
1. Navigate to [`http://localhost`](http://localhost) and click **Register**.
2. Select the **Patient** role, provide username, email, full name, and password.
3. Log in with your new patient credentials.
4. Click **Create New Case** and input patient details (age, biological sex, anatomical body location, medical history).
5. Attach a dermoscopy photograph (JPEG, PNG, or WebP).
6. The system executes automated **Image Quality Assessment (IQA)**:
   - Validates sharpness (Laplacian variance $> 20.0$).
   - Validates contrast (standard deviation $> 12.0$).
   - Validates brightness ($40.0 \le \mu \le 225.0$).
   - Validates resolution ($\ge 150 \times 150$ pixels).
7. Upon successful validation, initiate AI inference to receive predicted risk scores across MobileNetV2, EfficientNetB0, ViT, or Ensemble, alongside Grad-CAM activation heatmaps.

#### 2. Clinician Workflow & Approval Barrier
1. Navigate to [`http://localhost`](http://localhost), click **Register**, and select the **Clinician** role.
2. Complete registration.
3. Newly registered clinicians are created with `is_verified = False` to protect patient privacy.
4. Until approved, the clinician dashboard displays an *Access Pending* notice and blocks triage queue access (`HTTP 403`).
5. Once verified by an Administrator, the clinician gains full access to:
   - Browse the unassigned triage queue.
   - Claim cases for review.
   - Inspect AI diagnostic predictions and explainability maps.
   - Author formal clinical notes, confirm/override diagnoses, set follow-up alerts, and generate downloadable clinical PDF summaries.

#### 3. Administrator Workflow & First-Time Admin Provisioning

Public registration (`POST /api/v1/auth/register`) strictly permits only `patient` and `clinician` roles. Registering an `admin` via the public API is blocked (`HTTP 400 Bad Request`).

To provision the **initial administrator account**, run the secure provisioning command below:

##### A. Provisioning Admin in Docker (Recommended):
Run inside the running `backend` container via PowerShell:
```powershell
docker compose exec backend python -c "from app.database import SessionLocal; from app.models import User, UserRole; from app.auth import hash_password; db = SessionLocal(); exists = db.query(User).filter(User.role == UserRole.ADMIN).first(); print('Admin account already exists.') if exists else (db.add(User(username='admin', email='admin@dermai.local', full_name='System Administrator', role=UserRole.ADMIN, is_verified=True, hashed_password=hash_password('YOUR_SECURE_PASSWORD'))), db.commit(), print('Administrator provisioned successfully.'))"
```

##### B. Provisioning Admin in Manual Development (Without Docker):
Run in your active Python virtual environment from the `backend/` directory:
```powershell
python -c "from app.database import SessionLocal; from app.models import User, UserRole; from app.auth import hash_password; db = SessionLocal(); exists = db.query(User).filter(User.role == UserRole.ADMIN).first(); print('Admin account already exists.') if exists else (db.add(User(username='admin', email='admin@dermai.local', full_name='System Administrator', role=UserRole.ADMIN, is_verified=True, hashed_password=hash_password('YOUR_SECURE_PASSWORD'))), db.commit(), print('Administrator provisioned successfully.'))"
```

Once logged in as an Administrator, you can:
- Navigate to the **Admin Portal** (`/api/v1/admin/clinicians/pending`).
- Click **Verify Clinician** to approve newly registered physicians.
- Reassign clinical cases to specific doctors.
- Monitor system health metrics and audit trails.

### Development Demo Accounts
When `SEED_DEMO_USERS=true` in `.env` (development mode), the application seeds initial accounts on first startup:
- Clinician account: `dr_smith`
- Patient account: `patient_demo`

*In production deployments, demo seeding is disabled to guarantee zero unauthenticated or default-credential exposure.*

---

## 6. Manual Development Setup (Without Docker)

If you prefer developing on your host machine without Docker containers, you can run the FastAPI backend and React frontend separately.

### Backend Setup

```powershell
# 1. Navigate to backend directory
cd backend

# 2. Create and activate a Python 3.11/3.12 virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install backend dependencies
pip install -r requirements.txt

# 4. Configure local environment variables
$env:ENVIRONMENT = "development"
$env:SECRET_KEY = "your-local-development-secret-key-minimum-32-chars"

# Database Configuration:
# When DATABASE_URL is unset, DermAI defaults automatically to local SQLite:
# sqlite:///backend/dermai.db
# Alternatively, if you have a local PostgreSQL server:
# $env:DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5432/dermai"

# 5. Start the FastAPI development server
python run.py
```
The backend API and Swagger UI will be accessible at [`http://127.0.0.1:8000`](http://127.0.0.1:8000).

### Frontend Setup

```powershell
# 1. Open a new PowerShell terminal and navigate to frontend directory
cd frontend

# 2. Install dependencies deterministically
npm ci

# 3. Start Vite development server
npm run dev
```
The frontend development server will launch at [`http://localhost:5173`](http://localhost:5173).

*Frontend Communication:* In development mode, `frontend/vite.config.js` proxies `/api` requests to the running backend. If you configure a custom port, specify `VITE_API_BASE_URL` in `frontend/.env.local`.

---

## 7. Machine Learning Models and Dataset

### Bundled Model Checkpoint Inventory
All four validated deep learning checkpoints reside in [`ml_models/saved_models/`](ml_models/saved_models/) and are directly tracked in Git (each file is $<20\text{ MB}$, well within standard Git file limits):

| Checkpoint File | Architecture | File Size | MD5 Checksum | Purpose / Role |
| :--- | :--- | :---: | :--- | :--- |
| `mobilenetv2_skin_lesion_best.keras` | MobileNetV2 (Phase 2 Fine-Tuned) | 11.1 MB | `e4b31552d46e54c871bf8a6bbc5a8421` | Baseline CNN & Ensemble constituent |
| `phase1_best_model.keras` | MobileNetV2 (Phase 1 Frozen Backbone) | 11.1 MB | `e4b31552d46e54c871bf8a6bbc5a8421` | Training checkpoint baseline |
| `efficientnetb0_skin_lesion.keras` | EfficientNetB0 (Compound Scaled) | 18.2 MB | `6862780bd2cf4eb24b31d19de82f4218` | High-capacity CNN & Ensemble constituent |
| `vit_skin_lesion_exp2_best.keras` | Vision Transformer (ViT-16 Patch) | 6.2 MB | `3e7f9c8afc6de4b4d2e0213df4dd2c16` | Transformer self-attention classifier |

### Inference vs. Retraining Requirements
- **Standard Inference & Clinical Demonstration:** **No dataset download required.** The pre-trained checkpoints above provide immediate, fully functional multi-model predictions and explainability maps without needing the external HAM10000 image archives.
- **Model Retraining & Evaluation:** Due to storage constraints (2.6+ GB, ~10,015 images), raw training images are excluded from Git. To retrain or run full cohort evaluations, acquire the dataset using the instructions below.

### Obtaining the HAM10000 Dataset for Retraining
1. Follow the instructions in [`ml_dataset/download_instructions.md`](ml_dataset/download_instructions.md).
2. Download via Kaggle CLI:
   ```bash
   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   unzip skin-cancer-mnist-ham10000.zip -d ml_dataset/raw_ham10000/
   ```
3. Verified patient-stratified split manifests are located in [`ml_models/data_splits/`](ml_models/data_splits/):
   - `train_split.csv`
   - `val_split.csv`
   - `test_split.csv`

---

## 8. Testing and Verification

DermAI includes a comprehensive regression, security, and integration test suite across the backend and frontend.

### 1. Run Backend Pytest Suite
In an active Python environment from the repository root:
```powershell
pytest backend/
```
Or execute targeted test suites individually:
```powershell
pytest backend/test_storage.py
pytest backend/test_e2e_workflows.py
pytest backend/test_prediction_api.py
pytest backend/test_role_workflow_correction.py
pytest backend/test_security_hardening.py
pytest backend/test_stage6_security_validation.py
pytest backend/test_verification_suite.py
```

### 2. Standalone Verification Scripts
```powershell
# Run end-to-end integration workflows:
python backend/test_combined_workflow.py
python backend/test_frontend_integration.py
```

> [!NOTE]
> **Fresh-Clone Fixture Notice for `backend/test_ml_service_vit.py`:**  
> The standalone test `backend/test_ml_service_vit.py` expects a sample lesion image in `backend/uploads/original/`. On a fresh clone, run the demo case seeder first to initialize the synthetic lesion fixture before running this script:
> ```powershell
> python backend/seed_demo_case.py
> python backend/test_ml_service_vit.py
> ```

### 3. Frontend Production Build Verification
From the `frontend/` directory:
```powershell
cd frontend
npm ci
npm run build
```
Verify that the build exits with code `0` and generates the production bundle in `frontend/dist/`.

### 4. Docker Compose Configuration Validation
Validate Compose file syntax and environment interpolation without starting containers:
```powershell
# Development Compose validation:
docker compose config

# Production Compose validation:
docker compose -f docker-compose.production.yml config
```

---

## 9. Troubleshooting Guide

| Issue / Error Symptom | Probable Cause | Recommended Resolution |
| :--- | :--- | :--- |
| `failed to connect to the docker API at npipe://...` | Docker Desktop is not running on the Windows host. | Open Docker Desktop from the Start Menu. Verify the tray icon indicates "Docker Desktop is running" before retrying commands. |
| `Bind for 0.0.0.0:80 failed: port is already allocated` | Another service (IIS, Skype, local Nginx) is using port 80. | In `.env`, set `FRONTEND_PORT=8080`. The application will then be accessible at `http://localhost:8080`. |
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | Another service is using port 8000. | In `.env`, set `BACKEND_PORT=8001`. Access backend docs at `http://localhost:8001/docs`. |
| `services.backend.environment.SECRET_KEY is missing a value` | Running production Compose without defining `SECRET_KEY`. | Production mode forbids default secrets. Generate a key using `python -c "import secrets; print(secrets.token_hex(32))"` and set `SECRET_KEY=...` in `.env`. |
| `postgres healthcheck timeout / failed` | Low memory or disk I/O contention during PostgreSQL startup. | Increase Docker Desktop memory allocation in Settings > Resources (minimum 4 GB allocated to WSL2/Docker). |
| `Network Error` or `Failed to fetch` in browser | Backend is still starting up or restarting. | Run `docker compose ps` to verify `dermai_backend` is healthy. Check logs with `docker compose logs backend`. |
| `401 Unauthorized` immediately after login | System clock jitter or timezone drift between host and container. | Ensure Windows and WSL2 clocks are synchronized (`sudo hwclock -s` in WSL). |
| `FileNotFoundError` during `test_ml_service_vit.py` | Standalone script executed before creating an upload fixture. | Execute `python backend/seed_demo_case.py` to seed the initial synthetic sample lesion. |

---

## 10. Security Architecture & Data Handling

> [!WARNING]
> ### Regulatory Compliance Disclaimer
> **DermAI has NOT been evaluated, audited, or certified for regulatory compliance under HIPAA, GDPR, HITECH, FDA 21 CFR Part 11, or other medical device/health data standards.** The security mechanisms described below are engineering safeguards designed for research and prototyping. They do not constitute legal or statutory compliance. Organizations deploying DermAI are solely responsible for conducting independent security audits, risk assessments, and compliance evaluations before processing any Protected Health Information (PHI) or personal health data.

### Implemented Engineering Security Controls

1. **Authentication & Role-Based Access Control (RBAC):**
   - Cryptographic password hashing using Bcrypt with work factor 12.
   - Time-limited JSON Web Tokens (HS256) with 24-hour expiration for stateless API authorization.
   - Strict role segregation (`Patient`, `Clinician`, `Administrator`) enforced at the API route layer via FastAPI dependency injection.
   - Administrative gatekeeping: newly registered clinician accounts default to unverified (`is_verified = False`) and are strictly blocked (`HTTP 403`) from accessing patient queues or records until verified by an Administrator.
   - Broken Object Level Authorization (BOLA/IDOR) guards preventing cross-patient case, report, or notification access.

2. **Protected Patient Uploads & Local File Segregation:**
   - Patient-uploaded dermoscopy photos are segregated into local directories (`backend/uploads/original/`, `preview/`, `preprocessed/`).
   - All upload directories are strictly excluded from version control via `.gitignore` to prevent committing patient imagery.
   - Path traversal protections reject directory traversal vectors (`../`, absolute paths outside the designated upload root).

3. **Environment-Based Secret Management:**
   - Secrets, database credentials, and signing keys are decoupled from code and configured exclusively through environment variables or `.env`.
   - `.env` files are strictly excluded from version control via `.gitignore`.
   - In production mode (`ENVIRONMENT=production`), the application strictly validates that `SECRET_KEY` is not using the default development fallback string, aborting startup with a `RuntimeError` if an insecure key is detected.

4. **Local Database & Upload Persistence:**
   - When running in Docker, clinical records, user accounts, and diagnostic reviews are persisted in isolated Docker named volumes (`postgres_data`).
   - Patient uploads are stored in a dedicated named volume (`persistent_uploads`).
   - Data persists across container restarts unless volumes are explicitly deleted (`docker compose down -v`).
   - Backup procedures: In production, database dumps can be executed with `pg_dump`:
     ```powershell
     docker compose exec postgres pg_dump -U dermai_admin dermai_production > dermai_backup.sql
     ```
     And restored with:
     ```powershell
     Get-Content dermai_backup.sql | docker compose exec -T postgres psql -U dermai_admin -d dermai_production
     ```

5. **Data Handling Limitations:**
   - DermAI does not incorporate built-in data encryption at rest for the filesystem upload directory or database volumes; host-level volume encryption (e.g., BitLocker or LUKS) must be managed by the infrastructure administrator.
   - TLS/HTTPS termination is not bundled in the default Docker Compose configuration; production deployments must place DermAI behind an external reverse proxy (e.g., Traefik, AWS ALB, Nginx with Certbot) configured with valid SSL/TLS certificates.
   - Exported reports (PDF, CSV, JSON) contain clinical case details and must be handled in accordance with local organizational data governance policies.

---

## 11. Project Structure

```
dermai/
├── .dockerignore                    # Build context exclusions for Docker images
├── .env.example                     # Reference environment configuration template
├── .gitignore                       # Git exclusion rules (PHI, secrets, caches, db)
├── docker-compose.yml               # Development multi-container orchestration
├── docker-compose.production.yml    # Hardened production Docker orchestration
├── package.json                     # Root script runner
├── run.py                           # Root Python application launcher
├── README.md                        # Primary project documentation
│
├── backend/                         # FastAPI Application Service
│   ├── Dockerfile                   # Python 3.11-slim production container definition
│   ├── requirements.txt             # Python backend dependencies
│   ├── run.py                       # Local Uvicorn server launcher
│   ├── seed_demo_case.py            # Development fixture and synthetic image generator
│   ├── app/                         # Core Application Package
│   │   ├── auth.py                  # JWT creation, Bcrypt password hashing, RBAC guards
│   │   ├── config.py                # Environment configuration and IQA thresholds
│   │   ├── database.py              # SQLAlchemy engine and session management
│   │   ├── main.py                  # FastAPI route controllers and lifespan hooks
│   │   ├── models.py                # SQLAlchemy ORM entity models (User, Case, Image, etc.)
│   │   ├── schemas.py               # Pydantic validation and serialization schemas
│   │   ├── preprocessing.py         # Image Quality Assessment & OpenCV pipelines
│   │   ├── ml_service.py            # Inference engine (CNN, ViT, Ensemble, Grad-CAM)
│   │   ├── report_service.py        # PDF clinical summary report generator (ReportLab)
│   │   └── notification_service.py  # User notification and alert dispatching
│   ├── uploads/                     # Local storage root for dermoscopy photos (gitignored)
│   │   ├── original/                # Untouched clinical image uploads
│   │   ├── preview/                 # Display-optimized web thumbnails
│   │   └── preprocessed/            # 224x224 normalized tensors for ML inference
│   └── test_*.py                    # Comprehensive backend regression test suite
│
├── frontend/                        # React 19 Frontend Client
│   ├── Dockerfile                   # Multi-stage build (Node 20 Alpine -> Nginx 1.27 Alpine)
│   ├── package.json                 # Node dependencies and UI build scripts
│   ├── package-lock.json            # Deterministic lockfile
│   ├── vite.config.js               # Vite bundler configuration and reverse proxy
│   ├── index.html                   # HTML entrypoint
│   └── src/                         # React UI Source Code
│       ├── App.jsx                  # Main application state and view router
│       ├── api.js                   # Axios/Fetch client with JWT interceptors
│       ├── components/              # Modular UI components
│       │   ├── Navbar.jsx           # Role-aware navigation bar
│       │   ├── Dashboard.jsx        # Clinical analytics and case summaries
│       │   ├── CaseList.jsx         # Case triage queue and filtering
│       │   ├── CaseDetail.jsx       # Detailed case view with AI & explainability maps
│       │   ├── CreateCaseAndUpload.jsx # Case creation form with drag-and-drop IQA upload
│       │   ├── AdminCliniciansView.jsx # Admin portal for clinician credential approvals
│       │   └── ClinicalGuidelines.jsx  # Diagnostic reference guidelines
│       └── assets/                  # Static design graphics and logos
│
├── ml_models/                       # Machine Learning Architectures & Weights
│   ├── advanced_cnn.py              # EfficientNetB0 custom head definition
│   ├── config.py                    # Class taxonomy, hyperparameter configurations
│   ├── dataset_pipeline.py          # Data augmentation and K-fold generator
│   ├── model.py                     # MobileNetV2 architecture definition
│   ├── vision_transformer.py        # ViT-16 patch tokenization and transformer encoder
│   ├── train_mobilenet.py           # Two-phase training script
│   ├── data_splits/                 # Patient-stratified split manifests (CSV)
│   └── saved_models/                # Production model checkpoints & registry (<20MB)
│       ├── efficientnetb0_skin_lesion.keras
│       ├── mobilenetv2_skin_lesion_best.keras
│       ├── phase1_best_model.keras
│       ├── vit_skin_lesion_exp2_best.keras
│       ├── model_metadata.json
│       └── model_registry.json
│
├── ml_dataset/                      # Dataset Manifests & Documentation
│   ├── dataset_info.json            # HAM10000 taxonomy statistics
│   ├── eda_summary.json             # Exploratory Data Analysis distributions
│   └── download_instructions.md     # Guide for downloading raw 2.6+ GB images
│
└── reports/                         # Research & Verification Evidence
    ├── benchmarks/                  # System latency and concurrency benchmarks
    ├── model_evaluation/            # ROC curves, confusion matrices, performance tables
    ├── hyperparameter_optimization/ # Multi-configuration grid search results
    └── testing/                     # Security verification and regression test logs
```

---

## 12. Important Limitations & AI Medical Disclaimer

> [!WARNING]
> ### AI Medical Device & Decision-Support Disclaimer
> **DermAI is strictly an experimental clinical decision-support and research software tool.**
>
> 1. **Not an Autonomous Diagnostic Authority:** Output class probabilities, diagnostic risk categorizations, and visual activation heatmaps generated by DermAI do not constitute a definitive medical diagnosis.
> 2. **Clinical Correlation Required:** DermAI outputs must always be reviewed, interpreted, and validated by a board-certified dermatologist or qualified healthcare practitioner in conjunction with histopathological biopsy, full patient clinical history, and physical examination.
> 3. **Uncalibrated Output Probabilities:** Softmax output scores represent relative neural network feature distances across diagnostic classes, not absolute epidemiological probabilities of malignancy.
> 4. **Demographic & Capture Variabilities:** The underlying models were trained on the HAM10000 dataset, which primarily represents fair-skinned demographic populations (Fitzpatrick Skin Types I–III) captured via contact polarized/non-polarized dermoscopy. Performance may degrade on darker skin types (Fitzpatrick IV–VI) or low-resolution non-dermoscopic smartphone camera captures.
> 5. **Regulatory Status:** This software is not cleared, approved, or evaluated by the U.S. FDA, European EMA, or equivalent medical device regulatory bodies for independent clinical diagnosis.
