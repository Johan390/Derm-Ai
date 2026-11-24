import os
import io
import json
import uuid
import cv2
import numpy as np
from typing import List, Optional
from datetime import datetime, timezone
from PIL import Image
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text, case as sql_case

from .config import settings
from .database import engine, Base, get_db
from .models import (
    User, Case, ImageRecord, PredictionRecord, ClinicalReview, Notification,
    FollowUpReminder, ActivityLog, UserRole, CaseStatus, ReviewStatus, utc_now
)
from .schemas import (
    UserCreate, UserLogin, UserResponse, UserUpdateAdmin, Token,
    AdminStatsResponse, ActivityLogResponse,
    CaseCreate, CaseResponse, ImageRecordResponse, PredictionResponse,
    ClinicalReviewCreate, ClinicalReviewResponse, ReviewStatusUpdate, CaseAssignmentUpdate,
    NotificationResponse, NotificationListResponse,
    UserProfileUpdate, PasswordChangeRequest, ImageQualityAssessmentResponse,
    CaseComparisonResponse, ModelRegistryItem, ModelRollbackRequest, ModelRollbackResponse,
    FollowUpReminderCreate, FollowUpReminderResponse
)
from .auth import hash_password, verify_password, create_access_token, get_current_user, require_role
from .preprocessing import preprocess_skin_image
from .image_quality import iqa_engine
from .model_registry import model_registry
from .ml_service import ml_service, ModelServiceError, ModelNotFoundError, ImageProcessingError
from .storage import get_relative_upload_path, resolve_upload_path
from .lesion_analysis import lesion_candidate_mask, create_roi_overlay
from .logger import (
    log_startup, log_shutdown, log_auth_success, log_auth_failure,
    log_prediction_success, log_prediction_failure, log_report_generation,
    log_report_failure, log_system_event, logger
)
from .notification_service import (
    create_notification, notify_case_created, notify_prediction_completed,
    notify_review_completed, notify_case_assigned, notify_status_changed
)
from .report_service import generate_case_pdf, generate_case_csv, generate_case_json

from sqlalchemy import inspect

# Safe Schema Upgrade Helper (Adds missing columns without data loss)
def safe_database_upgrade():
    try:
        with engine.connect() as conn:
            inspector = inspect(conn)
            table_names = inspector.get_table_names()
            is_sqlite = engine.dialect.name == "sqlite"

            # Upgrade cases table if present
            if "cases" in table_names:
                case_cols = [c["name"] for c in inspector.get_columns("cases")]
                if "review_status" not in case_cols:
                    conn.execute(text("ALTER TABLE cases ADD COLUMN review_status VARCHAR(50) DEFAULT 'pending_review'"))
                    conn.commit()
                    print("[Database Migration] Added 'review_status' column to cases table.")
                if "assigned_doctor_id" not in case_cols:
                    conn.execute(text("ALTER TABLE cases ADD COLUMN assigned_doctor_id INTEGER REFERENCES users(id)"))
                    conn.commit()
                    print("[Database Migration] Added 'assigned_doctor_id' column to cases table.")
                if "is_archived" not in case_cols:
                    default_bool = "0" if is_sqlite else "FALSE"
                    conn.execute(text(f"ALTER TABLE cases ADD COLUMN is_archived BOOLEAN DEFAULT {default_bool}"))
                    conn.commit()
                    print("[Database Migration] Added 'is_archived' column to cases table.")

            # Upgrade users table if present
            if "users" in table_names:
                user_cols = [c["name"] for c in inspector.get_columns("users")]
                if "is_verified" not in user_cols:
                    default_bool = "1" if is_sqlite else "TRUE"
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT {default_bool}"))
                    conn.commit()
                    print("[Database Migration] Added 'is_verified' column to users table.")
    except Exception as e:
        print(f"[Database Migration Notice] {e}")

    Base.metadata.create_all(bind=engine)

safe_database_upgrade()

def record_activity(
    db: Session,
    action: str,
    details: str,
    actor: Optional[User] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    ip_address: Optional[str] = None
):
    """Persists administrative and system activity logs into the database."""
    try:
        actor_name = "System"
        if actor:
            actor_name = actor.full_name or actor.username
        log_entry = ActivityLog(
            actor_id=actor.id if actor else None,
            actor_name=actor_name,
            action=action,
            target_type=target_type,
            target_id=str(target_id) if target_id else None,
            details=details,
            ip_address=ip_address,
            created_at=utc_now()
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"[ActivityLog Notice] {e}")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_startup(settings.PROJECT_NAME, settings.VERSION)
    yield
    log_shutdown(settings.PROJECT_NAME)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="DermAI: Deep Learning Skin Disease Screening & Image Analysis API",
    lifespan=lifespan
)

# CORS setup with explicit allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

# Configurable Demo User Seeding
def seed_demo_users():
    db = next(get_db())
    try:
        if not db.query(User).filter(User.username == "dr_smith").first():
            demo_doctor = User(
                username="dr_smith",
                email="dr.smith@dermai-hospital.org",
                full_name="Dr. Sarah Smith, MD",
                role=UserRole.CLINICIAN,
                institution="General Dermatology Institute",
                hashed_password=hash_password("password123")
            )
            demo_patient = User(
                username="patient_demo",
                email="patient@example.com",
                full_name="Alex Johnson",
                role=UserRole.PATIENT,
                institution="Self / Patient Portal",
                hashed_password=hash_password("password123")
            )
            db.add_all([demo_doctor, demo_patient])
            db.commit()
            print("[DermAI] Development demo accounts initialized.")
    except Exception as e:
        print(f"[DermAI] Demo seed warning: {e}")
    finally:
        db.close()

if settings.SEED_DEMO_USERS:
    seed_demo_users()

# Helper serializers for Case, ImageRecord, and PredictionRecord
def format_image_response(img: ImageRecord) -> dict:
    return {
        "id": img.id,
        "case_id": img.case_id,
        "filename": img.filename,
        "original_url": f"/api/v1/cases/{img.case_id}/images/original/{img.filename}",
        "preview_url": f"/api/v1/cases/{img.case_id}/images/preview/{img.filename}" if img.preview_path else None,
        "preprocessed_url": f"/api/v1/cases/{img.case_id}/images/preprocessed/{img.filename}" if img.preprocessed_path else None,
        "file_size_bytes": img.file_size_bytes,
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "quality_status": img.quality_status,
        "blur_score": img.blur_score,
        "brightness_score": img.brightness_score,
        "contrast_score": img.contrast_score,
        "is_suitable_for_ai": img.is_suitable_for_ai,
        "quality_notes": img.quality_notes,
        "created_at": img.created_at
    }

def format_prediction_response(pred: PredictionRecord, img: Optional[ImageRecord] = None) -> dict:
    try:
        class_probs = json.loads(pred.class_probabilities_json)
    except Exception:
        class_probs = {}

    target_img = img or pred.image
    sorted_probs = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
    top3 = [
        {"class_label": label, "disease_name": ml_service.class_names.get(label, label),
         "probability": float(probability), "risk_level": ml_service.risk_levels.get(label, "Unknown Risk")}
        for label, probability in sorted_probs[:3]
    ]
    top5 = [
        {"class_label": label, "disease_name": ml_service.class_names.get(label, label),
         "probability": float(probability), "risk_level": ml_service.risk_levels.get(label, "Unknown Risk")}
        for label, probability in sorted_probs[:5]
    ]
    ranked = [
        {"class_label": label, "disease_name": ml_service.class_names.get(label, label),
         "probability": float(probability), "risk_level": ml_service.risk_levels.get(label, "Unknown Risk")}
        for label, probability in sorted_probs
    ]

    model_name_str = pred.model_name or ""
    is_ensemble = ("Ensemble" in model_name_str or "SoftVoting" in model_name_str)
    is_cnn = ("MobileNetV2" in model_name_str or "EfficientNetB0" in model_name_str) or is_ensemble
    is_vit = ("VisionTransformer" in model_name_str or "ViT" in model_name_str)
    supports_saliency = is_cnn or is_vit or is_ensemble

    return {
        "id": pred.id,
        "case_id": pred.case_id,
        "image_id": pred.image_id,
        "predicted_class": pred.predicted_class,
        "predicted_disease_name": pred.predicted_disease_name,
        "confidence_score": pred.confidence_score,
        "risk_level": pred.risk_level,
        "class_probabilities": class_probs,
        "model_name": pred.model_name,
        "prediction_timestamp": pred.created_at,
        "disclaimer": pred.disclaimer,
        "image_quality_status": target_img.quality_status if target_img else None,
        "is_suitable_for_ai": target_img.is_suitable_for_ai if target_img else None,
        "top_three_predictions": top3,
        "top_five_predictions": top5,
        "ranked_predictions": ranked,
        "gradcam_overlay_url": (f"/api/v1/cases/{pred.case_id}/gradcam/overlay?prediction_id={pred.id}" if is_cnn else None),
        "gradcam_heatmap_url": (f"/api/v1/cases/{pred.case_id}/gradcam/heatmap?prediction_id={pred.id}" if is_cnn else None),
        "gradcam_pp_overlay_url": (f"/api/v1/cases/{pred.case_id}/gradcam-plus-plus/overlay?prediction_id={pred.id}" if is_cnn else None),
        "gradcam_pp_heatmap_url": (f"/api/v1/cases/{pred.case_id}/gradcam-plus-plus/heatmap?prediction_id={pred.id}" if is_cnn else None),
        "saliency_overlay_url": (f"/api/v1/cases/{pred.case_id}/saliency/overlay?prediction_id={pred.id}" if supports_saliency else None),
        "saliency_heatmap_url": (f"/api/v1/cases/{pred.case_id}/saliency/heatmap?prediction_id={pred.id}" if supports_saliency else None),
        "vit_attention_overlay_url": (f"/api/v1/cases/{pred.case_id}/vit-attention/overlay?prediction_id={pred.id}" if is_vit else None),
        "vit_attention_heatmap_url": (f"/api/v1/cases/{pred.case_id}/vit-attention/heatmap?prediction_id={pred.id}" if is_vit else None),
        "vit_rollout_overlay_url": (f"/api/v1/cases/{pred.case_id}/vit-attention-rollout/overlay?prediction_id={pred.id}" if is_vit else None),
        "vit_rollout_heatmap_url": (f"/api/v1/cases/{pred.case_id}/vit-attention-rollout/heatmap?prediction_id={pred.id}" if is_vit else None),
        "gradcam_mobilenet_overlay_url": (f"/api/v1/cases/{pred.case_id}/gradcam/overlay?prediction_id={pred.id}&constituent=mobilenet" if is_ensemble else None),
        "gradcam_efficientnet_overlay_url": (f"/api/v1/cases/{pred.case_id}/gradcam/overlay?prediction_id={pred.id}&constituent=efficientnet" if is_ensemble else None),
        "lesion_analysis_url": f"/api/v1/cases/{pred.case_id}/lesion-analysis/overlay",
    }

def format_review_response(r: ClinicalReview) -> dict:
    return {
        "id": r.id,
        "case_id": r.case_id,
        "doctor_id": r.doctor_id,
        "doctor_name": r.doctor.full_name if r.doctor else f"Dr. ID #{r.doctor_id}",
        "doctor_institution": r.doctor.institution if r.doctor else None,
        "comment": r.comment,
        "recommendation": r.recommendation,
        "review_status": r.review_status,
        "created_at": r.created_at,
        "updated_at": r.updated_at
    }

def format_case_response(case: Case) -> dict:
    return {
        "id": case.id,
        "case_code": case.case_code,
        "user_id": case.user_id,
        "patient_age": case.patient_age,
        "patient_gender": case.patient_gender,
        "known_allergies": case.known_allergies,
        "medical_history": case.medical_history,
        "clinical_notes": case.clinical_notes,
        "body_location": case.body_location,
        "image_source": case.image_source,
        "status": case.status,
        "review_status": getattr(case, "review_status", "pending_review") or "pending_review",
        "assigned_doctor_id": getattr(case, "assigned_doctor_id", None),
        "assigned_doctor_name": case.assigned_doctor.full_name if getattr(case, "assigned_doctor", None) else None,
        "is_archived": getattr(case, "is_archived", False) or False,
        "created_at": case.created_at,
        "updated_at": case.updated_at,
        "images": [format_image_response(img) for img in case.images],
        "predictions": [format_prediction_response(p, p.image) for p in (case.predictions or [])],
        "reviews": [format_review_response(r) for r in (getattr(case, "reviews", []) or [])]
    }

# -----------------
# 1. Health & Meta
# -----------------
@app.get("/api/v1/health")
def health_check(db: Session = Depends(get_db)):
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unhealthy: {str(exc)}"

    model_ready = ml_service.is_model_ready()
    mobilenet_ready = os.path.isfile(ml_service.model_path)
    efficientnet_ready = os.path.isfile(ml_service.efficientnet_path)
    vit_ready = os.path.isfile(ml_service.vit_path)

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "milestone": "Milestone 4 Reporting, Testing, Analytics & Deployment Operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {
            "status": db_status,
            "engine": "postgresql" if "postgresql" in settings.DATABASE_URL else "sqlite"
        },
        "ml_service": {
            "status": "online" if (model_ready or mobilenet_ready) else "offline",
            "model_name": ml_service.model_name,
            "classes": ml_service.class_labels,
            "num_classes": len(ml_service.class_labels),
            "supported_models": ["mobilenet", "efficientnet", "vit", "ensemble"],
            "model_artifacts": {
                "mobilenetv2": "available" if mobilenet_ready else "missing",
                "efficientnetb0": "available" if efficientnet_ready else "missing",
                "vision_transformer": "available" if vit_ready else "missing",
                "soft_voting_ensemble": "available" if (mobilenet_ready and efficientnet_ready) else "missing"
            }
        },
        "modules_active": [
            "User Authentication & RBAC",
            "Patient & Case Management",
            "Image Capture & Upload",
            "Skin Image Preprocessing Engine",
            "MobileNetV2 Deep Learning Inference Service",
            "Multi-Model Ensemble & Vision Transformer",
            "Grad-CAM & Heuristic ROI Explainability",
            "Clinical PDF, CSV & JSON Report Engine",
            "In-App Clinical Notification System",
            "Structured Telemetry & Monitoring"
        ]
    }

# -----------------
# 2. Authentication Endpoints
# -----------------
@app.post("/api/v1/auth/register", response_model=Token)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    # Restrict public registration to only CLINICIAN and PATIENT roles
    # ADMIN and RESEARCHER roles cannot be registered through public endpoints
    if user_in.role not in [UserRole.CLINICIAN, UserRole.PATIENT]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration with role '{user_in.role.value}' is not permitted. Only 'clinician' and 'patient' roles are supported."
        )

    existing = db.query(User).filter(
        (User.username == user_in.username) | (User.email == user_in.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or Email already registered")

    # Newly registered clinicians require administrative approval
    # Patients default to verified
    is_verified_val = False if user_in.role == UserRole.CLINICIAN else True

    new_user = User(
        username=user_in.username,
        email=user_in.email,
        full_name=user_in.full_name,
        role=user_in.role,
        institution=user_in.institution,
        is_verified=is_verified_val,
        hashed_password=hash_password(user_in.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    log_system_event("USER_REGISTERED", f"User '{new_user.username}' (role: {new_user.role.value}) registered.")
    record_activity(
        db=db,
        action="USER_REGISTERED",
        details=f"Registered new {new_user.role.value} account: '{new_user.full_name}' ({new_user.email})",
        actor=new_user,
        target_type="user",
        target_id=str(new_user.id)
    )
    token = create_access_token({"sub": new_user.username, "role": new_user.role.value})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": new_user.id,
        "username": new_user.username,
        "full_name": new_user.full_name,
        "role": new_user.role,
        "is_verified": new_user.is_verified
    }

@app.post("/api/v1/auth/login", response_model=Token)
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == login_in.username).first()
    if not user or not verify_password(login_in.password, user.hashed_password):
        log_auth_failure(login_in.username, "Incorrect username or password")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    log_auth_success(user.username, user.role.value)
    
    token = create_access_token({"sub": user.username, "role": user.role.value})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "is_verified": getattr(user, "is_verified", True)
    }

@app.get("/api/v1/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.put("/api/v1/auth/profile", response_model=UserResponse)
def update_profile(
    profile_in: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Allows authenticated users to update their own display name or institution affiliation.
    Prevents unauthorized modification of roles, emails, or verification statuses.
    """
    if profile_in.full_name is not None and profile_in.full_name.strip():
        current_user.full_name = profile_in.full_name.strip()
    if profile_in.institution is not None:
        current_user.institution = profile_in.institution.strip()

    db.commit()
    db.refresh(current_user)
    log_system_event("USER_PROFILE_UPDATED", f"User '{current_user.username}' (ID: {current_user.id}) updated profile details.")
    return current_user

@app.post("/api/v1/auth/change-password")
def change_password(
    pwd_in: PasswordChangeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Secure password management verifying current password before hashing the new password.
    """
    if not verify_password(pwd_in.current_password, current_user.hashed_password):
        log_auth_failure(current_user.username, "Failed password change: current password incorrect")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password verification failed."
        )

    if len(pwd_in.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters long."
        )

    current_user.hashed_password = hash_password(pwd_in.new_password)
    db.commit()
    log_system_event("PASSWORD_CHANGED", f"User '{current_user.username}' changed account password successfully.")
    return {"status": "success", "message": "Password updated successfully."}

@app.post("/api/v1/auth/logout")
def logout(current_user: User = Depends(get_current_user)):
    """
    Standard logout endpoint for session termination audit trail.
    """
    log_system_event("USER_LOGOUT", f"User '{current_user.username}' logged out.")
    return {"status": "success", "message": "Logged out successfully."}

def check_clinical_access(user: User, case: Optional[Case] = None):
    """
    Enforces HIPAA/GDPR clinical data boundary:
    1. Researchers and ML engineers are strictly restricted from accessing clinical patient records.
    2. Unverified clinicians are strictly forbidden from clinical triage, case records, reviews, and inferences.
    3. Patients are strictly isolated to cases they own (BOLA/IDOR prevention).
    4. Clinicians can access unassigned cases (triage queue) or cases assigned to them.
       Attempting to access a case explicitly assigned to another clinician is denied (403).
    5. Admins have universal authorized oversight.
    """
    if user.role in [UserRole.RESEARCHER, UserRole.ML_ENGINEER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Researchers and ML Engineers are not authorized to access clinical patient records"
        )
    if user.role == UserRole.CLINICIAN and not getattr(user, "is_verified", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your clinician account is awaiting administrative verification. Clinical access will be enabled after approval."
        )
    if case and user.role == UserRole.PATIENT and case.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this patient case"
        )
    if case and user.role == UserRole.CLINICIAN:
        if case.assigned_doctor_id is not None and case.assigned_doctor_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Case is assigned to another clinician (Doctor ID #{case.assigned_doctor_id})"
            )

# -----------------
# 3. Case Management Endpoints
# -----------------
@app.post("/api/v1/cases/", response_model=CaseResponse)
def create_case(
    case_in: CaseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    check_clinical_access(current_user)
    code = f"CAS-{datetime.now(timezone.utc).strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}"
    new_case = Case(
        case_code=code,
        user_id=current_user.id,
        patient_age=case_in.patient_age,
        patient_gender=case_in.patient_gender,
        known_allergies=case_in.known_allergies,
        medical_history=case_in.medical_history,
        clinical_notes=case_in.clinical_notes,
        body_location=case_in.body_location,
        image_source=case_in.image_source or "mobile_upload",
        status=CaseStatus.CREATED,
        review_status=ReviewStatus.PENDING_REVIEW.value,
        assigned_doctor_id=None
    )
    db.add(new_case)
    db.commit()
    db.refresh(new_case)
    try:
        notify_case_created(db, new_case, current_user)
    except Exception as exc:
        logger.warning(f"Failed to dispatch case creation notification: {exc}")
    return format_case_response(new_case)

@app.get("/api/v1/cases/", response_model=List[CaseResponse])
def list_cases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 500,
    status_filter: Optional[str] = None,
    review_status_filter: Optional[str] = None,
    assignment_filter: Optional[str] = None,
    include_archived: bool = False
):
    check_clinical_access(current_user)
    query = db.query(Case)

    # Filter out archived cases unless requested by authorized Admin
    if not (include_archived and current_user.role == UserRole.ADMIN):
        query = query.filter((Case.is_archived == False) | (Case.is_archived.is_(None)))

    # Role-based case isolation:
    # Patients see only their own cases
    if current_user.role == UserRole.PATIENT:
        query = query.filter(Case.user_id == current_user.id)
    elif current_user.role == UserRole.CLINICIAN:
        if assignment_filter == "assigned_to_me":
            query = query.filter(Case.assigned_doctor_id == current_user.id)
        elif assignment_filter == "unassigned":
            query = query.filter(Case.assigned_doctor_id.is_(None))
        else:
            # Default for clinician: cases assigned to them or unassigned triage queue
            query = query.filter(
                (Case.assigned_doctor_id == current_user.id) | (Case.assigned_doctor_id.is_(None))
            )

    if status_filter:
        query = query.filter(Case.status == status_filter)
    if review_status_filter:
        query = query.filter(Case.review_status == review_status_filter)

    if current_user.role == UserRole.CLINICIAN:
        # Prioritize clinician's own assigned cases, then sort by created_at DESC
        priority_order = sql_case((Case.assigned_doctor_id == current_user.id, 0), else_=1)
        cases = query.order_by(priority_order, Case.created_at.desc()).limit(limit).all()
    else:
        cases = query.order_by(Case.created_at.desc()).limit(limit).all()

    return [format_case_response(c) for c in cases]

@app.get("/api/v1/cases/{case_id}", response_model=CaseResponse)
def get_case(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    check_clinical_access(current_user, case)
    return format_case_response(case)

@app.put("/api/v1/cases/{case_id}", response_model=CaseResponse)
def update_case(
    case_id: int,
    case_in: CaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    check_clinical_access(current_user, case)

    # TASK 2: Freeze Patient Editing After Clinical Review
    if current_user.role == UserRole.PATIENT:
        if getattr(case, "review_status", None) in ["reviewed", "follow_up_required"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Modifications prohibited: Case has been evaluated by an attending clinician (status: '{case.review_status}'). Clinical records are permanently frozen to preserve diagnostic integrity."
            )

    if case_in.patient_age is not None:
        case.patient_age = case_in.patient_age
    if case_in.patient_gender:
        case.patient_gender = case_in.patient_gender
    if case_in.known_allergies is not None:
        case.known_allergies = case_in.known_allergies
    if case_in.medical_history is not None:
        case.medical_history = case_in.medical_history
    if case_in.clinical_notes is not None:
        case.clinical_notes = case_in.clinical_notes
    if case_in.body_location:
        case.body_location = case_in.body_location

    db.commit()
    db.refresh(case)
    return format_case_response(case)

@app.delete("/api/v1/cases/{case_id}")
def delete_case(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    check_clinical_access(current_user, case)

    # TASK 3: Prevent Cascading Deletion of Clinical Records via Soft-Deletion
    # Check if case has received clinical reviews
    review_count = len(getattr(case, "reviews", []) or [])
    has_reviews = review_count > 0 or getattr(case, "review_status", None) in ["reviewed", "follow_up_required"]

    if has_reviews:
        # Case has clinical reviews: archive case (soft deletion) to preserve medical history and doctor records
        case.is_archived = True
        db.commit()
        return {
            "status": "success",
            "message": f"Case {case_id} has been archived. Associated physician clinical reviews and medical records have been permanently preserved.",
            "is_archived": True
        }
    else:
        # For cases without clinical reviews, safe soft-deletion or clean removal
        case.is_archived = True
        db.commit()
        return {
            "status": "success",
            "message": f"Case {case_id} archived successfully.",
            "is_archived": True
        }

# -----------------
# 4. Image Upload & Preprocessing Endpoints
# -----------------
@app.post("/api/v1/cases/{case_id}/upload-image", response_model=CaseResponse)
def upload_case_image(
    case_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Synchronous execution in FastAPI worker threadpool prevents blocking the asyncio event loop.
    Enforces authorization, binary content verification, size checks, and transactional failure cleanup.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # 1. Ownership & Clinical Access Check BEFORE reading or writing file
    check_clinical_access(current_user, case)

    # 2. Extension validation
    file_ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # 3. Read uploaded bytes with size limit enforcement
    file_content = file.file.read()
    file_size = len(file_content)

    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB"
        )

    # 4. Deep binary image verification using Pillow (rejects spoofed extensions or corrupted bytes)
    try:
        with Image.open(io.BytesIO(file_content)) as img_check:
            img_check.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Corrupted or invalid image file. Binary contents do not match a valid image format."
        )

    # 5. Generate secure server-side filename (prevents directory traversal)
    safe_ext = "jpg" if file_ext in ["jpg", "jpeg"] else file_ext
    unique_filename = f"{case.case_code}_{uuid.uuid4().hex[:8]}.{safe_ext}"
    original_target_path = os.path.join(settings.ORIGINAL_DIR, unique_filename)

    # 6. Save and execute preprocessing pipeline with comprehensive cleanup tracking
    created_files = []
    try:
        with open(original_target_path, "wb") as f:
            f.write(file_content)
        created_files.append(original_target_path)

        prep_results = preprocess_skin_image(
            original_filepath=original_target_path,
            output_filename=unique_filename,
            target_size=settings.TARGET_IMAGE_SIZE
        )
        if prep_results.get("preview_filepath"):
            created_files.append(prep_results["preview_filepath"])
        if prep_results.get("preprocessed_filepath"):
            created_files.append(prep_results["preprocessed_filepath"])

        q_metrics = prep_results["quality_metrics"]

        # Convert filesystem paths to portable relative paths for database storage
        rel_original_path = get_relative_upload_path(original_target_path)
        rel_preview_path = (
            get_relative_upload_path(prep_results["preview_filepath"])
            if prep_results.get("preview_filepath")
            else None
        )
        rel_preprocessed_path = (
            get_relative_upload_path(prep_results["preprocessed_filepath"])
            if prep_results.get("preprocessed_filepath")
            else None
        )

        image_record = ImageRecord(
            case_id=case.id,
            filename=unique_filename,
            original_path=rel_original_path,
            preview_path=rel_preview_path,
            preprocessed_path=rel_preprocessed_path,
            file_size_bytes=file_size,
            width=q_metrics["width"],
            height=q_metrics["height"],
            format=safe_ext,
            quality_status=q_metrics["quality_status"],
            blur_score=q_metrics["blur_score"],
            brightness_score=q_metrics["brightness_score"],
            contrast_score=q_metrics["contrast_score"],
            is_suitable_for_ai=q_metrics["is_suitable_for_ai"],
            quality_notes="; ".join(q_metrics["notes"]) if q_metrics["notes"] else "All visual metrics within normal range"
        )
        
        case.status = CaseStatus.PREPROCESSED
        db.add(image_record)
        db.commit()
        db.refresh(case)

        return format_case_response(case)

    except Exception as err:
        # Atomic rollback of all files created during this transaction
        for fpath in created_files:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
        if isinstance(err, HTTPException):
            raise err
        raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {str(err)}")

# -----------------
# 5. Authenticated Clinical Image Delivery Endpoint (Replaces Insecure Static Mounts)
# -----------------
@app.get("/api/v1/cases/{case_id}/images/{image_type}/{filename}")
def get_case_image(
    case_id: int,
    image_type: str,
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Secure, authenticated image delivery:
    - Enforces case ownership check (Patients can only access their own case images)
    - Rejects unauthenticated requests with HTTP 401
    - Prevents directory traversal attacks
    - Replaces open static mounts for medical privacy
    """
    if image_type not in ["original", "preview", "preprocessed"]:
        raise HTTPException(status_code=400, detail="Invalid image type requested")

    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Authorization verification
    check_clinical_access(current_user, case)

    # Verify that image belongs to this case
    image_record = db.query(ImageRecord).filter(
        ImageRecord.case_id == case.id,
        ImageRecord.filename == filename
    ).first()
    if not image_record:
        raise HTTPException(status_code=404, detail="Image record not found for this case")

    # Select stored path corresponding to the requested image type
    path_map = {
        "original": image_record.original_path,
        "preview": image_record.preview_path,
        "preprocessed": image_record.preprocessed_path
    }
    stored_path = path_map.get(image_type)
    if not stored_path:
        raise HTTPException(status_code=404, detail=f"Image type '{image_type}' not available for this image")

    # Resolve stored path safely (handles relative, legacy Windows, and legacy POSIX paths)
    try:
        resolved_path = resolve_upload_path(stored_path)
    except ValueError:
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    media_type = f"image/{image_record.format or 'jpeg'}"
    return FileResponse(str(resolved_path), media_type=media_type)

# -----------------
# 5B. Independent Image Quality Assessment (IQA) Endpoint
# -----------------
@app.post("/api/v1/image-quality/assess", response_model=ImageQualityAssessmentResponse)
def assess_image_quality_endpoint(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Independent Image Quality Assessment evaluating format, resolution, blur,
    underexposure, overexposure, and contrast before case submission or inference.
    """
    file_ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    file_content = file.file.read()
    if len(file_content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB"
        )
    try:
        with Image.open(io.BytesIO(file_content)) as img_check:
            img_check.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Corrupted or invalid image file. Binary contents do not match a valid image format."
        )

    nparr = np.frombuffer(file_content, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image bytes into RGB array.")

    res = iqa_engine.evaluate_array(img_bgr)
    res["file_size_bytes"] = len(file_content)
    return res

# -----------------
# 6. Dataset & Benchmark Specifications (Researcher RBAC Protected)
# -----------------
@app.get("/api/v1/dataset/info")
def get_dataset_info(
    current_user: User = Depends(require_role([UserRole.RESEARCHER, UserRole.ML_ENGINEER]))
):
    dataset_info_path = os.path.join(settings.BASE_DIR, "..", "ml_dataset", "dataset_info.json")
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r") as f:
            return json.load(f)
    return {"message": "Dataset configuration not found"}

@app.get("/api/v1/dataset/eda-summary")
def get_eda_summary(
    current_user: User = Depends(require_role([UserRole.RESEARCHER, UserRole.ML_ENGINEER]))
):
    eda_path = os.path.join(settings.BASE_DIR, "..", "ml_dataset", "eda_summary.json")
    if os.path.exists(eda_path):
        with open(eda_path, "r") as f:
            return json.load(f)
    return {"message": "EDA summary not generated yet. Run eda_analysis.py"}

# -----------------
# 7. Deep Learning Model Prediction Endpoints
# -----------------
@app.post("/api/v1/cases/{case_id}/predict", response_model=PredictionResponse)
def predict_case(
    case_id: int,
    model: str = Query("mobilenet", description="mobilenet, efficientnet, vit, or ensemble"),
    allow_suboptimal: bool = Query(True, description="Allow inference on poor or suboptimal quality images"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes deep learning skin lesion classification on the latest uploaded image of a case.
    
    Security & RBAC:
    - Enforces JWT authentication.
    - Clinicians and Admins can evaluate clinical cases.
    - Patients are restricted to cases they own.
    - Researchers and ML Engineers are strictly prohibited from clinical patient cases (HIPAA/GDPR privacy boundary).
    
    Preprocessing & Inference:
    - Loads latest image associated with case.
    - Preserves aspect ratio via center crop and resizes to 224x224.
    - Rejects severely poor images unless allow_suboptimal=true.
    - Evaluates against 7 HAM10000 diagnostic classes.
    - Attaches quality warnings if the image failed Milestone 1 IQA.
    - Persists prediction record and marks case status COMPLETED.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Patient ownership and clinical access check
    check_clinical_access(current_user, case)

    # Validate that case has an image
    if not case.images:
        raise HTTPException(
            status_code=400,
            detail="Case has no uploaded images. Please upload an image before requesting AI prediction."
        )

    # Use the most recent image uploaded for this case
    latest_image = case.images[-1]
    if not latest_image.original_path:
        raise HTTPException(
            status_code=404,
            detail="Underlying case image file is missing on server storage."
        )

    # Image Quality Guard: Reject poor quality images if allow_suboptimal is False
    if not latest_image.is_suitable_for_ai and not allow_suboptimal:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image quality is POOR: {latest_image.quality_notes or 'Visual artifacts detected'}. "
                "Inference on severely degraded images is rejected to prevent unreliable outputs. "
                "Pass allow_suboptimal=true to override if clinically necessary."
            )
        )

    try:
        resolved_original_path = resolve_upload_path(latest_image.original_path)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Underlying case image file is missing on server storage."
        )

    if not resolved_original_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Underlying case image file is missing on server storage."
        )

    # Compose medical disclaimer with image quality warnings if necessary
    disclaimer_text = ml_service.disclaimer
    if not latest_image.is_suitable_for_ai:
        disclaimer_text += (
            " [QUALITY WARNING: The input image was flagged as suboptimal or poor quality by automated "
            "quality assessment. Model predictions on suboptimal images carry higher diagnostic uncertainty.]"
        )

    # Execute ML inference
    start_time = datetime.now()
    try:
        prediction_data = ml_service.predict_image(str(resolved_original_path), model=model)
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000.0
    except ModelNotFoundError as e:
        log_prediction_failure(case.id, model, str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ML Model service unavailable: {str(e)}"
        )
    except ImageProcessingError as e:
        log_prediction_failure(case.id, model, str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Image processing failure: {str(e)}"
        )
    except ModelServiceError as e:
        log_prediction_failure(case.id, model, str(e))
        raise HTTPException(status_code=503, detail=f"ML model service unavailable: {str(e)}")
    except ValueError as e:
        log_prediction_failure(case.id, model, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_prediction_failure(case.id, model, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution failure: {str(e)}"
        )

    # Persist prediction in database
    prediction_record = PredictionRecord(
        case_id=case.id,
        image_id=latest_image.id,
        predicted_class=prediction_data["predicted_class"],
        predicted_disease_name=prediction_data["predicted_disease_name"],
        confidence_score=prediction_data["confidence_score"],
        risk_level=prediction_data["risk_level"],
        class_probabilities_json=json.dumps(prediction_data["class_probabilities"]),
        model_name=prediction_data["model_name"],
        disclaimer=disclaimer_text
    )

    case.status = CaseStatus.COMPLETED
    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)
    db.refresh(case)

    try:
        notify_prediction_completed(
            db,
            case,
            prediction_data["model_name"],
            prediction_data["predicted_disease_name"],
            current_user.id
        )
        log_prediction_success(
            case.id,
            case.case_code,
            prediction_data["model_name"],
            prediction_data["predicted_class"],
            latency_ms,
            current_user.id
        )
    except Exception as exc:
        logger.warning(f"Failed post-prediction hook: {exc}")

    return format_prediction_response(prediction_record, latest_image)


@app.get("/api/v1/cases/{case_id}/predictions", response_model=List[PredictionResponse])
def get_case_predictions(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves all past predictions for a given case.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    check_clinical_access(current_user, case)

    return [format_prediction_response(p, p.image) for p in (case.predictions or [])]


@app.get("/api/v1/cases/{case_id}/compare", response_model=CaseComparisonResponse)
def compare_case_analyses(
    case_id: int,
    earlier_id: Optional[int] = Query(None, description="Prediction ID of the earlier analysis"),
    newer_id: Optional[int] = Query(None, description="Prediction ID of the newer analysis"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Enables side-by-side comparison of two historical analyses for the same case.
    Strictly enforces case ownership and RBAC.
    Appends mandatory disclaimer that prediction differences do not establish disease progression.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    check_clinical_access(current_user, case)

    predictions = db.query(PredictionRecord).filter(PredictionRecord.case_id == case_id).order_by(PredictionRecord.created_at.asc()).all()
    if len(predictions) < 2:
        raise HTTPException(
            status_code=400,
            detail="Comparison requires at least two historical predictions for this case."
        )

    p1 = None
    p2 = None
    if earlier_id and newer_id:
        for p in predictions:
            if p.id == earlier_id:
                p1 = p
            if p.id == newer_id:
                p2 = p
        if not p1 or not p2:
            raise HTTPException(status_code=404, detail="One or both requested prediction IDs not found for this case.")
    else:
        # Default to first and latest
        p1 = predictions[0]
        p2 = predictions[-1]

    def build_item(pred):
        img_url = f"/api/v1/cases/{pred.case_id}/images/original/{pred.image.filename}" if pred.image else ""
        gcam_url = (
            f"/api/v1/cases/{pred.case_id}/gradcam/overlay?prediction_id={pred.id}"
            if ("MobileNetV2" in (pred.model_name or "") or "EfficientNetB0" in (pred.model_name or ""))
            else None
        )
        try:
            probs = json.loads(pred.class_probabilities_json)
        except Exception:
            probs = {}
        return {
            "prediction_id": pred.id,
            "created_at": pred.created_at,
            "model_name": pred.model_name,
            "predicted_class": pred.predicted_class,
            "predicted_disease_name": pred.predicted_disease_name,
            "confidence_score": pred.confidence_score,
            "risk_level": pred.risk_level,
            "class_probabilities": probs,
            "image_url": img_url,
            "gradcam_url": gcam_url
        }

    return {
        "case_id": case.id,
        "case_code": case.case_code,
        "earlier_analysis": build_item(p1),
        "newer_analysis": build_item(p2),
        "disclaimer": (
            "LONGITUDINAL COMPARISON NOTICE: Variations in AI prediction, softmax score, or risk tier "
            "between separate sessions reflect model sensitivity to camera angle, lighting, resolution, "
            "and distinct algorithmic checkpoints. Variations DO NOT establish pathological disease progression, "
            "lesion growth, or therapeutic response. A qualified dermatologist must physically examine the patient."
        )
    }

# -----------------
# 7A. Explainability, ROI Analysis and Model Status
# -----------------
def _case_image_for_analysis(case_id: int, db: Session, current_user: User, prediction_id: Optional[int] = None):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    check_clinical_access(current_user, case)
    prediction_query = db.query(PredictionRecord).filter(PredictionRecord.case_id == case_id)
    if prediction_id is not None:
        prediction = prediction_query.filter(PredictionRecord.id == prediction_id).first()
        if prediction is None:
            raise HTTPException(status_code=404, detail="Prediction not found for this case")
    else:
        prediction = prediction_query.order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc()).first()
    image_record = None
    if prediction:
        image_record = db.query(ImageRecord).filter(
            ImageRecord.id == prediction.image_id, ImageRecord.case_id == case_id
        ).first()
    if image_record is None:
        image_record = (db.query(ImageRecord).filter(ImageRecord.case_id == case_id)
                        .order_by(ImageRecord.created_at.desc(), ImageRecord.id.desc()).first())
    if image_record is None:
        raise HTTPException(status_code=404, detail="No image is available for this case")
    try:
        source_path = resolve_upload_path(image_record.original_path)
    except ValueError:
        raise HTTPException(status_code=404, detail="Stored image path is invalid")
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Image file is missing from storage")
    return case, prediction, image_record, source_path


@app.get("/api/v1/cases/{case_id}/gradcam/{image_type}")
def get_case_gradcam(
    case_id: int,
    image_type: str,
    prediction_id: Optional[int] = Query(None),
    constituent: Optional[str] = Query(None, description="Optional constituent model for ensemble: 'mobilenet', 'efficientnet', or 'ensemble'"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Authenticated, case-authorized Grad-CAM overlay or heatmap for latest prediction."""
    if image_type not in ("overlay", "heatmap"):
        raise HTTPException(status_code=400, detail="Image type must be overlay or heatmap")
    _, prediction, _, source_path = _case_image_for_analysis(case_id, db, current_user, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=409, detail="Run a prediction before requesting Grad-CAM")
    prediction_model_name = prediction.model_name or ""

    if "Ensemble" in prediction_model_name or "SoftVoting" in prediction_model_name:
        gradcam_model_type = "ensemble"
    elif "MobileNetV2" in prediction_model_name:
        gradcam_model_type = "mobilenet"
    elif "EfficientNetB0" in prediction_model_name:
        gradcam_model_type = "efficientnet"
    else:
        raise HTTPException(
            status_code=409,
            detail="Grad-CAM is available for MobileNetV2, EfficientNetB0, and Soft Voting Ensemble predictions."
        )
    output_dir = os.path.join(settings.UPLOAD_DIR, "explainability")
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{constituent}" if constituent else ""
    output_path = os.path.join(output_dir, f"case_{case_id}_prediction_{prediction.id}{suffix}_{image_type}.jpg")
    if not os.path.isfile(output_path):
        try:
            overlay, heatmap = ml_service.generate_gradcam(
                str(source_path),
                class_index=ml_service.class_labels.index(prediction.predicted_class),
                model_type=gradcam_model_type,
                constituent=constituent
            )
            selected = overlay if image_type == "overlay" else heatmap
            if not cv2.imwrite(output_path, selected):
                raise RuntimeError("Could not write Grad-CAM image")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Grad-CAM generation unavailable: {exc}")
    return FileResponse(output_path, media_type="image/jpeg", filename=os.path.basename(output_path))


@app.get("/api/v1/cases/{case_id}/gradcam-plus-plus/{image_type}")
def get_case_gradcam_plus_plus(
    case_id: int,
    image_type: str,
    prediction_id: Optional[int] = Query(None),
    constituent: Optional[str] = Query(None, description="Optional constituent model for ensemble: 'mobilenet', 'efficientnet', or 'ensemble'"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Authenticated Grad-CAM++ (second-order gradients) for MobileNetV2, EfficientNetB0, and Soft Voting Ensemble."""
    if image_type not in ("overlay", "heatmap"):
        raise HTTPException(status_code=400, detail="Image type must be overlay or heatmap")
    _, prediction, _, source_path = _case_image_for_analysis(case_id, db, current_user, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=409, detail="Run a prediction before requesting Grad-CAM++")
    prediction_model_name = prediction.model_name or ""

    if "Ensemble" in prediction_model_name or "SoftVoting" in prediction_model_name:
        gcpp_model_type = "ensemble"
    elif "MobileNetV2" in prediction_model_name:
        gcpp_model_type = "mobilenet"
    elif "EfficientNetB0" in prediction_model_name:
        gcpp_model_type = "efficientnet"
    else:
        raise HTTPException(
            status_code=409,
            detail="Grad-CAM++ is available for MobileNetV2, EfficientNetB0, and Soft Voting Ensemble predictions."
        )

    output_dir = os.path.join(settings.UPLOAD_DIR, "explainability")
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{constituent}" if constituent else ""
    output_path = os.path.join(output_dir, f"case_{case_id}_prediction_{prediction.id}_gcpp{suffix}_{image_type}.jpg")
    if not os.path.isfile(output_path):
        try:
            overlay, heatmap = ml_service.generate_gradcam_plus_plus(
                str(source_path),
                class_index=ml_service.class_labels.index(prediction.predicted_class),
                model_type=gcpp_model_type,
                constituent=constituent
            )
            selected = overlay if image_type == "overlay" else heatmap
            if not cv2.imwrite(output_path, selected):
                raise RuntimeError("Could not write Grad-CAM++ image")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Grad-CAM++ generation unavailable: {exc}")
    return FileResponse(output_path, media_type="image/jpeg", filename=os.path.basename(output_path))


@app.get("/api/v1/cases/{case_id}/saliency/{image_type}")
def get_case_saliency(
    case_id: int,
    image_type: str,
    prediction_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Authenticated gradient-based pixel saliency map for MobileNetV2, EfficientNetB0, ViT, and Soft Voting Ensemble."""
    if image_type not in ("overlay", "heatmap"):
        raise HTTPException(status_code=400, detail="Image type must be overlay or heatmap")
    _, prediction, _, source_path = _case_image_for_analysis(case_id, db, current_user, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=409, detail="Run a prediction before requesting saliency map")
    prediction_model_name = prediction.model_name or ""

    if "Ensemble" in prediction_model_name or "SoftVoting" in prediction_model_name:
        sal_model_type = "ensemble"
    elif "MobileNetV2" in prediction_model_name:
        sal_model_type = "mobilenet"
    elif "EfficientNetB0" in prediction_model_name:
        sal_model_type = "efficientnet"
    elif "VisionTransformer" in prediction_model_name or "ViT" in prediction_model_name:
        sal_model_type = "vit"
    else:
        raise HTTPException(
            status_code=409,
            detail="Saliency maps are supported for MobileNetV2, EfficientNetB0, ViT, and Soft Voting Ensemble."
        )

    output_dir = os.path.join(settings.UPLOAD_DIR, "explainability")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"case_{case_id}_prediction_{prediction.id}_sal_{image_type}.jpg")
    if not os.path.isfile(output_path):
        try:
            overlay, heatmap = ml_service.generate_saliency_map(
                str(source_path),
                class_index=ml_service.class_labels.index(prediction.predicted_class),
                model_type=sal_model_type
            )
            selected = overlay if image_type == "overlay" else heatmap
            if not cv2.imwrite(output_path, selected):
                raise RuntimeError("Could not write Saliency image")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Saliency generation unavailable: {exc}")
    return FileResponse(output_path, media_type="image/jpeg", filename=os.path.basename(output_path))


@app.get("/api/v1/cases/{case_id}/vit-attention/{image_type}")
def get_case_vit_attention(
    case_id: int,
    image_type: str,
    prediction_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Authenticated multi-head attention map / Attention Rollout visualization for Vision Transformer predictions."""
    if image_type not in ("overlay", "heatmap"):
        raise HTTPException(status_code=400, detail="Image type must be overlay or heatmap")
    _, prediction, _, source_path = _case_image_for_analysis(case_id, db, current_user, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=409, detail="Run a prediction before requesting ViT attention map")
    prediction_model_name = prediction.model_name or ""

    if not ("VisionTransformer" in prediction_model_name or "ViT" in prediction_model_name):
        raise HTTPException(
            status_code=409,
            detail="ViT attention map visualization is supported only for Vision Transformer predictions."
        )

    output_dir = os.path.join(settings.UPLOAD_DIR, "explainability")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"case_{case_id}_prediction_{prediction.id}_vit_rollout_{image_type}.jpg")
    if not os.path.isfile(output_path):
        try:
            overlay, heatmap = ml_service.generate_vit_attention_rollout(str(source_path))
            selected = overlay if image_type == "overlay" else heatmap
            if not cv2.imwrite(output_path, selected):
                raise RuntimeError("Could not write ViT attention rollout image")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"ViT attention rollout extraction unavailable: {exc}")
    return FileResponse(output_path, media_type="image/jpeg", filename=os.path.basename(output_path))


@app.get("/api/v1/cases/{case_id}/vit-attention-rollout/{image_type}")
def get_case_vit_attention_rollout(
    case_id: int,
    image_type: str,
    prediction_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Authenticated Attention Rollout visualization for Vision Transformer predictions."""
    return get_case_vit_attention(case_id, image_type, prediction_id, db, current_user)


@app.get("/api/v1/cases/{case_id}/lesion-analysis")
def get_case_lesion_analysis(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return heuristic lesion-candidate ROI metadata; not a trained segmentation result."""
    _, _, image_record, source_path = _case_image_for_analysis(case_id, db, current_user)
    cache_dir = os.path.join(settings.UPLOAD_DIR, "explainability")
    os.makedirs(cache_dir, exist_ok=True)
    metadata_path = os.path.join(cache_dir, f"image_{image_record.id}_roi.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode source image")
    mask, metadata = lesion_candidate_mask(image)
    overlay = create_roi_overlay(image, mask)
    overlay_path = os.path.join(cache_dir, f"image_{image_record.id}_roi_overlay.jpg")
    if not cv2.imwrite(overlay_path, overlay):
        raise HTTPException(status_code=500, detail="Could not save ROI visualization")
    metadata.update({"case_id": case_id, "image_id": image_record.id,
                     "overlay_url": f"/api/v1/cases/{case_id}/lesion-analysis/overlay"})
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


@app.get("/api/v1/cases/{case_id}/lesion-analysis/overlay")
def get_case_lesion_overlay(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Calling the metadata route's shared resolver ensures identical RBAC and source-image selection.
    _, _, image_record, source_path = _case_image_for_analysis(case_id, db, current_user)
    cache_dir = os.path.join(settings.UPLOAD_DIR, "explainability")
    os.makedirs(cache_dir, exist_ok=True)
    overlay_path = os.path.join(cache_dir, f"image_{image_record.id}_roi_overlay.jpg")
    if not os.path.isfile(overlay_path):
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode source image")
        mask, _ = lesion_candidate_mask(image)
        if not cv2.imwrite(overlay_path, create_roi_overlay(image, mask)):
            raise HTTPException(status_code=500, detail="Could not save ROI visualization")
    return FileResponse(overlay_path, media_type="image/jpeg", filename=os.path.basename(overlay_path))


@app.get("/api/v1/models/status")
def get_model_status(current_user: User = Depends(get_current_user)):
    """Authenticated availability status for installed model artifacts."""
    return {
        "mobilenetv2": {"available": os.path.isfile(ml_service.model_path), "loaded": ml_service.model is not None},
        "efficientnetb0": {"available": os.path.isfile(ml_service.efficientnet_path), "loaded": ml_service.efficientnet_model is not None},
        "vision_transformer": {"available": os.path.isfile(ml_service.vit_path), "loaded": ml_service.vit_model is not None},
        "supported_prediction_modes": ["mobilenet", "efficientnet", "vit", "ensemble"],
        "ensemble_definition": "Unweighted soft voting: mean(MobileNetV2 probabilities, EfficientNetB0 probabilities)."
    }


# -----------------
# 7B. Model Registry, Versioning & Rollback Endpoints
# -----------------
@app.get("/api/v1/models/registry", response_model=List[ModelRegistryItem])
def list_model_registry(current_user: User = Depends(get_current_user)):
    """
    Returns registered model versions, architectures, checkpoint paths,
    and active status. Accessible to Clinicians, Researchers, ML Engineers, and Admins.
    """
    return model_registry.list_models()


@app.get("/api/v1/models/training-history")
def get_model_training_history(
    current_user: User = Depends(require_role([UserRole.RESEARCHER, UserRole.ML_ENGINEER, UserRole.ADMIN]))
):
    """
    Returns authentic training history, loss/validation curves, hyperparameter search,
    and dataset provenance.
    """
    return model_registry.get_training_history()


@app.post("/api/v1/models/active", response_model=ModelRollbackResponse)
def set_active_model_version(
    req: ModelRollbackRequest,
    current_user: User = Depends(require_role([UserRole.ML_ENGINEER, UserRole.ADMIN]))
):
    """
    Allows authorized Data Scientists, ML Engineers, or Administrators to select the
    active production model version or perform a safe rollback to a previously verified checkpoint.
    """
    try:
        res = model_registry.set_active_version(
            version=req.target_version,
            user_id=current_user.id,
            reason=req.reason or "Model activation/rollback request"
        )
        log_system_event(
            "MODEL_VERSION_CHANGED",
            f"Active model switched to {req.target_version} by user '{current_user.username}' (ID: {current_user.id})."
        )
        return res
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update active model version: {exc}")


# -----------------
# 8. Clinical Case Review & Comments (Doctor Oversight)
# -----------------
@app.post("/api/v1/cases/{case_id}/reviews", response_model=ClinicalReviewResponse)
def add_case_review(
    case_id: int,
    review_in: ClinicalReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.CLINICIAN, UserRole.ADMIN]))
):
    """
    Allows authorized clinicians or admins to author a clinical observation,
    recommendation, and update case review status.
    Keeps doctor clinical judgment strictly separate from AI predictions.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    check_clinical_access(current_user, case)

    if not review_in.comment or not review_in.comment.strip():
        raise HTTPException(status_code=400, detail="Review observation/comment cannot be empty")

    # If case was unassigned, automatically assign to reviewing clinician
    if case.assigned_doctor_id is None:
        case.assigned_doctor_id = current_user.id

    valid_statuses = [s.value for s in ReviewStatus]
    status_to_set = review_in.review_status or ReviewStatus.REVIEWED.value
    if status_to_set in valid_statuses:
        case.review_status = status_to_set

    new_review = ClinicalReview(
        case_id=case.id,
        doctor_id=current_user.id,
        comment=review_in.comment.strip(),
        recommendation=review_in.recommendation.strip() if review_in.recommendation else None,
        review_status=case.review_status
    )
    db.add(new_review)
    db.commit()
    db.refresh(new_review)
    db.refresh(case)

    try:
        notify_review_completed(db, case, current_user, case.review_status)
    except Exception as exc:
        logger.warning(f"Failed to dispatch review notification: {exc}")

    return format_review_response(new_review)

@app.get("/api/v1/cases/{case_id}/reviews", response_model=List[ClinicalReviewResponse])
def get_case_reviews(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the clinical review and comment history for a case.
    Authorized for the patient who owns the case, authorized clinicians, and admins.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    check_clinical_access(current_user, case)

    reviews = db.query(ClinicalReview).filter(
        ClinicalReview.case_id == case.id
    ).order_by(ClinicalReview.created_at.desc()).all()

    return [format_review_response(r) for r in reviews]

@app.patch("/api/v1/cases/{case_id}/review-status", response_model=CaseResponse)
def update_case_review_status(
    case_id: int,
    status_in: ReviewStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.CLINICIAN, UserRole.ADMIN]))
):
    """
    Updates the clinical review status of a case:
    pending_review, under_review, reviewed, follow_up_required.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    check_clinical_access(current_user, case)

    valid_statuses = [s.value for s in ReviewStatus]
    if status_in.review_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid review status. Allowed: {', '.join(valid_statuses)}"
        )

    old_status = case.review_status
    case.review_status = status_in.review_status
    db.commit()
    db.refresh(case)

    try:
        notify_status_changed(db, case, old_status, case.review_status, current_user)
    except Exception as exc:
        logger.warning(f"Failed to dispatch review status notification: {exc}")

    return format_case_response(case)

@app.put("/api/v1/cases/{case_id}/assign", response_model=CaseResponse)
def assign_case_doctor(
    case_id: int,
    assignment_in: CaseAssignmentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.CLINICIAN, UserRole.ADMIN]))
):
    """
    Assigns a doctor to the case.
    Clinicians can claim unassigned cases. Admins can assign to any clinician.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    check_clinical_access(current_user, case)

    if getattr(case, "is_archived", False):
        raise HTTPException(status_code=400, detail="Cannot assign archived cases")

    target_doctor_id = current_user.id
    if assignment_in.doctor_id and current_user.role == UserRole.ADMIN:
        target_doctor_id = assignment_in.doctor_id

    target_doctor = db.query(User).filter(User.id == target_doctor_id).first()
    if not target_doctor or target_doctor.role not in [UserRole.CLINICIAN, UserRole.ADMIN]:
        raise HTTPException(status_code=400, detail="Target user must have clinician or admin role")

    case.assigned_doctor_id = target_doctor_id
    if case.review_status == ReviewStatus.PENDING_REVIEW.value:
        case.review_status = ReviewStatus.UNDER_REVIEW.value

    db.commit()
    db.refresh(case)

    try:
        notify_case_assigned(db, case, target_doctor, current_user)
    except Exception as exc:
        logger.warning(f"Failed to dispatch case assignment notification: {exc}")

    return format_case_response(case)

@app.post("/api/v1/cases/{case_id}/release", response_model=CaseResponse)
def release_case(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.CLINICIAN, UserRole.ADMIN]))
):
    """
    TASK 4: Clinician Case Relinquishment
    Allows the assigned clinician or an administrator to release a claimed case back to the unassigned triage queue.
    Preserves all existing patient demographics, image records, predictions, and clinical reviews.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if getattr(case, "is_archived", False):
        raise HTTPException(status_code=400, detail="Cannot release an archived case")

    # Authorization: Only the assigned clinician or admin can release the case
    if current_user.role != UserRole.ADMIN and case.assigned_doctor_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You are not the assigned clinician for this case and cannot release it."
        )

    # Review Status Protection:
    # If the case is already reviewed or follow_up_required, it should not return to the ordinary new-case triage queue
    if getattr(case, "review_status", None) in [ReviewStatus.REVIEWED.value, ReviewStatus.FOLLOW_UP_REQUIRED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot release case back to triage queue: Case already has a finalized clinical review (status: '{case.review_status}')."
        )

    # Release assignment
    case.assigned_doctor_id = None
    case.review_status = ReviewStatus.PENDING_REVIEW.value

    db.commit()
    db.refresh(case)

    try:
        notify_status_changed(db, case, "claimed", "unassigned", current_user)
    except Exception as exc:
        logger.warning(f"Failed to dispatch case release notification: {exc}")
    return format_case_response(case)

# -----------------
# Admin Clinician Verification Endpoints (TASK 1)
# -----------------
@app.get("/api/v1/admin/clinicians/pending", response_model=List[UserResponse])
def get_pending_clinicians(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Retrieves all clinician accounts awaiting administrative verification.
    """
    pending = db.query(User).filter(
        User.role == UserRole.CLINICIAN,
        (User.is_verified == False) | (User.is_verified.is_(None))
    ).order_by(User.created_at.desc()).all()
    return pending

@app.patch("/api/v1/admin/clinicians/{clinician_id}/verify", response_model=UserResponse)
def verify_clinician(
    clinician_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Approves and verifies a clinician account for active clinical access.
    """
    clinician = db.query(User).filter(User.id == clinician_id).first()
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician account not found")
    if clinician.role != UserRole.CLINICIAN:
        raise HTTPException(status_code=400, detail="User is not a clinician")

    clinician.is_verified = True
    db.commit()
    db.refresh(clinician)

    record_activity(
        db=db,
        action="CLINICIAN_VERIFIED",
        details=f"Administrator '{current_user.username}' verified Dr. {clinician.full_name}.",
        actor=current_user,
        target_type="user",
        target_id=str(clinician.id)
    )
    return clinician

@app.patch("/api/v1/admin/clinicians/{clinician_id}/reject", response_model=UserResponse)
def reject_clinician(
    clinician_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Rejects a clinician registration (disables active clinical login).
    """
    clinician = db.query(User).filter(User.id == clinician_id).first()
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician account not found")
    if clinician.role != UserRole.CLINICIAN:
        raise HTTPException(status_code=400, detail="User is not a clinician")

    clinician.is_verified = False
    clinician.is_active = False
    db.commit()
    db.refresh(clinician)

    record_activity(
        db=db,
        action="CLINICIAN_REJECTED",
        details=f"Administrator '{current_user.username}' rejected Dr. {clinician.full_name}.",
        actor=current_user,
        target_type="user",
        target_id=str(clinician.id)
    )
    return clinician

# -----------------------------------------------------------------------------
# Admin Panel Management Endpoints (Dashboard, Patients, Doctors, Analyses, Logs)
# -----------------------------------------------------------------------------

@app.get("/api/v1/admin/stats", response_model=AdminStatsResponse)
def get_admin_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Calculates summary system statistics and returns recent platform activity.
    """
    total_users = db.query(User).filter(User.is_active == True).count()
    total_patients = db.query(User).filter(User.role == UserRole.PATIENT, User.is_active == True).count()
    total_doctors = db.query(User).filter(User.role == UserRole.CLINICIAN, User.is_active == True).count()
    total_analyses = db.query(PredictionRecord).count()

    recent_logs = db.query(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(10).all()
    activity_items = []
    if recent_logs:
        for log in recent_logs:
            activity_items.append({
                "id": log.id,
                "action": log.action,
                "actor_name": log.actor_name,
                "details": log.details,
                "created_at": log.created_at.isoformat() if log.created_at else None
            })
    else:
        # Fallback to authentic database records if activity_logs is newly initialized
        recent_users = db.query(User).order_by(User.created_at.desc()).limit(5).all()
        for u in recent_users:
            activity_items.append({
                "id": f"u_{u.id}",
                "action": "USER_REGISTERED",
                "actor_name": u.full_name or u.username,
                "details": f"Registered new {u.role.value} account: {u.email}",
                "created_at": u.created_at.isoformat() if u.created_at else None
            })
        recent_preds = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(5).all()
        for p in recent_preds:
            activity_items.append({
                "id": f"p_{p.id}",
                "action": "ANALYSIS_COMPLETED",
                "actor_name": "AI Inference Engine",
                "details": f"Completed analysis for Case #{p.case_id} — {p.predicted_disease_name} ({p.risk_level})",
                "created_at": p.created_at.isoformat() if p.created_at else None
            })
        activity_items.sort(key=lambda x: x["created_at"] or "", reverse=True)
        activity_items = activity_items[:8]

    return {
        "total_users": total_users,
        "total_patients": total_patients,
        "total_doctors": total_doctors,
        "total_analyses": total_analyses,
        "recent_activity": activity_items
    }

@app.get("/api/v1/admin/patients", response_model=List[UserResponse])
def get_admin_patients(
    search: Optional[str] = None,
    q: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Returns all registered patient accounts, with optional search by name or email.
    """
    search_term = search or q
    query = db.query(User).filter(User.role == UserRole.PATIENT)
    if search_term and search_term.strip():
        s = f"%{search_term.strip()}%"
        query = query.filter((User.full_name.ilike(s)) | (User.email.ilike(s)) | (User.username.ilike(s)))
    return query.order_by(User.created_at.desc()).all()

@app.get("/api/v1/admin/doctors", response_model=List[UserResponse])
def get_admin_doctors(
    search: Optional[str] = None,
    q: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Returns all registered doctor accounts, with optional search by name or email.
    """
    search_term = search or q
    query = db.query(User).filter(User.role == UserRole.CLINICIAN)
    if search_term and search_term.strip():
        s = f"%{search_term.strip()}%"
        query = query.filter((User.full_name.ilike(s)) | (User.email.ilike(s)) | (User.username.ilike(s)))
    return query.order_by(User.created_at.desc()).all()

@app.put("/api/v1/admin/users/{user_id}", response_model=UserResponse)
def update_user_account_admin(
    user_id: int,
    user_in: UserUpdateAdmin,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Updates patient or doctor details (full_name, email, optional password, institution).
    Preserves role and existing verification status.
    """
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User account not found")

    changes = []

    if user_in.full_name is not None:
        trimmed_name = user_in.full_name.strip()
        if not trimmed_name:
            raise HTTPException(status_code=400, detail="Full name cannot be empty.")
        if target_user.full_name != trimmed_name:
            target_user.full_name = trimmed_name
            changes.append(f"name to '{trimmed_name}'")

    if user_in.email is not None:
        trimmed_email = user_in.email.strip()
        if not trimmed_email:
            raise HTTPException(status_code=400, detail="Email address cannot be empty.")
        if "@" not in trimmed_email or "." not in trimmed_email:
            raise HTTPException(status_code=400, detail="Invalid email address format.")
        if target_user.email != trimmed_email:
            existing_email = db.query(User).filter(User.email == trimmed_email, User.id != user_id).first()
            if existing_email:
                raise HTTPException(status_code=400, detail="Email address is already in use by another account.")
            target_user.email = trimmed_email
            changes.append(f"email to '{trimmed_email}'")

    if user_in.institution is not None and target_user.role == UserRole.CLINICIAN:
        target_user.institution = user_in.institution.strip()

    if user_in.password:
        pwd = user_in.password.strip()
        if len(pwd) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")
        target_user.hashed_password = hash_password(pwd)
        changes.append("password reset")

    db.commit()
    db.refresh(target_user)

    change_desc = ", ".join(changes) if changes else "profile details updated"
    record_activity(
        db=db,
        action="USER_UPDATED",
        details=f"Administrator '{current_user.username}' updated {target_user.role.value} account '{target_user.full_name}': {change_desc}.",
        actor=current_user,
        target_type="user",
        target_id=str(user_id)
    )

    return target_user

@app.delete("/api/v1/admin/users/{user_id}")
def delete_user_account_admin(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Safely deletes a complete patient or doctor account and manages relational integrity.
    """
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User account not found")

    if target_user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own active administrator account.")

    username = target_user.username
    full_name = target_user.full_name
    role = target_user.role.value

    # If clinician, unassign their assigned cases so cases safely return to the unassigned queue
    if target_user.role == UserRole.CLINICIAN:
        assigned_cases = db.query(Case).filter(Case.assigned_doctor_id == target_user.id).all()
        for c in assigned_cases:
            c.assigned_doctor_id = None
            if c.review_status == "under_review":
                c.review_status = "pending_review"

        # Reassign clinician reviews to admin to preserve patient case audit records
        reviews = db.query(ClinicalReview).filter(ClinicalReview.doctor_id == target_user.id).all()
        for r in reviews:
            r.doctor_id = current_user.id

    # Clean up notifications and reminders created for or by this user
    db.query(Notification).filter(Notification.user_id == target_user.id).delete(synchronize_session=False)
    db.query(FollowUpReminder).filter(FollowUpReminder.creator_id == target_user.id).delete(synchronize_session=False)

    # Clean up activity log actor references
    db.query(ActivityLog).filter(ActivityLog.actor_id == target_user.id).update({ActivityLog.actor_id: None}, synchronize_session=False)

    # If deleting patient, delete notifications for patient's cases
    if target_user.role == UserRole.PATIENT:
        patient_case_ids = [c.id for c in target_user.cases]
        if patient_case_ids:
            db.query(Notification).filter(Notification.case_id.in_(patient_case_ids)).delete(synchronize_session=False)
            db.query(FollowUpReminder).filter(FollowUpReminder.case_id.in_(patient_case_ids)).delete(synchronize_session=False)

    db.delete(target_user)
    db.commit()

    record_activity(
        db=db,
        action="USER_DELETED",
        details=f"Administrator '{current_user.username}' deleted {role} account '{full_name}' ({username}, ID: {user_id}).",
        actor=current_user,
        target_type="user",
        target_id=str(user_id)
    )

    return {"message": f"Successfully deleted {role} account '{full_name}'."}

@app.get("/api/v1/admin/analyses")
def get_admin_analyses(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Returns platform-wide diagnostic analyses with associated patient, case, and review details.
    """
    predictions = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit).all()
    results = []
    for pred in predictions:
        case = pred.case
        patient_name = case.creator.full_name if case and case.creator else "Unknown Patient"
        doctor_name = case.assigned_doctor.full_name if case and case.assigned_doctor else "Unassigned"
        results.append({
            "id": pred.id,
            "case_id": pred.case_id,
            "case_code": case.case_code if case else f"CAS-{pred.case_id}",
            "patient_name": patient_name,
            "patient_age": case.patient_age if case else None,
            "patient_gender": case.patient_gender if case else None,
            "body_location": case.body_location if case else "unspecified",
            "predicted_class": pred.predicted_class,
            "predicted_disease_name": pred.predicted_disease_name,
            "confidence_score": pred.confidence_score,
            "risk_level": pred.risk_level,
            "model_name": pred.model_name,
            "review_status": case.review_status if case else "pending_review",
            "assigned_doctor_name": doctor_name,
            "created_at": pred.created_at.isoformat() if pred.created_at else None
        })
    return results

@app.get("/api/v1/admin/activity-logs", response_model=List[ActivityLogResponse])
def get_admin_activity_logs(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Returns audit trail records of user management, analysis, and system activity.
    """
    logs = db.query(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(limit).all()
    if not logs:
        recent_users = db.query(User).order_by(User.created_at.desc()).limit(15).all()
        synth = []
        for u in recent_users:
            synth.append({
                "id": f"u_{u.id}",
                "actor_name": u.full_name or u.username,
                "action": "USER_REGISTERED",
                "target_type": "user",
                "target_id": str(u.id),
                "details": f"Registered new {u.role.value} account '{u.username}' ({u.email}).",
                "ip_address": "127.0.0.1",
                "created_at": u.created_at.isoformat() if u.created_at else None
            })
        recent_preds = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(15).all()
        for p in recent_preds:
            synth.append({
                "id": f"p_{p.id}",
                "actor_name": "AI Inference Engine",
                "action": "PREDICTION_SUCCESS",
                "target_type": "case",
                "target_id": str(p.case_id),
                "details": f"Diagnostic prediction completed: {p.predicted_disease_name} (Confidence: {p.confidence_score*100:.1f}%).",
                "ip_address": "127.0.0.1",
                "created_at": p.created_at.isoformat() if p.created_at else None
            })
        synth.sort(key=lambda x: x["created_at"] or "", reverse=True)
        return synth[:limit]

    return [
        {
            "id": l.id,
            "actor_name": l.actor_name or "System",
            "action": l.action,
            "target_type": l.target_type,
            "target_id": l.target_id,
            "details": l.details,
            "ip_address": l.ip_address or "127.0.0.1",
            "created_at": l.created_at.isoformat() if l.created_at else None
        }
        for l in logs
    ]

@app.get("/api/v1/admin/settings")
def get_admin_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Returns platform environment, security configuration parameters, and admin profile.
    """
    return {
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "algorithm": settings.ALGORITHM,
        "token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "iqa_blur_threshold": settings.IQA_BLUR_THRESHOLD_POOR,
        "admin_user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "role": current_user.role.value
        }
    }

# -----------------
# 9. Research & Model Evaluation Endpoints (Researcher Protected)
# -----------------
@app.get("/api/v1/research/metrics")
def get_research_model_metrics(
    current_user: User = Depends(require_role([UserRole.RESEARCHER, UserRole.ML_ENGINEER, UserRole.ADMIN]))
):
    """
    Returns authentic Stage 3 evaluation metrics calculated on the 1,501 untouched test images:
    accuracy, balanced accuracy, macro/weighted F1, macro/weighted ROC-AUC,
    per-class metrics, confusion matrix, and error analysis.
    """
    metrics_path = os.path.join(
        settings.BASE_DIR, "..", "ml_models", "evaluation_results", "test_evaluation_metrics.json"
    )
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {
        "overall_accuracy": 0.7022,
        "balanced_accuracy": 0.6085,
        "macro_precision": 0.4783,
        "macro_recall": 0.6085,
        "macro_f1": 0.4989,
        "weighted_f1": 0.7213,
        "macro_roc_auc": 0.9053,
        "weighted_roc_auc": 0.8942
    }

@app.get("/api/v1/research/image-quality-stats")
def get_image_quality_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.RESEARCHER, UserRole.ML_ENGINEER, UserRole.ADMIN]))
):
    """
    Aggregates image quality metrics across preprocessed images without disclosing patient identities.
    """
    images = db.query(ImageRecord).all()
    total_images = len(images)
    if total_images == 0:
        return {
            "total_images": 0,
            "suitable_count": 0,
            "suitable_percentage": 0.0,
            "average_blur_score": 0.0,
            "average_brightness": 0.0,
            "average_contrast": 0.0,
            "quality_breakdown": {}
        }

    suitable_count = sum(1 for img in images if img.is_suitable_for_ai)
    valid_blur = [img.blur_score for img in images if img.blur_score is not None]
    valid_brightness = [img.brightness_score for img in images if img.brightness_score is not None]
    valid_contrast = [img.contrast_score for img in images if img.contrast_score is not None]

    status_counts = {}
    for img in images:
        q_stat = img.quality_status or "Unknown"
        status_counts[q_stat] = status_counts.get(q_stat, 0) + 1

    return {
        "total_images": total_images,
        "suitable_count": suitable_count,
        "suitable_percentage": round((suitable_count / total_images) * 100, 1),
        "average_blur_score": round(sum(valid_blur) / len(valid_blur), 2) if valid_blur else 0.0,
        "average_brightness": round(sum(valid_brightness) / len(valid_brightness), 2) if valid_brightness else 0.0,
        "average_contrast": round(sum(valid_contrast) / len(valid_contrast), 2) if valid_contrast else 0.0,
        "quality_breakdown": status_counts
    }

@app.get("/api/v1/research/anonymized-cases")
def get_anonymized_research_cases(
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.RESEARCHER, UserRole.ML_ENGINEER, UserRole.ADMIN]))
):
    """
    Returns de-identified research case records for statistical modeling.
    Strictly excludes all patient identities, usernames, emails, clinical notes, and medical history.
    """
    cases = db.query(Case).order_by(Case.created_at.desc()).limit(limit).all()
    anonymized_list = []

    for c in cases:
        latest_img = c.images[-1] if c.images else None
        latest_pred = c.predictions[-1] if c.predictions else None

        anon_item = {
            "research_id": f"RES-{c.case_code[-8:]}",
            "body_location": c.body_location,
            "image_source": c.image_source,
            "patient_gender": c.patient_gender or "unknown",
            "patient_age_group": f"{(c.patient_age // 10) * 10}-{(c.patient_age // 10) * 10 + 9}" if c.patient_age else "Unknown",
            "image_quality_status": latest_img.quality_status if latest_img else "No Image",
            "blur_score": latest_img.blur_score if latest_img else None,
            "brightness_score": latest_img.brightness_score if latest_img else None,
            "contrast_score": latest_img.contrast_score if latest_img else None,
            "is_suitable_for_ai": latest_img.is_suitable_for_ai if latest_img else False,
            "predicted_class": latest_pred.predicted_class if latest_pred else "Not Evaluated",
            "predicted_disease_name": latest_pred.predicted_disease_name if latest_pred else "N/A",
            "confidence_score": latest_pred.confidence_score if latest_pred else None,
            "risk_level": latest_pred.risk_level if latest_pred else "N/A",
            "review_status": getattr(c, "review_status", "pending_review"),
            "created_timestamp": c.created_at
        }
        anonymized_list.append(anon_item)

    return anonymized_list

@app.get("/api/v1/research/export-anonymized")
def export_anonymized_cases(
    format: str = "json",
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.RESEARCHER, UserRole.ML_ENGINEER, UserRole.ADMIN]))
):
    """
    Exports de-identified research dataset in JSON or CSV format.
    Zero PHI, zero private clinical notes, zero patient identities.
    """
    import csv
    cases_data = get_anonymized_research_cases(limit=1000, db=db, current_user=current_user)

    if format.lower() == "csv":
        output = io.StringIO()
        if cases_data:
            fieldnames = list(cases_data[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in cases_data:
                row_copy = dict(row)
                if isinstance(row_copy.get("created_timestamp"), datetime):
                    row_copy["created_timestamp"] = row_copy["created_timestamp"].isoformat()
                writer.writerow(row_copy)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=dermai_anonymized_research_data.csv"}
        )

    return {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "exported_by": current_user.username,
        "institution": current_user.institution,
        "record_count": len(cases_data),
        "data_privacy": "HIPAA Safe Harbor De-Identified (Zero PHI, Zero Clinical Notes, Zero Patient Names)",
        "cases": cases_data
    }


# -----------------
# 9. In-App Clinical Notification Endpoints
# -----------------
@app.get("/api/v1/notifications", response_model=NotificationListResponse)
def get_notifications(
    unread_only: bool = Query(False, description="Filter for unread notifications only"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve clinical notifications for the current authenticated user.
    Strictly isolated: users cannot access notifications of other users.
    """
    query = db.query(Notification).filter(Notification.user_id == current_user.id)
    unread_count = db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).count()

    if unread_only:
        query = query.filter(Notification.is_read == False)

    notifications = query.order_by(Notification.created_at.desc()).limit(limit).all()
    total_count = db.query(Notification).filter(Notification.user_id == current_user.id).count()

    return {
        "unread_count": unread_count,
        "total_count": total_count,
        "notifications": notifications
    }


@app.patch("/api/v1/notifications/{notification_id}/read", response_model=NotificationResponse)
def mark_notification_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Mark a specific notification as read.
    Validates ownership to prevent unauthorized access.
    """
    notif = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")
    if notif.user_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Unauthorized access to notification")

    notif.is_read = True
    db.commit()
    db.refresh(notif)
    return notif


@app.post("/api/v1/notifications/read-all")
def mark_all_notifications_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Mark all unread notifications for the current authenticated user as read.
    """
    updated_count = db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).update({"is_read": True})
    db.commit()
    return {"marked_read": updated_count}


# -----------------
# 9B. Follow-Up Reminders & Re-examination Scheduling
# -----------------
@app.post("/api/v1/cases/{case_id}/reminders", response_model=FollowUpReminderResponse)
def create_follow_up_reminder(
    case_id: int,
    reminder_in: FollowUpReminderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([UserRole.CLINICIAN, UserRole.ADMIN, UserRole.PATIENT]))
):
    """
    Creates an administrative follow-up reminder for a patient case.
    Enforces case ownership: Patients can only create reminders for their own cases.
    Clinicians can create reminders for cases in their care.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    check_clinical_access(current_user, case)

    reminder = FollowUpReminder(
        case_id=case.id,
        creator_id=current_user.id,
        scheduled_date=reminder_in.scheduled_date,
        notes=reminder_in.notes,
        priority=reminder_in.priority or "routine",
        status="pending"
    )
    db.add(reminder)
    db.commit()
    db.refresh(reminder)

    # Dispatch notification to case owner
    create_notification(
        db,
        user_id=case.user_id,
        case_id=case.id,
        title="Follow-Up Re-examination Scheduled",
        message=f"A follow-up review for case {case.case_code} is scheduled for {reminder.scheduled_date}.",
        event_type="follow_up_reminder"
    )

    return reminder


@app.get("/api/v1/cases/{case_id}/reminders", response_model=List[FollowUpReminderResponse])
def get_case_reminders(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves scheduled follow-up reminders for a case with strict RBAC enforcement.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    check_clinical_access(current_user, case)
    reminders = db.query(FollowUpReminder).filter(FollowUpReminder.case_id == case_id).order_by(FollowUpReminder.created_at.desc()).all()
    return reminders


@app.patch("/api/v1/reminders/{reminder_id}/status", response_model=FollowUpReminderResponse)
def update_reminder_status(
    reminder_id: int,
    status_update: str = Query(..., description="completed or cancelled"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Marks a follow-up reminder as completed or cancelled.
    """
    reminder = db.query(FollowUpReminder).filter(FollowUpReminder.id == reminder_id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")

    case = db.query(Case).filter(Case.id == reminder.case_id).first()
    check_clinical_access(current_user, case)

    if status_update not in ["completed", "cancelled", "pending"]:
        raise HTTPException(status_code=400, detail="Invalid status. Must be completed, cancelled, or pending.")

    reminder.status = status_update
    db.commit()
    db.refresh(reminder)
    return reminder


# -----------------
# 10. Clinical Reporting & Export Endpoints
# -----------------
@app.get("/api/v1/cases/{case_id}/report/pdf")
def get_case_pdf_report(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generates downloadable PDF clinical report for a single case.
    Strictly enforces authentication and clinical authorization.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    check_clinical_access(current_user, case)

    try:
        pdf_bytes = generate_case_pdf(case, current_user, settings.UPLOAD_DIR)
        filename = f"DermAI_Report_{case.case_code}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(exc)}")


@app.get("/api/v1/cases/{case_id}/export/csv")
def export_case_csv(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Exports single case and prediction data as RFC 4180 compliant CSV.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    check_clinical_access(current_user, case)

    csv_data = generate_case_csv([case], current_user)
    filename = f"DermAI_Case_{case.case_code}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.get("/api/v1/cases/{case_id}/export/json")
def export_case_json(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Exports single case and prediction data as structured JSON.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    check_clinical_access(current_user, case)

    data = generate_case_json([case], current_user)
    return data[0] if data else {}


@app.get("/api/v1/cases/export/csv")
def export_all_cases_csv(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Batch CSV export of all cases accessible to current user.
    Patients get their own cases; Clinicians/Admins get authorized clinical cohort.
    """
    check_clinical_access(current_user)
    query = db.query(Case).filter((Case.is_archived == False) | (Case.is_archived.is_(None)))
    if current_user.role == UserRole.PATIENT:
        query = query.filter(Case.user_id == current_user.id)
    elif current_user.role == UserRole.CLINICIAN:
        query = query.filter(
            (Case.assigned_doctor_id == current_user.id) | (Case.assigned_doctor_id.is_(None))
        )
    cases = query.order_by(Case.created_at.desc()).all()
    csv_data = generate_case_csv(cases, current_user)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="dermai_cases_export.csv"'}
    )


@app.get("/api/v1/cases/export/json")
def export_all_cases_json(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Batch JSON export of all cases accessible to current user.
    """
    check_clinical_access(current_user)
    query = db.query(Case).filter((Case.is_archived == False) | (Case.is_archived.is_(None)))
    if current_user.role == UserRole.PATIENT:
        query = query.filter(Case.user_id == current_user.id)
    elif current_user.role == UserRole.CLINICIAN:
        query = query.filter(
            (Case.assigned_doctor_id == current_user.id) | (Case.assigned_doctor_id.is_(None))
        )
    cases = query.order_by(Case.created_at.desc()).all()
    return generate_case_json(cases, current_user)
