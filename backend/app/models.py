import enum
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from .database import Base

def utc_now():
    """Returns timezone-aware UTC datetime for consistent database timestamps."""
    return datetime.now(timezone.utc)

class UserRole(str, enum.Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"

class CaseStatus(str, enum.Enum):
    CREATED = "created"
    IMAGE_UPLOADED = "image_uploaded"
    PREPROCESSED = "preprocessed"
    PENDING_ANALYSIS = "pending_analysis"
    COMPLETED = "completed"

class ReviewStatus(str, enum.Enum):
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    FOLLOW_UP_REQUIRED = "follow_up_required"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.CLINICIAN, nullable=False)
    institution = Column(String(150), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)

    cases = relationship("Case", foreign_keys="[Case.user_id]", back_populates="creator", cascade="all, delete-orphan")

class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    case_code = Column(String(50), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Patient Demographics (De-identified for privacy)
    patient_age = Column(Integer, nullable=True)
    patient_gender = Column(String(20), nullable=True)
    known_allergies = Column(String(255), nullable=True)
    medical_history = Column(Text, nullable=True)
    clinical_notes = Column(Text, nullable=True)
    
    # Lesion Details
    body_location = Column(String(100), nullable=False, default="unspecified")
    image_source = Column(String(50), default="mobile_upload")
    status = Column(Enum(CaseStatus), default=CaseStatus.CREATED, nullable=False)
    
    # Clinical Review & Assignment
    review_status = Column(String(50), default="pending_review", nullable=False)
    assigned_doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_archived = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    creator = relationship("User", foreign_keys=[user_id], back_populates="cases")
    assigned_doctor = relationship("User", foreign_keys=[assigned_doctor_id])
    images = relationship("ImageRecord", back_populates="case", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="case", cascade="all, delete-orphan")
    reviews = relationship("ClinicalReview", back_populates="case", cascade="all, delete-orphan", order_by="desc(ClinicalReview.created_at)")

class ImageRecord(Base):
    __tablename__ = "image_records"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500), nullable=False)
    preview_path = Column(String(500), nullable=True)
    preprocessed_path = Column(String(500), nullable=True)
    
    # File specifications
    file_size_bytes = Column(Integer, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    format = Column(String(20), nullable=True)
    
    # Quality assessment
    quality_status = Column(String(30), default="pending")  # good, acceptable, poor
    blur_score = Column(Float, nullable=True)
    brightness_score = Column(Float, nullable=True)
    contrast_score = Column(Float, nullable=True)
    is_suitable_for_ai = Column(Boolean, default=False)
    quality_notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=utc_now)

    case = relationship("Case", back_populates="images")
    predictions = relationship("PredictionRecord", back_populates="image", cascade="all, delete-orphan")

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    image_id = Column(Integer, ForeignKey("image_records.id"), nullable=False)

    predicted_class = Column(String(20), nullable=False)
    predicted_disease_name = Column(String(100), nullable=False)
    confidence_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    class_probabilities_json = Column(Text, nullable=False)
    model_name = Column(String(100), default="DermAI_MobileNetV2", nullable=False)
    disclaimer = Column(Text, nullable=False)

    created_at = Column(DateTime, default=utc_now)

    case = relationship("Case", back_populates="predictions")
    image = relationship("ImageRecord", back_populates="predictions")

class ClinicalReview(Base):
    __tablename__ = "clinical_reviews"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    comment = Column(Text, nullable=False)  # Clinical observation
    recommendation = Column(Text, nullable=True)  # Clinical recommendation or follow-up instruction
    review_status = Column(String(50), default="reviewed", nullable=False)

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    case = relationship("Case", back_populates="reviews")
    doctor = relationship("User", foreign_keys=[doctor_id])

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=True, index=True)

    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    event_type = Column(String(50), nullable=False)  # case_created, prediction_completed, review_completed, case_assigned, status_changed
    is_read = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", foreign_keys=[user_id])
    case = relationship("Case", foreign_keys=[case_id])

class FollowUpReminder(Base):
    __tablename__ = "follow_up_reminders"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    scheduled_date = Column(String(50), nullable=False)  # YYYY-MM-DD
    notes = Column(Text, nullable=False)
    priority = Column(String(30), default="routine", nullable=False)  # routine, urgent
    status = Column(String(30), default="pending", nullable=False)  # pending, completed, cancelled
    disclaimer = Column(
        Text,
        default="Administrative follow-up reminder only; not medical advice or clinical triage instruction.",
        nullable=False
    )

    created_at = Column(DateTime, default=utc_now, nullable=False)

    case = relationship("Case", foreign_keys=[case_id])
    creator = relationship("User", foreign_keys=[creator_id])


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    actor_name = Column(String(100), nullable=True)
    action = Column(String(100), nullable=False)  # USER_REGISTERED, USER_UPDATED, USER_DELETED, PREDICTION_SUCCESS, etc.
    target_type = Column(String(50), nullable=True)  # user, case, system
    target_id = Column(String(50), nullable=True)
    details = Column(Text, nullable=False)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    actor = relationship("User", foreign_keys=[actor_id])


