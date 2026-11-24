from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict
from .models import UserRole, CaseStatus, ReviewStatus

# Auth Schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str
    full_name: str
    role: UserRole
    is_verified: bool = True

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str
    role: UserRole = UserRole.CLINICIAN
    institution: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: UserRole
    institution: Optional[str] = None
    is_active: bool
    is_verified: bool = True
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class UserUpdateAdmin(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    institution: Optional[str] = None

class AdminStatsResponse(BaseModel):
    total_users: int
    total_patients: int
    total_doctors: int
    total_analyses: int
    recent_activity: List[Dict[str, Any]]

class ActivityLogResponse(BaseModel):
    id: Any
    actor_name: Optional[str] = None
    action: str
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    details: str
    ip_address: Optional[str] = None
    created_at: Optional[str] = None

# Image Schemas
class ImageQualityMetrics(BaseModel):
    blur_score: float
    brightness_score: float
    contrast_score: float
    width: int
    height: int
    file_size_bytes: int
    quality_status: str  # "Good Quality", "Acceptable Quality", "Poor Quality"
    is_suitable_for_ai: bool
    notes: List[str]

class ImageRecordResponse(BaseModel):
    id: int
    case_id: int
    filename: str
    original_url: str
    preview_url: Optional[str] = None
    preprocessed_url: Optional[str] = None
    file_size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    quality_status: str
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    contrast_score: Optional[float] = None
    is_suitable_for_ai: bool
    quality_notes: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Case Schemas
class CaseCreate(BaseModel):
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = "unknown"
    known_allergies: Optional[str] = None
    medical_history: Optional[str] = None
    clinical_notes: Optional[str] = None
    body_location: str = "back"
    image_source: Optional[str] = "mobile_upload"

# Prediction Schemas
class PredictionResponse(BaseModel):
    id: int
    case_id: int
    image_id: int
    predicted_class: str
    predicted_disease_name: str
    confidence_score: float
    risk_level: str
    class_probabilities: Dict[str, float]
    model_name: str
    prediction_timestamp: datetime
    disclaimer: str
    image_quality_status: Optional[str] = None
    is_suitable_for_ai: Optional[bool] = None
    top_three_predictions: Optional[List[Dict[str, Any]]] = None
    top_five_predictions: Optional[List[Dict[str, Any]]] = None
    ranked_predictions: Optional[List[Dict[str, Any]]] = None
    gradcam_overlay_url: Optional[str] = None
    gradcam_heatmap_url: Optional[str] = None
    gradcam_pp_overlay_url: Optional[str] = None
    gradcam_pp_heatmap_url: Optional[str] = None
    saliency_overlay_url: Optional[str] = None
    saliency_heatmap_url: Optional[str] = None
    vit_attention_overlay_url: Optional[str] = None
    vit_attention_heatmap_url: Optional[str] = None
    vit_rollout_overlay_url: Optional[str] = None
    vit_rollout_heatmap_url: Optional[str] = None
    gradcam_mobilenet_overlay_url: Optional[str] = None
    gradcam_efficientnet_overlay_url: Optional[str] = None
    lesion_analysis_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

# Clinical Review Schemas
class ClinicalReviewCreate(BaseModel):
    comment: str
    recommendation: Optional[str] = None
    review_status: Optional[str] = "reviewed"

class ClinicalReviewResponse(BaseModel):
    id: int
    case_id: int
    doctor_id: int
    doctor_name: str
    doctor_institution: Optional[str] = None
    comment: str
    recommendation: Optional[str] = None
    review_status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class ReviewStatusUpdate(BaseModel):
    review_status: str

class CaseAssignmentUpdate(BaseModel):
    doctor_id: Optional[int] = None

class CaseResponse(BaseModel):
    id: int
    case_code: str
    user_id: int
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    known_allergies: Optional[str] = None
    medical_history: Optional[str] = None
    clinical_notes: Optional[str] = None
    body_location: str
    image_source: str
    status: CaseStatus
    review_status: str = "pending_review"
    assigned_doctor_id: Optional[int] = None
    assigned_doctor_name: Optional[str] = None
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime
    images: List[ImageRecordResponse] = []
    predictions: List[PredictionResponse] = []
    reviews: List[ClinicalReviewResponse] = []

    model_config = ConfigDict(from_attributes=True)

# Preprocessing Pipeline Schemas
class PreprocessOptions(BaseModel):
    target_width: int = 224
    target_height: int = 224
    apply_color_norm: bool = True
    apply_noise_reduction: bool = True
    apply_contrast_enhancement: bool = True

class PreprocessResult(BaseModel):
    image_id: int
    case_id: int
    original_dimensions: List[int]
    preprocessed_dimensions: List[int]
    quality_metrics: ImageQualityMetrics
    normalized_mean: List[float]
    normalized_std: List[float]
    preprocessed_image_url: str
    preview_image_url: str
    status: str

# Notification Schemas
class NotificationResponse(BaseModel):
    id: int
    user_id: int
    case_id: Optional[int] = None
    title: str
    message: str
    event_type: str
    is_read: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class NotificationListResponse(BaseModel):
    unread_count: int
    total_count: int
    notifications: List[NotificationResponse] = []


# Profile & User Management Schemas
class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    institution: Optional[str] = None

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


# Independent Image Quality Assessment Schemas
class ImageQualityAssessmentResponse(BaseModel):
    quality_status: str  # GOOD, ACCEPTABLE, POOR
    is_suitable_for_ai: bool
    blur_score: float
    brightness_score: float
    contrast_score: float
    shadow_clip_pct: float
    highlight_clip_pct: float
    width: int
    height: int
    file_size_bytes: int
    is_blurred: bool
    is_underexposed: bool
    is_overexposed: bool
    is_low_contrast: bool
    is_sub_resolution: bool
    aspect_ratio: Optional[float] = None
    is_extreme_aspect: Optional[bool] = None
    recommendation: str
    notes: List[str] = []


# Longitudinal Prediction Comparison Schemas
class PredictionComparisonItem(BaseModel):
    prediction_id: int
    created_at: datetime
    model_name: str
    predicted_class: str
    predicted_disease_name: str
    confidence_score: float
    risk_level: str
    class_probabilities: Dict[str, float]
    image_url: str
    gradcam_url: Optional[str] = None

class CaseComparisonResponse(BaseModel):
    case_id: int
    case_code: str
    earlier_analysis: PredictionComparisonItem
    newer_analysis: PredictionComparisonItem
    disclaimer: str


# Model Registry & Rollback Schemas
class ModelRegistryItem(BaseModel):
    version: str
    model_name: str
    architecture: str
    checkpoint_file: str
    dataset_reference: str
    input_dimensions: List[int]
    preprocessing: str
    num_classes: int
    class_labels: List[str]
    created_at: str
    is_active: bool
    evaluation_summary: Dict[str, Any]

class ModelRollbackRequest(BaseModel):
    target_version: str
    reason: Optional[str] = "Administrative model activation/rollback"

class ModelRollbackResponse(BaseModel):
    status: str
    active_version: str
    previous_version: str
    model_name: str
    architecture: str
    timestamp: str


# Follow-up Reminder Schemas
class FollowUpReminderCreate(BaseModel):
    scheduled_date: str  # YYYY-MM-DD
    notes: str
    priority: Optional[str] = "routine"  # routine, urgent

class FollowUpReminderResponse(BaseModel):
    id: int
    case_id: int
    creator_id: int
    scheduled_date: str
    notes: str
    priority: str
    status: str  # pending, completed, cancelled
    created_at: datetime
    disclaimer: str


