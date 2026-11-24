"""
DermAI Notification System Service
===================================
Manages creation, dispatch, retrieval, and status tracking for in-app clinical notifications.
Covers:
- Case intake events
- Deep learning inference completion
- Clinical review / observation completion
- Case triage and assignment updates
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from .models import Notification, Case, User, UserRole
from .logger import log_system_event

def create_notification(
    db: Session,
    user_id: int,
    case_id: Optional[int],
    title: str,
    message: str,
    event_type: str
) -> Notification:
    """Create and persist a new user notification."""
    notif = Notification(
        user_id=user_id,
        case_id=case_id,
        title=title,
        message=message,
        event_type=event_type,
        is_read=False
    )
    db.add(notif)
    db.commit()
    db.refresh(notif)
    log_system_event(
        "NOTIFICATION_CREATED",
        f"Event '{event_type}' notified to user ID {user_id} for case ID {case_id}"
    )
    return notif

def notify_case_created(db: Session, case: Case, creator: User):
    """Notify case creator and active verified clinicians upon intake."""
    # Notify creator
    create_notification(
        db,
        user_id=creator.id,
        case_id=case.id,
        title="Case Registered",
        message=f"Clinical screening case {case.case_code} was registered successfully.",
        event_type="case_created"
    )

    # If unassigned, notify active verified clinicians for triage awareness
    clinicians = db.query(User).filter(
        User.role == UserRole.CLINICIAN,
        User.is_active == True,
        User.is_verified == True,
        User.id != creator.id
    ).limit(10).all()

    for doc in clinicians:
        create_notification(
            db,
            user_id=doc.id,
            case_id=case.id,
            title="New Case in Triage Queue",
            message=f"New clinical case {case.case_code} (Site: {case.body_location}) awaits triage evaluation.",
            event_type="case_created"
        )

def notify_prediction_completed(
    db: Session,
    case: Case,
    model_name: str,
    predicted_disease: str,
    actor_id: int
):
    """Notify case owner and assigned doctor when AI inference completes."""
    recipients = {case.user_id}
    if case.assigned_doctor_id:
        recipients.add(case.assigned_doctor_id)

    for uid in recipients:
        create_notification(
            db,
            user_id=uid,
            case_id=case.id,
            title="AI Analysis Complete",
            message=f"Prediction generated for case {case.case_code} using {model_name}: {predicted_disease}.",
            event_type="prediction_completed"
        )

def notify_review_completed(
    db: Session,
    case: Case,
    doctor: User,
    review_status: str
):
    """Notify patient/case creator that clinician observation has been recorded."""
    create_notification(
        db,
        user_id=case.user_id,
        case_id=case.id,
        title="Doctor Review Completed",
        message=f"Dr. {doctor.full_name or doctor.username} completed clinical review for case {case.case_code}. Status: {review_status.replace('_', ' ').title()}.",
        event_type="review_completed"
    )

def notify_case_assigned(
    db: Session,
    case: Case,
    doctor: User,
    assigner: User
):
    """Notify assigned clinician and patient of physician assignment."""
    # Notify doctor
    if doctor.id != assigner.id:
        create_notification(
            db,
            user_id=doctor.id,
            case_id=case.id,
            title="Case Assigned to You",
            message=f"Case {case.case_code} has been assigned to you by {assigner.full_name or assigner.username}.",
            event_type="case_assigned"
        )
    # Notify patient
    create_notification(
        db,
        user_id=case.user_id,
        case_id=case.id,
        title="Physician Assigned",
        message=f"Dr. {doctor.full_name or doctor.username} has been assigned to review case {case.case_code}.",
        event_type="case_assigned"
    )

def notify_status_changed(
    db: Session,
    case: Case,
    old_status: str,
    new_status: str,
    actor: User
):
    """Notify patient of key status updates."""
    create_notification(
        db,
        user_id=case.user_id,
        case_id=case.id,
        title="Case Status Updated",
        message=f"Case {case.case_code} status changed from {old_status} to {new_status}.",
        event_type="status_changed"
    )
