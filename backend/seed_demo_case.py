"""
Creates realistic test lesion image and seeds demo cases for immediate Milestone 1 demonstration.
"""
import os
import cv2
import numpy as np
from app.database import SessionLocal, engine, Base
import app.main  # Ensures tables and demo users are initialized
from app.models import User, Case, ImageRecord, CaseStatus
from app.preprocessing import preprocess_skin_image
from app.config import settings

def seed_demo_data():
    db = SessionLocal()
    try:
        doctor = db.query(User).filter(User.username == "dr_smith").first()
        if not doctor:
            print("Doctor account not found.")
            return

        # Check if sample case already exists
        existing = db.query(Case).filter(Case.case_code == "CAS-202610-DEMO01").first()
        if existing:
            print("Demo cases already seeded.")
            return

        # 1. Create realistic synthetic skin lesion image
        img_h, img_w = 400, 400
        # Background healthy skin tone (warm peach tone)
        skin = np.full((img_h, img_w, 3), (160, 190, 230), dtype=np.uint8)
        
        # Add subtle skin texture noise
        noise = np.random.normal(0, 5, (img_h, img_w, 3)).astype(np.int16)
        skin = np.clip(skin.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw an asymmetrical pigmented nevus/lesion in the center
        center_x, center_y = 200, 200
        # Multi-layer pigmented ellipse
        cv2.ellipse(skin, (center_x, center_y), (85, 60), 25, 0, 360, (55, 75, 120), -1)
        cv2.ellipse(skin, (center_x - 10, center_y + 5), (65, 45), 35, 0, 360, (30, 45, 80), -1)
        cv2.ellipse(skin, (center_x + 15, center_y - 10), (35, 25), -15, 0, 360, (20, 25, 45), -1)
        
        # Soften edges (natural lesion borders)
        skin = cv2.GaussianBlur(skin, (11, 11), 3)

        sample_filename = "CAS-202610-DEMO01_sample_lesion.jpg"
        original_path = os.path.join(settings.ORIGINAL_DIR, sample_filename)
        cv2.imwrite(original_path, skin)

        # 2. Run Preprocessing Pipeline
        prep_results = preprocess_skin_image(
            original_filepath=original_path,
            output_filename=sample_filename,
            target_size=(224, 224)
        )
        q_metrics = prep_results["quality_metrics"]

        # 3. Create Case 1
        case1 = Case(
            case_code="CAS-202610-DEMO01",
            user_id=doctor.id,
            patient_age=54,
            patient_gender="male",
            known_allergies="None reported",
            medical_history="Patient reports sun exposure and dysplastic nevus history in family.",
            clinical_notes="Asymmetrical pigmented lesion on upper back. Irregular pigment network observed.",
            body_location="back",
            image_source="mobile_upload",
            status=CaseStatus.PREPROCESSED
        )
        db.add(case1)
        db.commit()
        db.refresh(case1)

        # Image record for Case 1
        img_rec = ImageRecord(
            case_id=case1.id,
            filename=sample_filename,
            original_path=original_path,
            preview_path=prep_results["preview_filepath"],
            preprocessed_path=prep_results["preprocessed_filepath"],
            file_size_bytes=os.path.getsize(original_path),
            width=img_w,
            height=img_h,
            format="jpg",
            quality_status=q_metrics["quality_status"],
            blur_score=q_metrics["blur_score"],
            brightness_score=q_metrics["brightness_score"],
            contrast_score=q_metrics["contrast_score"],
            is_suitable_for_ai=q_metrics["is_suitable_for_ai"],
            quality_notes="; ".join(q_metrics["notes"]) if q_metrics["notes"] else "All visual metrics within normal range"
        )
        db.add(img_rec)

        # 4. Create Case 2 (awaiting upload)
        case2 = Case(
            case_code="CAS-202610-DEMO02",
            user_id=doctor.id,
            patient_age=67,
            patient_gender="female",
            known_allergies="Latex",
            medical_history="Previous basal cell carcinoma excised 4 years ago.",
            clinical_notes="Erythematous scaly macule on left cheek. Patient requested mobile check.",
            body_location="face",
            image_source="mobile_upload",
            status=CaseStatus.CREATED
        )
        db.add(case2)
        db.commit()

        print("[Seed] Successfully created demo cases with verified preprocessed tensors.")
    finally:
        db.close()

if __name__ == "__main__":
    seed_demo_data()
