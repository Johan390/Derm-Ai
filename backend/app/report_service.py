"""
DermAI Clinical Reporting and Export Engine
============================================
Implements Milestone 4.1:
1. High-fidelity downloadable PDF Clinical Case Reports with ReportLab
   - Headers, Footers, Page Numbering ("Page X of Y")
   - Case metadata & authorized patient demographics
   - Embedded primary dermoscopy images
   - 7-Class deep learning predictions with explicit uncalibrated score labeling
   - Embedded explainability heatmaps (Grad-CAM & Heuristic ROI)
   - Clinical review notes & follow-up recommendations
   - Prominent medical safety disclaimer
2. Authorized CSV exports
3. Structured JSON exports
"""

import io
import os
import json
import csv
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas
from PIL import Image as PILImage

from .models import Case, PredictionRecord, ClinicalReview, User, UserRole
from .logger import log_report_generation, log_report_failure

# Mapping of HAM10000 classes to descriptive clinical names
DISEASE_NAME_MAP = {
    "akiec": "Actinic Keratoses & Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

DISCLAIMER_TEXT = (
    "<b>IMPORTANT MEDICAL DISCLAIMER & CLINICAL NOTICE:</b><br/>"
    "DermAI is an investigational deep-learning clinical decision support and screening platform. "
    "It is <b>NOT an autonomous medical diagnostic device</b>. Model confidence scores are raw, "
    "uncalibrated neural network softmax activations across output classes and must <b>NEVER be "
    "interpreted as calibrated clinical disease probabilities</b>. No clinical trial validation or "
    "autonomous diagnostic efficacy is claimed or implied. All algorithmic classifications and "
    "explainability visual heatmaps are adjunctive reference information only. Final diagnostic, biopsy, "
    "and management decisions remain the exclusive responsibility of a licensed medical clinician or dermatologist."
)

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that calculates total page count and prints running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0D9488"))  # Teal primary
        # Running Header
        self.drawString(36, 756, "DERMAI CLINICAL DECISION SUPPORT PLATFORM")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(576, 756, "CONFIDENTIAL CLINICAL REPORT")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 748, 576, 748)

        # Running Footer
        self.line(36, 42, 576, 42)
        self.setFont("Helvetica-Bold", 7)
        self.setFillColor(colors.HexColor("#DC2626"))
        self.drawString(36, 30, "CLINICAL DECISION SUPPORT ONLY - NOT AN AUTONOMOUS DIAGNOSIS")
        self.setFont("Helvetica", 7)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(576, 30, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def _get_safe_rl_image(file_path: Optional[str], max_width: float = 140, max_height: float = 140) -> Optional[RLImage]:
    """Safely validate and return a ReportLab Image flowable maintaining aspect ratio."""
    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        with PILImage.open(file_path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return None
            aspect = w / h
            if aspect >= 1:
                render_w = min(max_width, w)
                render_h = render_w / aspect
            else:
                render_h = min(max_height, h)
                render_w = render_h * aspect
            return RLImage(file_path, width=render_w, height=render_h)
    except Exception:
        return None


def generate_case_pdf(
    case: Case,
    current_user: User,
    upload_dir: str
) -> bytes:
    """
    Generate professional clinical PDF report for a case.
    Strictly follows role-based access control and medical safety constraints.
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=54,
            bottomMargin=54
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=12
        )
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0F766E"),
            spaceBefore=10,
            spaceAfter=6
        )
        cell_bold = ParagraphStyle(
            "CellBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#334155")
        )
        cell_normal = ParagraphStyle(
            "CellNormal",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1E293B")
        )
        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=9.5,
            textColor=colors.HexColor("#991B1B")
        )

        elements = []

        # Header Title Banner
        role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
        elements.append(Paragraph("DermAI Clinical Lesion Evaluation Report", title_style))
        elements.append(
            Paragraph(
                f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} • "
                f"Requesting User: {current_user.full_name or current_user.username} ({role_str.upper()})",
                subtitle_style
            )
        )
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0D9488"), spaceAfter=10))

        # 1. Case & Patient Information Table
        elements.append(Paragraph("1. CASE IDENTIFICATION & PATIENT METADATA", section_style))

        # Check if user is authorized to see clinical demographics
        # Patients can see their own; Clinicians and Admins can see all
        can_view_phi = (
            current_user.role in [UserRole.CLINICIAN, UserRole.ADMIN] or
            current_user.id == case.user_id
        )

        patient_age_val = f"{case.patient_age} years" if (can_view_phi and case.patient_age) else "De-identified / Redacted"
        patient_gender_val = case.patient_gender if (can_view_phi and case.patient_gender) else "De-identified"
        allergies_val = case.known_allergies if (can_view_phi and case.known_allergies) else "None disclosed"
        med_history_val = case.medical_history if (can_view_phi and case.medical_history) else "Not disclosed / Redacted"
        notes_val = case.clinical_notes if (can_view_phi and case.clinical_notes) else "De-identified"

        assigned_doc_name = "Unassigned (Triage Queue)"
        if case.assigned_doctor:
            assigned_doc_name = f"Dr. {case.assigned_doctor.full_name or case.assigned_doctor.username} ({case.assigned_doctor.institution or 'Staff'})"

        case_info_data = [
            [
                Paragraph("Case Identifier:", cell_bold),
                Paragraph(f"<b>{case.case_code}</b> (DB ID: #{case.id})", cell_normal),
                Paragraph("Clinical Status:", cell_bold),
                Paragraph(case.status.value.upper(), cell_normal)
            ],
            [
                Paragraph("Registration Date:", cell_bold),
                Paragraph(case.created_at.strftime("%Y-%m-%d %H:%M UTC") if case.created_at else "N/A", cell_normal),
                Paragraph("Review Status:", cell_bold),
                Paragraph(case.review_status.replace("_", " ").upper(), cell_normal)
            ],
            [
                Paragraph("Anatomical Location:", cell_bold),
                Paragraph(case.body_location.title(), cell_normal),
                Paragraph("Assigned Clinician:", cell_bold),
                Paragraph(assigned_doc_name, cell_normal)
            ],
            [
                Paragraph("Patient Age / Gender:", cell_bold),
                Paragraph(f"{patient_age_val} • {patient_gender_val}", cell_normal),
                Paragraph("Known Allergies:", cell_bold),
                Paragraph(allergies_val, cell_normal)
            ],
            [
                Paragraph("Medical History:", cell_bold),
                Paragraph(med_history_val, cell_normal),
                Paragraph("Intake Notes:", cell_bold),
                Paragraph(notes_val, cell_normal)
            ]
        ]

        t_case = Table(case_info_data, colWidths=[110, 160, 110, 160])
        t_case.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(t_case)
        elements.append(Spacer(1, 10))

        # 2. Deep Learning Inference Analysis
        elements.append(Paragraph("2. DEEP LEARNING MODEL PREDICTIONS", section_style))

        latest_prediction: Optional[PredictionRecord] = None
        if case.predictions and len(case.predictions) > 0:
            latest_prediction = case.predictions[-1]

        if latest_prediction:
            class_probs = {}
            if latest_prediction.class_probabilities_json:
                try:
                    class_probs = json.loads(latest_prediction.class_probabilities_json)
                except Exception:
                    class_probs = {}

            pred_category = latest_prediction.predicted_class
            full_disease_name = latest_prediction.predicted_disease_name or DISEASE_NAME_MAP.get(pred_category, pred_category)
            conf_score = latest_prediction.confidence_score

            pred_summary_data = [
                [
                    Paragraph("Prediction Model:", cell_bold),
                    Paragraph(f"<b>{latest_prediction.model_name}</b>", cell_normal),
                    Paragraph("Evaluation Date:", cell_bold),
                    Paragraph(latest_prediction.created_at.strftime("%Y-%m-%d %H:%M UTC") if latest_prediction.created_at else "N/A", cell_normal)
                ],
                [
                    Paragraph("Predicted Category:", cell_bold),
                    Paragraph(f"<b>{full_disease_name}</b> ({pred_category.upper()})", cell_normal),
                    Paragraph("Clinical Risk Tier:", cell_bold),
                    Paragraph(f"<b>{latest_prediction.risk_level.upper()}</b>", cell_normal)
                ],
                [
                    Paragraph("Model Confidence Score:", cell_bold),
                    Paragraph(
                        f"<b>{conf_score:.4f}</b> <font color='#DC2626' size='7'>*(Raw Uncalibrated Softmax Score; NOT a calibrated probability)*</font>",
                        cell_normal
                    ),
                    Paragraph("Total Model Inferences:", cell_bold),
                    Paragraph(f"{len(case.predictions)} inference run(s)", cell_normal)
                ]
            ]

            t_pred = Table(pred_summary_data, colWidths=[110, 160, 110, 160])
            t_pred.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0FDFA")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCFBF1")),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            elements.append(t_pred)
            elements.append(Spacer(1, 8))

            # Probability Breakdown Table
            if class_probs:
                prob_table_data = [
                    [
                        Paragraph("Class Code", cell_bold),
                        Paragraph("HAM10000 Skin Lesion Category", cell_bold),
                        Paragraph("Softmax Score", cell_bold),
                        Paragraph("Classification Status", cell_bold)
                    ]
                ]
                for code in sorted(class_probs.keys()):
                    score = float(class_probs[code])
                    d_name = DISEASE_NAME_MAP.get(code, code)
                    is_top = (code == pred_category)
                    status_lbl = "PRIMARY CANDIDATE" if is_top else "Differential"
                    bg_color = colors.HexColor("#FEF08A") if is_top else colors.white
                    prob_table_data.append([
                        Paragraph(f"<b>{code.upper()}</b>", cell_normal),
                        Paragraph(d_name, cell_normal),
                        Paragraph(f"{score:.4f} ({score*100:.1f}%)", cell_normal),
                        Paragraph(f"<b>{status_lbl}</b>", cell_normal)
                    ])

                t_prob = Table(prob_table_data, colWidths=[70, 240, 110, 120])
                t_prob.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]))
                elements.append(t_prob)
        else:
            elements.append(Paragraph("<i>No deep learning predictions have been executed for this case.</i>", cell_normal))

        elements.append(Spacer(1, 10))

        # 2B. Image Quality Assessment (IQA)
        if case.images and len(case.images) > 0:
            latest_img = case.images[-1]
            q_status = (latest_img.quality_status or "ACCEPTABLE").upper()
            q_color = "#16A34A" if "GOOD" in q_status else ("#D97706" if "ACCEPTABLE" in q_status else "#DC2626")
            iqa_table_data = [
                [
                    Paragraph("Quality Classification:", cell_bold),
                    Paragraph(f"<font color='{q_color}'><b>{q_status}</b></font> (AI Suitability: {'Suitable' if latest_img.is_suitable_for_ai else 'Suboptimal'})", cell_normal),
                    Paragraph("Blur Variance Score:", cell_bold),
                    Paragraph(f"{latest_img.blur_score:.1f}" if latest_img.blur_score is not None else "N/A", cell_normal)
                ],
                [
                    Paragraph("Mean Brightness:", cell_bold),
                    Paragraph(f"{latest_img.brightness_score:.1f} / 255" if latest_img.brightness_score is not None else "N/A", cell_normal),
                    Paragraph("Contrast (Std Dev):", cell_bold),
                    Paragraph(f"{latest_img.contrast_score:.1f}" if latest_img.contrast_score is not None else "N/A", cell_normal)
                ],
                [
                    Paragraph("Image Resolution:", cell_bold),
                    Paragraph(f"{latest_img.width}x{latest_img.height} ({latest_img.format or 'JPEG'})", cell_normal),
                    Paragraph("IQA Diagnostic Notes:", cell_bold),
                    Paragraph(latest_img.quality_notes or "Visual indicators within standard clinical range.", cell_normal)
                ]
            ]
            t_iqa = Table(iqa_table_data, colWidths=[110, 160, 110, 160])
            t_iqa.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(Paragraph("IMAGE QUALITY ASSESSMENT (IQA)", section_style))
            elements.append(t_iqa)
            elements.append(Spacer(1, 10))

        # 3. Clinical Images & Explainability Visualizations
        elements.append(Paragraph("3. CLINICAL IMAGES & EXPLAINABILITY VISUALIZATIONS", section_style))

        # Check for original image and explainability files
        img_cells = []
        img_labels = []

        # Primary Dermoscopy Image
        if case.images and len(case.images) > 0:
            primary_img_rec = case.images[-1]
            img_path = primary_img_rec.original_path
            if not os.path.isabs(img_path):
                img_path = os.path.join(upload_dir, img_path)
            rl_img = _get_safe_rl_image(img_path, max_width=150, max_height=150)
            if rl_img:
                img_cells.append(rl_img)
                img_labels.append(Paragraph("<b>Primary Lesion Image</b><br/>Preprocessed 224x224 RGB", cell_normal))

        # Grad-CAM Heatmap / Overlay (check if exists in uploads/explainability)
        if latest_prediction:
            gradcam_path = os.path.join(
                upload_dir,
                "explainability",
                f"case_{case.id}_prediction_{latest_prediction.id}_overlay.jpg"
            )
            rl_cam = _get_safe_rl_image(gradcam_path, max_width=150, max_height=150)
            if rl_cam:
                img_cells.append(rl_cam)
                img_labels.append(Paragraph("<b>Grad-CAM Explainability</b><br/>Salient Activation Overlay", cell_normal))

        # Heuristic ROI Overlay
        roi_path = os.path.join(upload_dir, "explainability", f"case_{case.id}_lesion_analysis_overlay.jpg")
        rl_roi = _get_safe_rl_image(roi_path, max_width=150, max_height=150)
        if rl_roi:
            img_cells.append(rl_roi)
            img_labels.append(Paragraph("<b>Heuristic Lesion ROI</b><br/>Classical Edge/Otsu Contour", cell_normal))

        if img_cells:
            # Build side-by-side or stacked image table
            img_row = img_cells
            lbl_row = img_labels
            col_w = 540 / len(img_cells)
            img_table = Table([img_row, lbl_row], colWidths=[col_w] * len(img_cells))
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(img_table)
        else:
            elements.append(Paragraph("<i>No photographic images or visual explainability artifacts available for this case.</i>", cell_normal))

        elements.append(Spacer(1, 10))

        # 4. Clinical Review & Physician Observations
        elements.append(Paragraph("4. CLINICAL REVIEW & PHYSICIAN OBSERVATIONS", section_style))

        if case.reviews and len(case.reviews) > 0:
            for rev in case.reviews:
                doc_name = f"Dr. {rev.doctor.full_name or rev.doctor.username}" if rev.doctor else "Clinician"
                rev_date = rev.created_at.strftime("%Y-%m-%d %H:%M UTC") if rev.created_at else "N/A"
                rec_text = rev.recommendation if rev.recommendation else "None provided."

                rev_table_data = [
                    [
                        Paragraph(f"<b>Reviewer:</b> {doc_name}", cell_bold),
                        Paragraph(f"<b>Date:</b> {rev_date}", cell_normal),
                        Paragraph(f"<b>Review Outcome:</b> {rev.review_status.upper()}", cell_bold)
                    ],
                    [
                        Paragraph("<b>Clinical Observations:</b>", cell_bold),
                        Paragraph(rev.comment, cell_normal),
                        Paragraph("", cell_normal)
                    ],
                    [
                        Paragraph("<b>Recommendations:</b>", cell_bold),
                        Paragraph(rec_text, cell_normal),
                        Paragraph("", cell_normal)
                    ]
                ]
                t_rev = Table(rev_table_data, colWidths=[120, 290, 130])
                t_rev.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ('SPAN', (1, 1), (2, 1)),
                    ('SPAN', (1, 2), (2, 2)),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                elements.append(t_rev)
                elements.append(Spacer(1, 6))
        else:
            elements.append(Paragraph("<i>This case is currently awaiting clinical physician review.</i>", cell_normal))

        elements.append(Spacer(1, 12))

        # 5. Prominent Medical Safety Disclaimer (Callout Box)
        elements.append(KeepTogether([
            Table(
                [[Paragraph(DISCLAIMER_TEXT, disclaimer_style)]],
                colWidths=[540],
                style=TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#EF4444")),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ])
            )
        ]))

        doc.build(elements, canvasmaker=NumberedCanvas)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        log_report_generation(case.id, "pdf", current_user.id)
        return pdf_bytes

    except Exception as exc:
        log_report_failure(case.id, "pdf", str(exc))
        raise


def generate_case_csv(cases: List[Case], current_user: User) -> str:
    """
    Generate RFC 4180 compliant CSV export containing relevant case and prediction details.
    Enforces authorization: patients receive only their records, researchers receive no PHI.
    """
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    can_view_phi = current_user.role in [UserRole.CLINICIAN, UserRole.ADMIN, UserRole.PATIENT]
    is_researcher = current_user.role in [UserRole.RESEARCHER, UserRole.ML_ENGINEER]

    headers = [
        "case_id",
        "case_code",
        "created_at",
        "body_location",
        "case_status",
        "review_status",
        "assigned_doctor"
    ]

    if can_view_phi and not is_researcher:
        headers.extend(["patient_age", "patient_gender", "known_allergies", "clinical_notes"])

    headers.extend([
        "image_quality_status",
        "model_name",
        "predicted_class",
        "predicted_disease_name",
        "confidence_score_uncalibrated",
        "risk_level",
        "clinical_review_count",
        "latest_clinician_comment"
    ])

    writer.writerow(headers)

    for case in cases:
        latest_pred = case.predictions[-1] if case.predictions else None
        latest_rev = case.reviews[0] if case.reviews else None
        latest_img = case.images[-1] if case.images else None

        row = [
            case.id,
            case.case_code,
            case.created_at.isoformat() if case.created_at else "",
            case.body_location,
            case.status.value if hasattr(case.status, "value") else str(case.status),
            case.review_status,
            case.assigned_doctor.full_name if case.assigned_doctor else "Unassigned"
        ]

        if can_view_phi and not is_researcher:
            row.extend([
                case.patient_age if case.patient_age is not None else "",
                case.patient_gender or "",
                case.known_allergies or "",
                (case.clinical_notes or "").replace("\n", " ")
            ])

        row.append(latest_img.quality_status if latest_img else "No Image")

        if latest_pred:
            row.extend([
                latest_pred.model_name,
                latest_pred.predicted_class,
                latest_pred.predicted_disease_name,
                f"{latest_pred.confidence_score:.4f}",
                latest_pred.risk_level
            ])
        else:
            row.extend(["", "", "", "", ""])

        row.append(len(case.reviews) if case.reviews else 0)
        row.append((latest_rev.comment or "").replace("\n", " ") if latest_rev else "")

        writer.writerow(row)

    csv_data = output.getvalue()
    output.close()
    return csv_data


def generate_case_json(cases: List[Case], current_user: User) -> List[Dict[str, Any]]:
    """
    Generate structured JSON export for individual cases or collections of cases.
    Respects role authorization boundaries.
    """
    can_view_phi = current_user.role in [UserRole.CLINICIAN, UserRole.ADMIN, UserRole.PATIENT]
    is_researcher = current_user.role in [UserRole.RESEARCHER, UserRole.ML_ENGINEER]

    results = []
    for case in cases:
        latest_img = case.images[-1] if case.images else None
        case_data = {
            "id": case.id,
            "case_code": case.case_code,
            "body_location": case.body_location,
            "status": case.status.value if hasattr(case.status, "value") else str(case.status),
            "review_status": case.review_status,
            "is_archived": case.is_archived,
            "created_at": case.created_at.isoformat() if case.created_at else None,
            "updated_at": case.updated_at.isoformat() if case.updated_at else None,
            "assigned_doctor": {
                "id": case.assigned_doctor.id,
                "name": case.assigned_doctor.full_name or case.assigned_doctor.username,
                "institution": case.assigned_doctor.institution
            } if case.assigned_doctor else None,
            "image_quality": {
                "quality_status": latest_img.quality_status,
                "is_suitable_for_ai": latest_img.is_suitable_for_ai,
                "blur_score": latest_img.blur_score,
                "brightness_score": latest_img.brightness_score,
                "contrast_score": latest_img.contrast_score,
                "resolution": f"{latest_img.width}x{latest_img.height}" if latest_img.width else None
            } if latest_img else None,
            "disclaimer": "DermAI is a clinical decision support system, not an autonomous diagnostic device. Softmax scores are uncalibrated."
        }

        # Include patient demographics if authorized
        if can_view_phi and not is_researcher:
            case_data["patient"] = {
                "age": case.patient_age,
                "gender": case.patient_gender,
                "known_allergies": case.known_allergies,
                "medical_history": case.medical_history,
                "clinical_notes": case.clinical_notes
            }

        # Predictions
        preds = []
        for p in case.predictions:
            probs = {}
            if p.class_probabilities_json:
                try:
                    probs = json.loads(p.class_probabilities_json)
                except Exception:
                    probs = {}
            sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            preds.append({
                "id": p.id,
                "model_name": p.model_name,
                "predicted_class": p.predicted_class,
                "predicted_disease_name": p.predicted_disease_name,
                "confidence_score": p.confidence_score,
                "confidence_score_label": "Uncalibrated Softmax Score (0.0 - 1.0)",
                "risk_level": p.risk_level,
                "class_probabilities": probs,
                "top_three_predictions": [
                    {"class_label": k, "probability": v, "disease_name": DISEASE_NAME_MAP.get(k, k)}
                    for k, v in sorted_probs[:3]
                ],
                "top_five_predictions": [
                    {"class_label": k, "probability": v, "disease_name": DISEASE_NAME_MAP.get(k, k)}
                    for k, v in sorted_probs[:5]
                ],
                "created_at": p.created_at.isoformat() if p.created_at else None
            })
        case_data["predictions"] = preds

        # Clinical reviews
        revs = []
        for r in case.reviews:
            revs.append({
                "id": r.id,
                "doctor_name": r.doctor.full_name if r.doctor else "Clinician",
                "doctor_institution": r.doctor.institution if r.doctor else None,
                "comment": r.comment,
                "recommendation": r.recommendation,
                "review_status": r.review_status,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })
        case_data["reviews"] = revs

        results.append(case_data)

    return results
