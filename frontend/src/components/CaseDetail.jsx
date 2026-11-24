import React, { useState, useEffect } from "react";
import {
  ArrowLeft,
  RefreshCw,
  Camera,
  Calendar,
  BrainCircuit,
  AlertTriangle,
  CheckCircle2,
  Sparkles,
  ShieldCheck,
  MessageSquare,
  UserCheck,
  Send,
  Download,
  BellRing,
  Plus,
  Trash2,
  MapPin
} from "lucide-react";
import ImageUploader from "./ImageUploader";
import PredictionResults from "./PredictionResults";
import PredictionHistory from "./PredictionHistory";
import { api } from "../api";

export default function CaseDetail({ currentCase, onBack, onCaseUpdated, user }) {
  const [showReupload, setShowReupload] = useState(false);
  const [predictions, setPredictions] = useState(currentCase?.predictions || []);
  const [activePrediction, setActivePrediction] = useState(
    currentCase?.predictions && currentCase.predictions.length > 0
      ? currentCase.predictions[currentCase.predictions.length - 1]
      : null
  );
  const [isPredicting, setIsPredicting] = useState(false);
  const [selectedModel, setSelectedModel] = useState("mobilenet");
  const [predictionError, setPredictionError] = useState("");
  const [predictionSuccessMessage, setPredictionSuccessMessage] = useState("");

  // Clinical Review State
  const [reviews, setReviews] = useState(currentCase?.reviews || []);
  const [reviewStatus, setReviewStatus] = useState(currentCase?.review_status || "pending_review");
  const [assignedDoctorId, setAssignedDoctorId] = useState(currentCase?.assigned_doctor_id || null);
  const [assignedDoctorName, setAssignedDoctorName] = useState(currentCase?.assigned_doctor_name || "");
  const [newComment, setNewComment] = useState("");
  const [newRecommendation, setNewRecommendation] = useState("");
  const [selectedReviewStatus, setSelectedReviewStatus] = useState("reviewed");
  const [isSubmittingReview, setIsSubmittingReview] = useState(false);
  const [isReleasingCase, setIsReleasingCase] = useState(false);
  const [isArchivingCase, setIsArchivingCase] = useState(false);
  const [reviewError, setReviewError] = useState("");
  const [reviewSuccess, setReviewSuccess] = useState("");
  const [isClaimingCase, setIsClaimingCase] = useState(false);

  // Reporting & Export State
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [isExportingCsv, setIsExportingCsv] = useState(false);
  const [isExportingJson, setIsExportingJson] = useState(false);
  const [reportError, setReportError] = useState("");
  const [reportSuccess, setReportSuccess] = useState("");

  // Follow-Up Reminders State (Milestone 4 API integration)
  const [reminders, setReminders] = useState([]);
  const [isAddingReminder, setIsAddingReminder] = useState(false);
  const [reminderDate, setReminderDate] = useState("");
  const [reminderType, setReminderType] = useState("Dermoscopy Surveillance");
  const [reminderNotes, setReminderNotes] = useState("");
  const [reminderLoading, setReminderLoading] = useState(false);
  const [reminderFeedback, setReminderFeedback] = useState("");

  const isClinician = user && (user.role === "clinician" || user.role === "admin");
  const isPatient = user && user.role === "patient";

  const handleDownloadPdf = async () => {
    if (!currentCase?.id || isDownloadingPdf) return;
    setIsDownloadingPdf(true);
    setReportError("");
    setReportSuccess("");
    try {
      await api.downloadCasePdf(currentCase.id, currentCase.case_code);
      setReportSuccess("PDF report generated successfully.");
      setTimeout(() => setReportSuccess(""), 4000);
    } catch (err) {
      setReportError(err.message || "Failed to download PDF report");
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const handleExportCsv = async () => {
    if (!currentCase?.id || isExportingCsv) return;
    setIsExportingCsv(true);
    setReportError("");
    try {
      await api.exportCaseCsv(currentCase.id, currentCase.case_code);
      setReportSuccess("Case exported to CSV successfully.");
      setTimeout(() => setReportSuccess(""), 4000);
    } catch (err) {
      setReportError(err.message || "Failed to export CSV");
    } finally {
      setIsExportingCsv(false);
    }
  };

  const handleExportJson = async () => {
    if (!currentCase?.id || isExportingJson) return;
    setIsExportingJson(true);
    setReportError("");
    try {
      await api.exportCaseJson(currentCase.id, currentCase.case_code);
      setReportSuccess("Case exported to JSON successfully.");
      setTimeout(() => setReportSuccess(""), 4000);
    } catch (err) {
      setReportError(err.message || "Failed to export JSON");
    } finally {
      setIsExportingJson(false);
    }
  };

  // Synchronize and load predictions, reviews, & reminders from backend
  useEffect(() => {
    let mounted = true;
    if (currentCase?.id) {
      api.getCasePredictions(currentCase.id)
        .then((data) => {
          if (mounted && Array.isArray(data)) {
            setPredictions(data);
            if (data.length > 0) {
              setActivePrediction(data[data.length - 1]);
            }
          }
        })
        .catch((err) => {
          console.warn("Could not fetch case predictions:", err);
        });

      api.getCaseReviews(currentCase.id)
        .then((revs) => {
          if (mounted && Array.isArray(revs)) {
            setReviews(revs);
          }
        })
        .catch((err) => {
          console.warn("Could not fetch case reviews:", err);
        });

      api.getCaseReminders(currentCase.id)
        .then((rems) => {
          if (mounted && Array.isArray(rems)) {
            setReminders(rems);
          }
        })
        .catch((err) => {
          console.warn("Could not fetch reminders:", err);
        });
    }
    return () => {
      mounted = false;
    };
  }, [currentCase?.id]);

  if (!currentCase) return null;

  const image =
    currentCase.images && currentCase.images.length > 0
      ? currentCase.images[currentCase.images.length - 1]
      : null;

  const getStatusBadge = (status) => {
    switch (status) {
      case "preprocessed":
        return <span className="badge badge-preprocessed">Preprocessed</span>;
      case "completed":
        return <span className="badge badge-teal">AI Analyzed</span>;
      default:
        return <span className="badge badge-awaiting">Awaiting Photo</span>;
    }
  };

  const getReviewStatusBadge = (rStatus) => {
    switch (rStatus) {
      case "reviewed":
        return <span className="badge badge-low">Reviewed</span>;
      case "under_review":
        return <span className="badge badge-teal">Under Review</span>;
      case "follow_up_required":
        return <span className="badge badge-high">Follow-up Required</span>;
      default:
        return <span className="badge badge-awaiting">Pending Review</span>;
    }
  };

  const getQualityBadge = (qualityStatus) => {
    if (qualityStatus === "Good Quality" || qualityStatus?.includes("Good")) {
      return <span className="badge badge-low">{qualityStatus}</span>;
    } else if (qualityStatus === "Acceptable Quality" || qualityStatus?.includes("Acceptable")) {
      return <span className="badge badge-moderate">{qualityStatus}</span>;
    }
    return <span className="badge badge-high">{qualityStatus || "Poor Quality"}</span>;
  };

  const handleRunPrediction = async () => {
    if (!currentCase?.id || isPredicting) return;
    setIsPredicting(true);
    setPredictionError("");
    setPredictionSuccessMessage("");

    try {
      const result = await api.predictCase(currentCase.id, selectedModel);
      const updatedList = [...predictions.filter((p) => p.id !== result.id), result];
      setPredictions(updatedList);
      setActivePrediction(result);
      setPredictionSuccessMessage(
        `AI prediction completed: ${result.predicted_disease_name} (${(result.confidence_score * 100).toFixed(1)}%, ${result.risk_level})`
      );

      if (onCaseUpdated) {
        onCaseUpdated({
          ...currentCase,
          status: "completed",
          predictions: updatedList,
        });
      }
    } catch (err) {
      setPredictionError(err.message || "Failed to execute AI prediction on case image");
    } finally {
      setIsPredicting(false);
    }
  };

  const handleSubmitReview = async (e) => {
    e?.preventDefault();
    if (!newComment.trim() || isSubmittingReview) return;
    setIsSubmittingReview(true);
    setReviewError("");
    setReviewSuccess("");

    try {
      const reviewPayload = {
        comment: newComment.trim(),
        recommendation: newRecommendation.trim() || null,
        review_status: selectedReviewStatus,
      };
      const createdReview = await api.addCaseReview(currentCase.id, reviewPayload);
      const updatedReviews = [createdReview, ...reviews.filter((r) => r.id !== createdReview.id)];
      setReviews(updatedReviews);
      setReviewStatus(selectedReviewStatus);
      if (!assignedDoctorId && user) {
        setAssignedDoctorId(user.id || user.user_id);
        setAssignedDoctorName(user.full_name || user.username);
      }
      setNewComment("");
      setNewRecommendation("");
      setReviewSuccess("Clinical review and observation submitted successfully.");
      setTimeout(() => setReviewSuccess(""), 4000);

      if (onCaseUpdated) {
        onCaseUpdated({
          ...currentCase,
          review_status: selectedReviewStatus,
          assigned_doctor_id: assignedDoctorId || user?.id || user?.user_id,
          assigned_doctor_name: assignedDoctorName || user?.full_name || user?.username,
          reviews: updatedReviews,
        });
      }
    } catch (err) {
      setReviewError(err.message || "Failed to submit clinical review");
    } finally {
      setIsSubmittingReview(false);
    }
  };

  const handleClaimCase = async () => {
    if (!currentCase?.id || isClaimingCase) return;
    setIsClaimingCase(true);
    setReviewError("");
    try {
      const updated = await api.assignCaseDoctor(currentCase.id);
      setAssignedDoctorId(user.id || user.user_id);
      setAssignedDoctorName(user.full_name || user.username);
      if (reviewStatus === "pending_review") {
        setReviewStatus("under_review");
      }
      setReviewSuccess(`Case assigned to Dr. ${user.full_name || user.username}.`);
      setTimeout(() => setReviewSuccess(""), 4000);
      if (onCaseUpdated) {
        onCaseUpdated(updated);
      }
    } catch (err) {
      setReviewError(err.message || "Failed to assign case");
    } finally {
      setIsClaimingCase(false);
    }
  };

  const handleReleaseCase = async () => {
    if (!currentCase?.id || isReleasingCase) return;
    if (!window.confirm("Release this case back to the unassigned queue?")) {
      return;
    }
    setIsReleasingCase(true);
    setReviewError("");
    try {
      const updated = await api.releaseCase(currentCase.id);
      setAssignedDoctorId(null);
      setAssignedDoctorName("");
      setReviewStatus("pending_review");
      setReviewSuccess("Case released to unassigned queue.");
      setTimeout(() => setReviewSuccess(""), 4000);
      if (onCaseUpdated) {
        onCaseUpdated(updated);
      }
    } catch (err) {
      setReviewError(err.message || "Failed to release case");
    } finally {
      setIsReleasingCase(false);
    }
  };

  const handleArchiveCase = async () => {
    if (!currentCase?.id || isArchivingCase) return;
    if (!window.confirm("Archive this case? All associated records will be preserved.")) {
      return;
    }
    setIsArchivingCase(true);
    setReviewError("");
    try {
      await api.deleteCase(currentCase.id);
      setReviewSuccess("Case archived successfully.");
      setTimeout(() => {
        if (onBack) onBack();
      }, 1200);
    } catch (err) {
      setReviewError(err.message || "Failed to archive case");
      setIsArchivingCase(false);
    }
  };

  const handleCreateReminder = async (e) => {
    e.preventDefault();
    if (!reminderDate || reminderLoading) return;
    setReminderLoading(true);
    setReminderFeedback("");
    try {
      const payload = {
        reminder_date: reminderDate,
        reminder_type: reminderType,
        notes: reminderNotes.trim() || null,
      };
      const created = await api.createFollowUpReminder(currentCase.id, payload);
      setReminders([created, ...reminders]);
      setIsAddingReminder(false);
      setReminderDate("");
      setReminderNotes("");
      setReminderFeedback("Recall reminder scheduled successfully.");
      setTimeout(() => setReminderFeedback(""), 4000);
    } catch (err) {
      setReminderFeedback(err.message || "Failed to create reminder");
    } finally {
      setReminderLoading(false);
    }
  };

  const handleUpdateReminderStatus = async (reminderId, newStatus) => {
    try {
      const updated = await api.updateReminderStatus(reminderId, newStatus);
      setReminders(reminders.map((r) => (r.id === reminderId ? updated : r)));
    } catch (err) {
      console.warn("Failed to update reminder status:", err);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Top Action Bar */}
      <div className="toolbar-row">
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button
            onClick={onBack}
            className="btn btn-secondary btn-sm"
          >
            <ArrowLeft size={15} />
            <span>Case Queue</span>
          </button>

          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Case: <strong style={{ color: "var(--text-main)", fontFamily: "var(--font-mono)" }}>{currentCase.case_code}</strong>
            </span>
            {getStatusBadge(currentCase.status)}
            {getReviewStatusBadge(reviewStatus)}
          </div>
        </div>

        {/* Consolidated Action Buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {reportFeedbackOrError(reportSuccess, reportError)}

          <button
            onClick={handleDownloadPdf}
            disabled={isDownloadingPdf}
            className="btn btn-secondary btn-sm"
            title="Download PDF Clinical Summary Report"
          >
            <Download size={14} />
            <span>{isDownloadingPdf ? "Compiling..." : "PDF Report"}</span>
          </button>

          <button
            onClick={handleExportCsv}
            disabled={isExportingCsv}
            className="btn btn-ghost btn-sm"
            title="Export Case CSV"
          >
            <span>CSV</span>
          </button>

          <button
            onClick={handleExportJson}
            disabled={isExportingJson}
            className="btn btn-ghost btn-sm"
            title="Export Case JSON"
          >
            <span>JSON</span>
          </button>

          {isClinician && (
            assignedDoctorId ? (
              (assignedDoctorId === (user?.id || user?.user_id) || (user && assignedDoctorId === Number(user.id || user.user_id))) ? (
                <button
                  onClick={handleReleaseCase}
                  disabled={isReleasingCase}
                  className="btn btn-secondary btn-sm"
                >
                  <span>Release Triage</span>
                </button>
              ) : null
            ) : (
              <button
                onClick={handleClaimCase}
                disabled={isClaimingCase}
                className="btn btn-primary btn-sm"
              >
                <UserCheck size={14} />
                <span>Claim Case</span>
              </button>
            )
          )}

          {user?.role === "admin" && (
            <button
              onClick={handleArchiveCase}
              disabled={isArchivingCase}
              className="btn btn-ghost btn-sm"
              style={{ color: "var(--error)" }}
              title="Archive Case"
            >
              <Trash2 size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Main Workstation 2-Column Split */}
      <div className="workstation-split">
        {/* LEFT COLUMN: Patient Demographics, Imagery Inspector, IQA & Reminders */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Patient Demographics Card */}
          <div className="medical-card" style={{ padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
              <div>
                <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Patient Demographics
                </h2>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 1 }}>
                  Registered clinical metadata
                </p>
              </div>

              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                <Calendar size={12} />
                <span>{new Date(currentCase.created_at).toLocaleDateString()}</span>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
              <div style={{ background: "var(--bg-surface-subtle)", padding: "8px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
                <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Age & Sex</span>
                <div style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--text-main)", marginTop: 1 }}>
                  {currentCase.patient_age ? `${currentCase.patient_age} yrs` : "N/A"} • <span style={{ textTransform: "capitalize" }}>{currentCase.patient_gender || "Unknown"}</span>
                </div>
              </div>

              <div style={{ background: "var(--bg-surface-subtle)", padding: "8px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
                <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Anatomical Site</span>
                <div style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--text-main)", textTransform: "capitalize", display: "flex", alignItems: "center", gap: 4, marginTop: 1 }}>
                  <MapPin size={13} color="var(--primary-hover)" />
                  <span>{currentCase.body_location}</span>
                </div>
              </div>
            </div>

            {currentCase.known_allergies && (
              <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: 6 }}>
                <strong style={{ color: "var(--text-main)" }}>Allergies: </strong>{currentCase.known_allergies}
              </div>
            )}

            {currentCase.medical_history && (
              <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: 6 }}>
                <strong style={{ color: "var(--text-main)" }}>History: </strong>{currentCase.medical_history}
              </div>
            )}

            {currentCase.clinical_notes && (
              <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 6, padding: "8px 10px", background: "var(--bg-surface-subtle)", borderRadius: "var(--radius-xs)", border: "1px solid var(--border-color)" }}>
                <strong style={{ color: "var(--text-main)" }}>Clinical Notes: </strong>{currentCase.clinical_notes}
              </div>
            )}
          </div>

          {/* Lesion Photography Inspector Card */}
          <div className="medical-card" style={{ padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Camera size={16} color="var(--primary-hover)" />
                <h3 style={{ fontSize: "0.98rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Lesion Photography
                </h3>
              </div>
              <button
                onClick={() => setShowReupload(!showReupload)}
                className="btn btn-ghost btn-sm"
                style={{ fontSize: "0.72rem", padding: "2px 6px" }}
              >
                {showReupload ? "Cancel" : "Re-Upload"}
              </button>
            </div>

            {showReupload ? (
              <ImageUploader
                caseId={currentCase.id}
                onUploadSuccess={(updated) => {
                  setShowReupload(false);
                  onCaseUpdated(updated);
                }}
              />
            ) : image ? (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
                  {/* Original */}
                  <div>
                    <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 4 }}>
                      Original Capture
                    </span>
                    <div
                      style={{
                        height: 150,
                        borderRadius: "var(--radius-sm)",
                        overflow: "hidden",
                        background: "#0F172A",
                        border: "1px solid var(--border-color)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <img
                        src={api.getImageUrl(image.original_url)}
                        alt="Original"
                        style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                      />
                    </div>
                  </div>

                  {/* Preprocessed */}
                  <div>
                    <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 4 }}>
                      Standardized Tensor (224px)
                    </span>
                    <div
                      style={{
                        height: 150,
                        borderRadius: "var(--radius-sm)",
                        overflow: "hidden",
                        background: "#0F172A",
                        border: "1px solid var(--border-color)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <img
                        src={api.getImageUrl(image.preprocessed_url)}
                        alt="Preprocessed"
                        style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                      />
                    </div>
                  </div>
                </div>

                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "flex", justifyContent: "space-between" }}>
                  <span>Filename: {image.original_filename || "capture.jpg"}</span>
                  <span>{new Date(image.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            ) : (
              <div style={{ padding: 24, textAlign: "center", color: "var(--text-muted)", fontSize: "0.82rem" }}>
                No photograph attached to this case.
                <button
                  onClick={() => setShowReupload(true)}
                  className="btn btn-primary btn-sm"
                  style={{ marginTop: 10, display: "block", margin: "10px auto 0" }}
                >
                  Upload Photograph
                </button>
              </div>
            )}
          </div>

          {/* IQA Telemetry Card */}
          {image && (
            <div className="medical-card" style={{ padding: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <span style={{ fontSize: "0.92rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Image Quality Assessment (IQA)
                </span>
                {getQualityBadge(image.quality_status)}
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
                <div style={{ background: "var(--bg-surface-subtle)", padding: "8px 10px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
                  <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Sharpness</div>
                  <div style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-main)" }}>
                    {image.blur_score ? image.blur_score.toFixed(1) : "--"}
                  </div>
                  <div style={{ fontSize: "0.7rem", color: image.blur_score > 100 ? "var(--success-text)" : "var(--warning-text)" }}>
                    {image.blur_score > 100 ? "✓ Sharp focus (>100)" : "Moderate focus"}
                  </div>
                </div>

                <div style={{ background: "var(--bg-surface-subtle)", padding: "8px 10px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
                  <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Illumination</div>
                  <div style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-main)" }}>
                    {image.brightness_score ? Math.round(image.brightness_score) : "--"} / 255
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--success-text)" }}>
                    ✓ Normal exposure
                  </div>
                </div>
              </div>

              <div style={{ fontSize: "0.74rem", color: "var(--text-secondary)", background: "var(--bg-surface-subtle)", padding: "8px 10px", borderRadius: "var(--radius-xs)", border: "1px solid var(--border-color)" }}>
                <strong>Quality Notes: </strong>{image.quality_notes || "Meets clinical screening requirements."}
              </div>
            </div>
          )}

          {/* Follow-Up Reminders Panel (Milestone 4 API integration) */}
          <div className="medical-card" style={{ padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <BellRing size={16} color="var(--primary-hover)" />
                <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Follow-Up Surveillance ({reminders.length})
                </h3>
              </div>
              {isClinician && (
                <button
                  onClick={() => setIsAddingReminder(!isAddingReminder)}
                  className="btn btn-secondary btn-sm"
                  style={{ fontSize: "0.72rem", padding: "3px 8px" }}
                >
                  <Plus size={13} />
                  <span>{isAddingReminder ? "Cancel" : "Add Recall"}</span>
                </button>
              )}
            </div>

            {reminderFeedback && (
              <div style={{ fontSize: "0.75rem", color: "var(--text-main)", fontWeight: 600, marginBottom: 8 }}>
                {reminderFeedback}
              </div>
            )}

            {isAddingReminder && (
              <form onSubmit={handleCreateReminder} style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 12, padding: 12, background: "var(--bg-surface-subtle)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.75rem" }}>Recall Date</label>
                  <input
                    type="date"
                    value={reminderDate}
                    onChange={(e) => setReminderDate(e.target.value)}
                    className="form-control"
                    style={{ height: 32, fontSize: "0.78rem" }}
                    required
                  />
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.75rem" }}>Surveillance Protocol</label>
                  <select
                    value={reminderType}
                    onChange={(e) => setReminderType(e.target.value)}
                    className="form-control"
                    style={{ height: 32, fontSize: "0.78rem" }}
                  >
                    <option value="Dermoscopy Surveillance">3-Month Repeat Dermoscopy</option>
                    <option value="Biopsy Procedure">Biopsy / Histopathology</option>
                    <option value="Clinical Examination">Routine Dermatological Follow-up</option>
                    <option value="Annual Skin Check">Annual Whole-Body Skin Check</option>
                  </select>
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.75rem" }}>Clinical Instructions (Optional)</label>
                  <input
                    type="text"
                    value={reminderNotes}
                    onChange={(e) => setReminderNotes(e.target.value)}
                    placeholder="e.g. Inspect border change"
                    className="form-control"
                    style={{ height: 32, fontSize: "0.78rem" }}
                  />
                </div>

                <button
                  type="submit"
                  disabled={reminderLoading || !reminderDate}
                  className="btn btn-primary btn-sm"
                  style={{ marginTop: 4 }}
                >
                  <span>{reminderLoading ? "Scheduling..." : "Schedule Surveillance Recall"}</span>
                </button>
              </form>
            )}

            {reminders.length === 0 ? (
              <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", textAlign: "center", padding: "12px 0" }}>
                No active follow-up reminders scheduled.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {reminders.map((rem) => (
                  <div
                    key={rem.id}
                    style={{
                      padding: "8px 10px",
                      borderRadius: "var(--radius-xs)",
                      background: rem.status === "completed" ? "var(--success-bg)" : "var(--bg-surface-subtle)",
                      border: "1px solid var(--border-color)",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 8,
                    }}
                  >
                    <div>
                      <strong style={{ fontSize: "0.78rem", color: "var(--text-main)", display: "block" }}>
                        {rem.reminder_type}
                      </strong>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                        Due: {new Date(rem.reminder_date).toLocaleDateString()}
                      </span>
                    </div>

                    {isClinician && (
                      rem.status !== "completed" ? (
                        <button
                          onClick={() => handleUpdateReminderStatus(rem.id, "completed")}
                          className="btn btn-ghost btn-sm"
                          style={{ fontSize: "0.68rem", padding: "2px 6px" }}
                        >
                          Mark Done
                        </button>
                      ) : (
                        <span className="badge badge-low" style={{ fontSize: "0.65rem" }}>Completed</span>
                      )
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: Inference Engine, Explainability, Predictions & Physician Sign-Off */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Inference Engine & Model Selector Card */}
          <div className="medical-card" style={{ padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, marginBottom: 14 }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <BrainCircuit size={18} color="var(--primary-hover)" />
                  <h3 style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-main)" }}>
                    Deep Learning Decision Support
                  </h3>
                </div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Execute 7-class lesion inference with MobileNetV2 or Vision Transformer
                </p>
              </div>

              {/* Model Select and Run Button */}
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={isPredicting}
                  className="form-control"
                  style={{ width: "auto", minWidth: 160, height: 38 }}
                >
                  <option value="mobilenet">MobileNetV2 (Baseline)</option>
                  <option value="efficientnet">EfficientNetB0</option>
                  <option value="vit">Vision Transformer (ViT)</option>
                  <option value="ensemble">Soft-Voting Ensemble</option>
                </select>

                <button
                  type="button"
                  onClick={handleRunPrediction}
                  disabled={isPredicting || !image}
                  className="btn btn-primary"
                  style={{ height: 38 }}
                >
                  {isPredicting ? (
                    <>
                      <RefreshCw size={15} className="spin" />
                      <span>Inference Running...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} />
                      <span>{predictions.length > 0 ? "Re-Analyze Lesion" : "Run AI Prediction"}</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {predictionError && (
              <div
                style={{
                  padding: "10px 14px",
                  background: "var(--error-bg)",
                  border: "1px solid var(--error-border)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--error-text)",
                  fontSize: "0.82rem",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  marginBottom: 12,
                }}
              >
                <AlertTriangle size={16} />
                <span>{predictionError}</span>
              </div>
            )}

            {predictionSuccessMessage && (
              <div
                style={{
                  padding: "10px 14px",
                  background: "var(--success-bg)",
                  border: "1px solid var(--success-border)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--success-text)",
                  fontSize: "0.82rem",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <CheckCircle2 size={16} />
                <span>{predictionSuccessMessage}</span>
              </div>
            )}
          </div>

          {/* Active Prediction & Explainability */}
          {activePrediction ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <PredictionResults
                prediction={activePrediction}
                isLatest={activePrediction.id === predictions[predictions.length - 1]?.id}
                caseImage={image}
                caseId={currentCase?.id}
                reviewStatus={reviewStatus}
              />

              <PredictionHistory
                predictions={predictions}
                selectedPredictionId={activePrediction.id}
                onSelectPrediction={(pred) => {
                  setActivePrediction(pred);
                  setPredictionSuccessMessage("");
                }}
              />
            </div>
          ) : (
            <div
              className="medical-card"
              style={{
                padding: "36px 20px",
                textAlign: "center",
                background: "var(--bg-surface-subtle)",
              }}
            >
              <BrainCircuit size={32} color="var(--primary-hover)" style={{ margin: "0 auto 10px" }} />
              <h4 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 4 }}>
                Ready for Automated Deep Learning Analysis
              </h4>
              <p style={{ fontSize: "0.84rem", color: "var(--text-secondary)", maxWidth: 440, margin: "0 auto 16px", lineHeight: 1.5 }}>
                Click <strong>"Run AI Prediction"</strong> above to evaluate the lesion tensor across 7 disease categories with clinical triage risk classification.
              </p>
              <button
                type="button"
                onClick={handleRunPrediction}
                disabled={isPredicting || !image}
                className="btn btn-primary"
              >
                <Sparkles size={16} />
                <span>Initiate AI Diagnostic Screening</span>
              </button>
            </div>
          )}

          {/* Attending Physician Review & Oversight Card */}
          <div className="medical-card" style={{ padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14, paddingBottom: 10, borderBottom: "1px solid var(--border-subtle)" }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <MessageSquare size={17} color="var(--primary-hover)" />
                  <h3 style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-main)" }}>
                    Physician Review & Clinical Observations
                  </h3>
                  {getReviewStatusBadge(reviewStatus)}
                </div>
                <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Human practitioner oversight strictly separated from AI decision-support activations
                </p>
              </div>

              {assignedDoctorName && (
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--text-main)",
                    background: "var(--primary-light)",
                    border: "1px solid var(--primary-light-border)",
                    padding: "3px 10px",
                    borderRadius: "var(--radius-pill)",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 5,
                  }}
                >
                  <UserCheck size={13} />
                  <span>Dr. {assignedDoctorName}</span>
                </span>
              )}
            </div>

            {reviewError && (
              <div
                style={{
                  padding: "10px 14px",
                  background: "var(--error-bg)",
                  border: "1px solid var(--error-border)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--error-text)",
                  fontSize: "0.82rem",
                  marginBottom: 12,
                }}
              >
                {reviewError}
              </div>
            )}

            {reviewSuccess && (
              <div
                style={{
                  padding: "10px 14px",
                  background: "var(--success-bg)",
                  border: "1px solid var(--success-border)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--success-text)",
                  fontSize: "0.82rem",
                  marginBottom: 12,
                }}
              >
                {reviewSuccess}
              </div>
            )}

            {/* Clinician Review Submission Form */}
            {isClinician && (
              <form onSubmit={handleSubmitReview} style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 16 }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">
                    <span>Clinical Diagnostic Observation</span>
                    <span className="form-label-hint">Physician evaluation</span>
                  </label>
                  <textarea
                    value={newComment}
                    onChange={(e) => setNewComment(e.target.value)}
                    placeholder="Record formal clinical impression, dermoscopic patterns observed, and differential diagnosis..."
                    className="form-control"
                    style={{ minHeight: 74 }}
                    required
                  />
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">
                    <span>Recommended Next Steps & Follow-Up</span>
                    <span className="form-label-hint">Action plan</span>
                  </label>
                  <input
                    type="text"
                    value={newRecommendation}
                    onChange={(e) => setNewRecommendation(e.target.value)}
                    placeholder="e.g. Schedule excisional biopsy, repeat dermoscopy in 3 months"
                    className="form-control"
                  />
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontSize: "0.78rem", fontWeight: 600 }}>Review Status:</span>
                    <select
                      value={selectedReviewStatus}
                      onChange={(e) => setSelectedReviewStatus(e.target.value)}
                      className="form-control"
                      style={{ width: "auto", height: 34, fontSize: "0.8rem" }}
                    >
                      <option value="reviewed">Reviewed (Normal / Verified)</option>
                      <option value="under_review">Under Review</option>
                      <option value="follow_up_required">Follow-up Required</option>
                      <option value="pending_review">Pending Review</option>
                    </select>
                  </div>

                  <button
                    type="submit"
                    disabled={isSubmittingReview || !newComment.trim()}
                    className="btn btn-primary"
                    style={{ height: 34, fontSize: "0.84rem" }}
                  >
                    <Send size={14} />
                    <span>{isSubmittingReview ? "Submitting..." : "Submit Review"}</span>
                  </button>
                </div>
              </form>
            )}

            {/* Patient View: Doctor Impression */}
            {isPatient && (
              <div style={{ marginBottom: 14 }}>
                {reviews.length > 0 ? (
                  <div style={{ background: "var(--primary-light)", padding: "14px 16px", borderRadius: "var(--radius-md)", border: "1px solid var(--primary-light-border)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <strong style={{ fontSize: "0.88rem", color: "var(--text-main)" }}>
                        Dr. {reviews[0].doctor_name}
                      </strong>
                      <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                        {new Date(reviews[0].created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <p style={{ fontSize: "0.84rem", color: "var(--text-main)", lineHeight: 1.5 }}>
                      "{reviews[0].comment}"
                    </p>
                    {reviews[0].recommendation && (
                      <div style={{ marginTop: 8, fontSize: "0.8rem", color: "var(--text-main)", fontWeight: 600 }}>
                        Next Steps: {reviews[0].recommendation}
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ textAlign: "center", padding: "16px 0", color: "var(--text-muted)", fontSize: "0.82rem" }}>
                    Your screening case is in the doctor review queue. Doctor observations will appear here once finalized.
                  </div>
                )}
              </div>
            )}

            {/* Review History Timeline */}
            {reviews.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--border-subtle)" }}>
                <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
                  Clinical Review Log ({reviews.length})
                </span>
                {reviews.map((rev) => (
                  <div
                    key={rev.id}
                    style={{
                      background: "var(--bg-surface-subtle)",
                      padding: "10px 12px",
                      borderRadius: "var(--radius-sm)",
                      border: "1px solid var(--border-color)",
                      fontSize: "0.8rem",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <strong style={{ color: "var(--text-main)" }}>{rev.doctor_name}</strong>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                        {new Date(rev.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <p style={{ color: "var(--text-secondary)", lineHeight: 1.45 }}>{rev.comment}</p>
                    {rev.recommendation && (
                      <div style={{ fontSize: "0.74rem", color: "var(--text-main)", marginTop: 4 }}>
                        <strong>Plan: </strong>{rev.recommendation}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 12 }}>
              <ShieldCheck size={14} color="var(--primary-hover)" />
              <span>Doctor-authored observations are permanent medical records. AI predictions serve as decision-support aids.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function reportFeedbackOrError(success, error) {
  if (error) {
    return <span style={{ fontSize: "0.75rem", color: "var(--error-text)" }}>{error}</span>;
  }
  if (success) {
    return <span style={{ fontSize: "0.75rem", color: "var(--success-text)", fontWeight: 600 }}>{success}</span>;
  }
  return null;
}
