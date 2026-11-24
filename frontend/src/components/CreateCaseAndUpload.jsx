import React, { useState, useRef } from "react";
import {
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Camera,
  X,
  ArrowRight,
  Sliders,
  RotateCcw,
  Sparkles,
  User
} from "lucide-react";
import { api } from "../api";

export default function CreateCaseAndUpload({ onCaseCreatedAndProcessed, onViewCaseDetails }) {
  // Section 1: Demographics & Clinical Information
  const [patientAge, setPatientAge] = useState("");
  const [patientGender, setPatientGender] = useState("male");
  const [bodyLocation, setBodyLocation] = useState("back");
  const [knownAllergies, setKnownAllergies] = useState("");
  const [medicalHistory, setMedicalHistory] = useState("");
  const [clinicalNotes, setClinicalNotes] = useState("");

  // Section 2: Image state
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileDetails, setFileDetails] = useState(null);

  // Workflow execution state
  const [processing, setProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState(""); // 'validating' | 'creating_case' | 'uploading_image' | 'completed' | 'upload_failed'
  const [createdCase, setCreatedCase] = useState(null);
  const [processedResultCase, setProcessedResultCase] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [imageError, setImageError] = useState("");
  const inputRef = useRef(null);

  // Image Selection Handler
  const handleFileSelect = (file) => {
    if (!file) return;
    setImageError("");
    setErrorMessage("");

    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      setImageError("Unsupported image format. Please upload JPG, PNG, or WEBP.");
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setImageError("File exceeds 15MB limit. Please upload an image under 15MB.");
      return;
    }

    setSelectedFile(file);
    const objUrl = URL.createObjectURL(file);
    setPreviewUrl(objUrl);

    // Read dimensions
    const img = new Image();
    img.src = objUrl;
    img.onload = () => {
      setFileDetails({
        name: file.name,
        sizeMb: (file.size / (1024 * 1024)).toFixed(2),
        width: img.width,
        height: img.height,
        type: file.type.replace("image/", "").toUpperCase(),
      });
    };
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setFileDetails(null);
    setImageError("");
  };

  // Primary Action: Create Case & Process Image
  const handleCreateCaseAndProcess = async (e) => {
    e.preventDefault();
    if (processing) return;

    setErrorMessage("");
    setImageError("");

    if (patientAge && (parseInt(patientAge, 10) < 1 || parseInt(patientAge, 10) > 120)) {
      setErrorMessage("Please enter a valid patient age between 1 and 120.");
      return;
    }

    if (!selectedFile) {
      setImageError("A skin lesion photograph is required for clinical screening.");
      return;
    }

    setProcessing(true);
    let newCase = null;

    try {
      setProcessingStage("creating_case");
      const payload = {
        patient_age: patientAge ? parseInt(patientAge, 10) : null,
        patient_gender: patientGender,
        body_location: bodyLocation,
        known_allergies: knownAllergies.trim() || null,
        medical_history: medicalHistory.trim() || null,
        clinical_notes: clinicalNotes.trim() || null,
        image_source: "mobile_upload",
      };

      newCase = await api.createCase(payload);
      setCreatedCase(newCase);

      setProcessingStage("uploading_image");
      const updatedCase = await api.uploadCaseImage(newCase.id, selectedFile);

      setProcessingStage("completed");
      setProcessedResultCase(updatedCase);
      if (onCaseCreatedAndProcessed) {
        onCaseCreatedAndProcessed(updatedCase);
      }
    } catch (err) {
      if (newCase) {
        setProcessingStage("upload_failed");
        setErrorMessage(
          `Case ${newCase.case_code} was registered, but image processing encountered an issue: ${err.message || "Network error"}. You can retry uploading below.`
        );
      } else {
        setProcessingStage("");
        setErrorMessage(err.message || "Failed to register screening case. Please check credentials and try again.");
      }
    } finally {
      setProcessing(false);
    }
  };

  const handleRetryUpload = async () => {
    if (!createdCase || !selectedFile || processing) return;

    setProcessing(true);
    setProcessingStage("uploading_image");
    setErrorMessage("");
    setImageError("");

    try {
      const updatedCase = await api.uploadCaseImage(createdCase.id, selectedFile);
      setProcessingStage("completed");
      setProcessedResultCase(updatedCase);
      if (onCaseCreatedAndProcessed) {
        onCaseCreatedAndProcessed(updatedCase);
      }
    } catch (err) {
      setProcessingStage("upload_failed");
      setErrorMessage(
        `Retry failed for case ${createdCase.case_code}: ${err.message || "Upload error"}. Please check file format and try again.`
      );
    } finally {
      setProcessing(false);
    }
  };

  const handleStartFresh = () => {
    setPatientAge("");
    setPatientGender("male");
    setBodyLocation("back");
    setKnownAllergies("");
    setMedicalHistory("");
    setClinicalNotes("");
    handleRemoveImage();
    setCreatedCase(null);
    setProcessedResultCase(null);
    setProcessingStage("");
    setErrorMessage("");
    setImageError("");
  };

  const latestImage =
    processedResultCase && processedResultCase.images && processedResultCase.images.length > 0
      ? processedResultCase.images[processedResultCase.images.length - 1]
      : null;

  return (
    <div style={{ maxWidth: 1040, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div
        className="medical-card"
        style={{
          padding: "24px 28px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 16,
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span className="badge badge-teal" style={{ textTransform: "uppercase" }}>
              Clinical Intake
            </span>
            <span className="badge badge-low">Automated IQA & Standardization</span>
          </div>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-main)" }}>
            New Screening Case & Photography Intake
          </h1>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
            Register patient demographics and attach lesion photography in a single, coordinated clinical intake
          </p>
        </div>
      </div>

      {/* Error Banners */}
      {errorMessage && (
        <div
          style={{
            padding: "12px 16px",
            background: processingStage === "upload_failed" ? "var(--warning-bg)" : "var(--error-bg)",
            border: `1px solid ${processingStage === "upload_failed" ? "var(--warning-border)" : "var(--error-border)"}`,
            borderRadius: "var(--radius-md)",
            color: processingStage === "upload_failed" ? "var(--warning-text)" : "var(--error-text)",
            fontSize: "0.85rem",
            display: "flex",
            alignItems: "flex-start",
            gap: 10,
          }}
        >
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: 1 }} />
          <div style={{ flex: 1, lineHeight: 1.5 }}>{errorMessage}</div>
        </div>
      )}

      {/* SUCCESS CONFIRMATION VIEW */}
      {processedResultCase && latestImage ? (
        <div className="medical-card" style={{ padding: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: "50%",
                background: "var(--success-bg)",
                color: "var(--success-text)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <CheckCircle2 size={24} />
            </div>
            <div>
              <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)" }}>
                Screening Case {processedResultCase.case_code} Successfully Initialized
              </h2>
              <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                Lesion photograph verified, preprocessed to 224×224 tensor, and queued for clinical decision support.
              </p>
            </div>
          </div>

          {/* Dual Image Comparison Preview */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
              gap: 16,
              marginBottom: 20,
            }}
          >
            {/* Original */}
            <div
              style={{
                background: "var(--bg-surface-subtle)",
                borderRadius: "var(--radius-md)",
                padding: 16,
                border: "1px solid var(--border-color)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Original Clinical Capture
                </span>
                <span className="badge badge-neutral" style={{ fontSize: "0.68rem" }}>
                  Source File
                </span>
              </div>
              <div
                style={{
                  height: 220,
                  borderRadius: "var(--radius-sm)",
                  overflow: "hidden",
                  border: "1px solid var(--border-color)",
                  background: "#0F172A",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <img
                  src={api.getImageUrl(latestImage.original_url)}
                  alt="Original"
                  style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                />
              </div>
            </div>

            {/* Standardized Tensor */}
            <div
              style={{
                background: "var(--primary-light)",
                borderRadius: "var(--radius-md)",
                padding: 16,
                border: "1px solid var(--primary-light-border)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Standardized Lesion Tensor (224×224)
                </span>
                <span className="badge badge-teal" style={{ fontSize: "0.68rem" }}>
                  CLAHE Enhanced
                </span>
              </div>
              <div
                style={{
                  height: 220,
                  borderRadius: "var(--radius-sm)",
                  overflow: "hidden",
                  border: "1px solid var(--primary-light-border)",
                  background: "#0F172A",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <img
                  src={api.getImageUrl(latestImage.preprocessed_url)}
                  alt="Preprocessed"
                  style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                />
              </div>
            </div>
          </div>

          {/* IQA Metrics Verification Strip */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: 12,
              marginBottom: 24,
            }}
          >
            <div style={{ background: "var(--bg-surface-subtle)", padding: "12px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Sharpness</span>
              <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-main)", marginTop: 2 }}>
                {latestImage.blur_score ? latestImage.blur_score.toFixed(1) : "--"}
              </div>
              <span style={{ fontSize: "0.72rem", color: "var(--success-text)" }}>✓ Above threshold (&gt;100)</span>
            </div>

            <div style={{ background: "var(--bg-surface-subtle)", padding: "12px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Illumination</span>
              <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-main)", marginTop: 2 }}>
                {latestImage.brightness_score ? Math.round(latestImage.brightness_score) : "--"} / 255
              </div>
              <span style={{ fontSize: "0.72rem", color: "var(--success-text)" }}>✓ Normal exposure</span>
            </div>

            <div style={{ background: "var(--bg-surface-subtle)", padding: "12px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Suitability</span>
              <div style={{ fontSize: "1.1rem", fontWeight: 800, color: latestImage.is_suitable_for_ai ? "var(--success-text)" : "var(--error-text)", marginTop: 2 }}>
                {latestImage.is_suitable_for_ai ? "Suitable" : "Suboptimal"}
              </div>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>AI Screening ready</span>
            </div>
          </div>

          {/* Action CTAs */}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <button
              onClick={() => onViewCaseDetails && onViewCaseDetails(processedResultCase)}
              className="btn btn-primary btn-lg"
            >
              <Sparkles size={18} />
              <span>Open Clinical Workstation & Run AI</span>
              <ArrowRight size={16} />
            </button>

            <button
              onClick={handleStartFresh}
              className="btn btn-secondary btn-lg"
            >
              <RotateCcw size={16} />
              <span>Intake Another Screening Case</span>
            </button>
          </div>
        </div>
      ) : (
        /* INTAKE FORM (2-COLUMN DESKTOP / STACKED MOBILE) */
        <form onSubmit={handleCreateCaseAndProcess}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1.1fr 0.9fr",
              gap: 20,
              alignItems: "start",
            }}
            className="intake-grid"
          >
            {/* Left Column: Patient Demographics */}
            <div className="medical-card" style={{ padding: 24 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                <User size={18} color="var(--primary-hover)" />
                <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Patient Demographics & Case History
                </h2>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div className="form-group">
                  <label className="form-label">
                    <span>Patient Age</span>
                    <span className="form-label-hint">Years (1-120)</span>
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="120"
                    value={patientAge}
                    onChange={(e) => setPatientAge(e.target.value)}
                    placeholder="e.g. 45"
                    className="form-control"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Biological Sex</label>
                  <select
                    value={patientGender}
                    onChange={(e) => setPatientGender(e.target.value)}
                    className="form-control"
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other / Not Specified</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>Anatomical Lesion Location</span>
                  <span className="form-label-hint">Standard site</span>
                </label>
                <select
                  value={bodyLocation}
                  onChange={(e) => setBodyLocation(e.target.value)}
                  className="form-control"
                >
                  <option value="back">Back / Posterior Trunk</option>
                  <option value="lower extremity">Lower Extremity (Leg / Thigh / Foot)</option>
                  <option value="upper extremity">Upper Extremity (Arm / Hand)</option>
                  <option value="abdomen">Abdomen / Anterior Trunk</option>
                  <option value="chest">Chest</option>
                  <option value="face">Face</option>
                  <option value="neck">Neck</option>
                  <option value="scalp">Scalp</option>
                  <option value="ear">Ear</option>
                  <option value="acral">Acral (Palms / Soles / Subungual)</option>
                  <option value="genital">Genital</option>
                  <option value="other">Other Anatomical Location</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>Known Dermatological Allergies</span>
                  <span className="form-label-hint">Optional</span>
                </label>
                <input
                  type="text"
                  value={knownAllergies}
                  onChange={(e) => setKnownAllergies(e.target.value)}
                  placeholder="e.g. Topical corticosteroids, latex"
                  className="form-control"
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>Medical History / Family History</span>
                  <span className="form-label-hint">Optional</span>
                </label>
                <input
                  type="text"
                  value={medicalHistory}
                  onChange={(e) => setMedicalHistory(e.target.value)}
                  placeholder="e.g. Prior dysplastic nevus excision, excessive UV exposure"
                  className="form-control"
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">
                  <span>Clinical Examination Notes</span>
                  <span className="form-label-hint">Observations</span>
                </label>
                <textarea
                  value={clinicalNotes}
                  onChange={(e) => setClinicalNotes(e.target.value)}
                  placeholder="Document lesion evolution, itching, bleeding, or ABCD clinical observations..."
                  className="form-control"
                  style={{ minHeight: 76 }}
                />
              </div>
            </div>

            {/* Right Column: Lesion Photography Dropzone */}
            <div className="medical-card" style={{ padding: 24, display: "flex", flexDirection: "column", gap: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Camera size={18} color="var(--primary-hover)" />
                <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Skin Lesion Photography
                </h2>
              </div>

              {imageError && (
                <div
                  style={{
                    padding: "8px 12px",
                    background: "var(--error-bg)",
                    border: "1px solid var(--error-border)",
                    borderRadius: "var(--radius-sm)",
                    color: "var(--error-text)",
                    fontSize: "0.8rem",
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}
                >
                  <AlertTriangle size={15} />
                  <span>{imageError}</span>
                </div>
              )}

              {/* Upload Dropzone */}
              {!selectedFile ? (
                <div
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => inputRef.current?.click()}
                  style={{
                    border: dragActive ? "2px dashed var(--primary-hover)" : "2px dashed var(--border-color)",
                    borderRadius: "var(--radius-md)",
                    padding: "36px 20px",
                    textAlign: "center",
                    cursor: "pointer",
                    background: dragActive ? "var(--primary-light)" : "var(--bg-surface-subtle)",
                    transition: "all 0.15s ease",
                  }}
                >
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
                    style={{ display: "none" }}
                  />

                  <div
                    style={{
                      width: 44,
                      height: 44,
                      borderRadius: "var(--radius-md)",
                      background: "var(--primary-light)",
                      color: "var(--text-main)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 12px",
                    }}
                  >
                    <UploadCloud size={22} color="var(--primary-hover)" />
                  </div>

                  <strong style={{ fontSize: "0.92rem", color: "var(--text-main)", display: "block" }}>
                    Click or drag lesion photograph here
                  </strong>
                  <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 4 }}>
                    Supports high-resolution JPG, PNG, or WEBP up to 15MB
                  </p>

                  <div
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: "0.72rem",
                      color: "var(--text-muted)",
                      marginTop: 14,
                    }}
                  >
                    <Sliders size={13} />
                    <span>Real-time blur, illumination & contrast verification</span>
                  </div>
                </div>
              ) : (
                /* Selected File Preview Box */
                <div
                  style={{
                    border: "1px solid var(--border-color)",
                    borderRadius: "var(--radius-md)",
                    padding: 16,
                    background: "var(--bg-surface-subtle)",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <CheckCircle2 size={16} color="var(--success)" />
                      <strong style={{ fontSize: "0.82rem", color: "var(--text-main)" }}>Image Ready for Verification</strong>
                    </div>
                    <button
                      type="button"
                      onClick={handleRemoveImage}
                      className="btn btn-ghost btn-sm"
                      style={{ padding: "3px 8px", fontSize: "0.72rem" }}
                    >
                      <X size={14} />
                      <span>Remove</span>
                    </button>
                  </div>

                  <div
                    style={{
                      height: 200,
                      borderRadius: "var(--radius-sm)",
                      overflow: "hidden",
                      background: "#0F172A",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      marginBottom: 12,
                    }}
                  >
                    <img
                      src={previewUrl}
                      alt="Preview"
                      style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                    />
                  </div>

                  {fileDetails && (
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "0.74rem",
                        color: "var(--text-secondary)",
                        background: "#FFFFFF",
                        padding: "8px 12px",
                        borderRadius: "var(--radius-xs)",
                        border: "1px solid var(--border-color)",
                      }}
                    >
                      <span>{fileDetails.width} × {fileDetails.height} px</span>
                      <span>{fileDetails.type}</span>
                      <span>{fileDetails.sizeMb} MB</span>
                    </div>
                  )}
                </div>
              )}

              {/* Submit / Intake Action */}
              <button
                type="submit"
                disabled={processing || !selectedFile}
                className="btn btn-primary btn-lg"
                style={{ width: "100%", marginTop: "auto" }}
              >
                {processing ? (
                  <>
                    <RefreshCw size={17} className="spin" />
                    <span>
                      {processingStage === "creating_case"
                        ? "Registering Screening Case..."
                        : processingStage === "uploading_image"
                        ? "Running IQA & Standardization..."
                        : "Processing Case Intake..."}
                    </span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={18} />
                    <span>Register Case & Preprocess Image</span>
                  </>
                )}
              </button>

              {createdCase && processingStage === "upload_failed" && (
                <button
                  type="button"
                  onClick={handleRetryUpload}
                  disabled={processing}
                  className="btn btn-secondary"
                  style={{ width: "100%" }}
                >
                  <RotateCcw size={16} />
                  <span>Retry Uploading Photograph</span>
                </button>
              )}
            </div>
          </div>
        </form>
      )}

      <style>{`
        @media (max-width: 860px) {
          .intake-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}
