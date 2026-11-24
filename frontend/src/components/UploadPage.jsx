import React, { useState, useRef } from "react";
import {
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Plus,
  X,
  Eye
} from "lucide-react";
import { api } from "../api";

export default function UploadPage({
  cases,
  selectedCaseId,
  onCaseSelected,
  onUploadSuccess,
  onOpenNewCase,
  onViewCaseDetails
}) {
  const [activeCaseId, setActiveCaseId] = useState(
    selectedCaseId || (cases.length > 0 ? cases[0].id : "")
  );
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileDetails, setFileDetails] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [lastProcessedCase, setLastProcessedCase] = useState(null);
  const inputRef = useRef(null);

  const handleFileSelect = (file) => {
    if (!file) return;
    setError("");
    setLastProcessedCase(null);

    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      setError("Unsupported format. Please upload a standard JPG, PNG, or WEBP image.");
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setError("File exceeds 15MB maximum size limit.");
      return;
    }

    setSelectedFile(file);
    const objUrl = URL.createObjectURL(file);
    setPreviewUrl(objUrl);

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

  const handleRemove = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setFileDetails(null);
  };

  const handleUpload = async () => {
    if (!selectedFile || !activeCaseId) return;

    setUploading(true);
    setError("");

    try {
      const updatedCase = await api.uploadCaseImage(activeCaseId, selectedFile);
      setLastProcessedCase(updatedCase);
      if (onUploadSuccess) onUploadSuccess(updatedCase);
      setSelectedFile(null);
      setPreviewUrl(null);
      setFileDetails(null);
    } catch (err) {
      setError(err.message || "Failed to process image");
    } finally {
      setUploading(false);
    }
  };

  const latestImg =
    lastProcessedCase && lastProcessedCase.images && lastProcessedCase.images.length > 0
      ? lastProcessedCase.images[lastProcessedCase.images.length - 1]
      : null;

  return (
    <div style={{ maxWidth: 880, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
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
              Image Processing Service
            </span>
          </div>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-main)" }}>
            Upload Skin Lesion Photograph
          </h1>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
            Attach photography to an existing screening case for quality assessment and standardization
          </p>
        </div>

        <button onClick={onOpenNewCase} className="btn btn-secondary">
          <Plus size={16} />
          <span>New Case</span>
        </button>
      </div>

      {error && (
        <div
          style={{
            padding: "12px 16px",
            background: "var(--error-bg)",
            border: "1px solid var(--error-border)",
            borderRadius: "var(--radius-md)",
            color: "var(--error-text)",
            fontSize: "0.85rem",
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <AlertTriangle size={17} />
          <span>{error}</span>
        </div>
      )}

      {/* Case Selector */}
      <div className="medical-card" style={{ padding: 20 }}>
        <label className="form-label" style={{ marginBottom: 8 }}>
          <span>Target Screening Case</span>
          <span className="form-label-hint">Select destination case</span>
        </label>
        <select
          value={activeCaseId}
          onChange={(e) => {
            setActiveCaseId(e.target.value);
            if (onCaseSelected) onCaseSelected(e.target.value);
          }}
          className="form-control"
        >
          {cases.map((c) => (
            <option key={c.id} value={c.id}>
              {c.case_code} — {c.body_location} ({c.patient_gender}, {c.patient_age ? `${c.patient_age}y` : "Age N/A"})
            </option>
          ))}
        </select>
      </div>

      {/* Upload Zone */}
      <div className="medical-card" style={{ padding: 24 }}>
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
              padding: "40px 20px",
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
            <strong style={{ fontSize: "0.95rem", color: "var(--text-main)", display: "block" }}>
              Click or drag lesion photograph to upload
            </strong>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 4 }}>
              JPG, PNG, or WEBP up to 15MB
            </p>
          </div>
        ) : (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <strong style={{ fontSize: "0.88rem", color: "var(--text-main)" }}>Selected Photograph</strong>
              <button onClick={handleRemove} className="btn btn-ghost btn-sm" style={{ padding: 4 }}>
                <X size={16} />
              </button>
            </div>

            <div
              style={{
                height: 220,
                borderRadius: "var(--radius-sm)",
                overflow: "hidden",
                background: "#0F172A",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                marginBottom: 14,
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
                  fontSize: "0.75rem",
                  color: "var(--text-secondary)",
                  background: "var(--bg-surface-subtle)",
                  padding: "8px 12px",
                  borderRadius: "var(--radius-xs)",
                  border: "1px solid var(--border-color)",
                  marginBottom: 16,
                }}
              >
                <span>{fileDetails.width} × {fileDetails.height} px</span>
                <span>{fileDetails.type}</span>
                <span>{fileDetails.sizeMb} MB</span>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={uploading}
              className="btn btn-primary btn-lg"
              style={{ width: "100%" }}
            >
              {uploading ? (
                <>
                  <RefreshCw size={17} className="spin" />
                  <span>Standardizing Image...</span>
                </>
              ) : (
                <>
                  <UploadCloud size={17} />
                  <span>Upload & Standardize</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Success Result View */}
      {latestImg && (
        <div className="medical-card" style={{ padding: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <CheckCircle2 size={20} color="var(--success)" />
            <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
              Standardization Completed
            </h3>
          </div>
          <button
            onClick={() => onViewCaseDetails && onViewCaseDetails(lastProcessedCase)}
            className="btn btn-primary"
          >
            <span>View Case & Predictions</span>
            <Eye size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
