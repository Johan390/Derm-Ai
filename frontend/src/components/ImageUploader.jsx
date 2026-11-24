import React, { useState, useRef } from "react";
import { UploadCloud, CheckCircle2, AlertTriangle, RefreshCw, X } from "lucide-react";
import { api } from "../api";

export default function ImageUploader({ caseId, onUploadSuccess }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileDetails, setFileDetails] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    setError("");

    const validExtensions = ["image/jpeg", "image/png", "image/webp"];
    if (!validExtensions.includes(file.type)) {
      setError("Unsupported format. Please upload JPG, PNG, or WEBP.");
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setError("File exceeds 15MB limit.");
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    const img = new Image();
    img.src = objectUrl;
    img.onload = () => {
      setFileDetails({
        name: file.name,
        sizeMb: (file.size / (1024 * 1024)).toFixed(2),
        width: img.width,
        height: img.height,
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
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleUploadAndProcess = async () => {
    if (!selectedFile || !caseId) return;
    setUploading(true);
    setError("");

    try {
      const updatedCase = await api.uploadCaseImage(caseId, selectedFile);
      if (onUploadSuccess) onUploadSuccess(updatedCase);
      setSelectedFile(null);
      setPreviewUrl(null);
      setFileDetails(null);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="medical-card" style={{ padding: 20 }}>
      <div style={{ marginBottom: 14 }}>
        <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
          Upload Lesion Photography
        </h3>
        <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 2 }}>
          Attach a clear photograph for automated quality assessment and 224×224 tensor standardization.
        </p>
      </div>

      {error && (
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
            marginBottom: 14,
          }}
        >
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

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
            padding: "32px 16px",
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
            onChange={(e) => e.target.files && handleFile(e.target.files[0])}
            style={{ display: "none" }}
          />

          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: "var(--radius-md)",
              background: "var(--primary-light)",
              color: "var(--text-main)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 10px",
            }}
          >
            <UploadCloud size={20} color="var(--primary-hover)" />
          </div>

          <strong style={{ fontSize: "0.88rem", color: "var(--text-main)", display: "block" }}>
            Click or drag lesion photograph to upload
          </strong>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 4, display: "block" }}>
            JPG, PNG, or WEBP up to 15MB
          </span>
        </div>
      ) : (
        <div
          style={{
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-md)",
            padding: 16,
            background: "var(--bg-surface-subtle)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--text-main)" }}>
              Selected Photograph
            </span>
            <button
              onClick={() => {
                setSelectedFile(null);
                setPreviewUrl(null);
                setFileDetails(null);
              }}
              className="btn btn-ghost btn-sm"
              style={{ padding: 4 }}
            >
              <X size={15} />
            </button>
          </div>

          <div
            style={{
              height: 180,
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
                padding: "6px 10px",
                borderRadius: "var(--radius-xs)",
                border: "1px solid var(--border-color)",
                marginBottom: 12,
              }}
            >
              <span>{fileDetails.width} × {fileDetails.height} px</span>
              <span>{fileDetails.sizeMb} MB</span>
            </div>
          )}

          <button
            onClick={handleUploadAndProcess}
            disabled={uploading}
            className="btn btn-primary"
            style={{ width: "100%" }}
          >
            {uploading ? (
              <>
                <RefreshCw size={15} className="spin" />
                <span>Standardizing Lesion Tensor...</span>
              </>
            ) : (
              <>
                <CheckCircle2 size={16} />
                <span>Upload & Standardize</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
