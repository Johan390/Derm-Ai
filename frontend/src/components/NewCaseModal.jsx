import React, { useState } from "react";
import { X, AlertCircle, FileText, Plus, RefreshCw } from "lucide-react";
import { api } from "../api";

export default function NewCaseModal({ isOpen, onClose, onCaseCreated }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [patientAge, setPatientAge] = useState("");
  const [patientGender, setPatientGender] = useState("male");
  const [bodyLocation, setBodyLocation] = useState("back");
  const [knownAllergies, setKnownAllergies] = useState("");
  const [medicalHistory, setMedicalHistory] = useState("");
  const [clinicalNotes, setClinicalNotes] = useState("");

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const payload = {
        patient_age: patientAge ? parseInt(patientAge, 10) : null,
        patient_gender: patientGender,
        body_location: bodyLocation,
        known_allergies: knownAllergies.trim() || null,
        medical_history: medicalHistory.trim() || null,
        clinical_notes: clinicalNotes.trim() || null,
        image_source: "mobile_upload",
      };

      const newCase = await api.createCase(payload);
      if (onCaseCreated) onCaseCreated(newCase);
      onClose();
    } catch (err) {
      setError(err.message || "Failed to create case");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 480 }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 16,
            paddingBottom: 12,
            borderBottom: "1px solid var(--border-color)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <FileText size={18} color="var(--primary-hover)" />
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-main)" }}>
              Register Screening Case
            </h3>
          </div>
          <button onClick={onClose} className="btn btn-ghost btn-sm" style={{ padding: 4 }}>
            <X size={18} />
          </button>
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
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Patient Age</label>
              <input
                type="number"
                min="1"
                max="120"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                placeholder="Years"
                className="form-control"
              />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Biological Sex</label>
              <select
                value={patientGender}
                onChange={(e) => setPatientGender(e.target.value)}
                className="form-control"
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Anatomical Site</label>
            <select
              value={bodyLocation}
              onChange={(e) => setBodyLocation(e.target.value)}
              className="form-control"
            >
              <option value="back">Back</option>
              <option value="lower extremity">Lower Extremity</option>
              <option value="upper extremity">Upper Extremity</option>
              <option value="abdomen">Abdomen</option>
              <option value="chest">Chest</option>
              <option value="face">Face</option>
              <option value="neck">Neck</option>
              <option value="scalp">Scalp</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Known Allergies</label>
            <input
              type="text"
              value={knownAllergies}
              onChange={(e) => setKnownAllergies(e.target.value)}
              placeholder="e.g. Latex"
              className="form-control"
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Medical History</label>
            <input
              type="text"
              value={medicalHistory}
              onChange={(e) => setMedicalHistory(e.target.value)}
              placeholder="e.g. Previous dysplastic nevi"
              className="form-control"
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Clinical Notes</label>
            <textarea
              value={clinicalNotes}
              onChange={(e) => setClinicalNotes(e.target.value)}
              placeholder="Case history and examination notes..."
              className="form-control"
              style={{ minHeight: 64 }}
            />
          </div>

          <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 10 }}>
            <button type="button" onClick={onClose} className="btn btn-secondary">
              Cancel
            </button>
            <button type="submit" disabled={loading} className="btn btn-primary">
              {loading ? (
                <>
                  <RefreshCw size={15} className="spin" />
                  <span>Registering...</span>
                </>
              ) : (
                <>
                  <Plus size={16} />
                  <span>Register Case</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
