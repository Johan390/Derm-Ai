import React, { useState, useEffect } from "react";
import { ShieldCheck, UserCheck, UserX, Clock, AlertCircle, RefreshCw, Stethoscope, Building2, Mail } from "lucide-react";
import { api } from "../api";

export default function AdminCliniciansView() {
  const [pendingClinicians, setPendingClinicians] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const loadPending = async () => {
    try {
      setLoading(true);
      setError("");
      const data = await api.getPendingClinicians();
      setPendingClinicians(data || []);
    } catch (err) {
      setError(err.message || "Failed to load pending clinician registrations");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let ignore = false;
    api.getPendingClinicians()
      .then((data) => {
        if (!ignore) {
          setPendingClinicians(data || []);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!ignore) {
          setError(err.message || "Failed to load pending clinician registrations");
          setLoading(false);
        }
      });
    return () => {
      ignore = true;
    };
  }, []);

  const handleVerify = async (id, name) => {
    try {
      setActionLoading(id);
      setError("");
      await api.verifyClinician(id);
      setMessage(`Successfully verified Dr. ${name}. Clinical triage access enabled.`);
      await loadPending();
      setTimeout(() => setMessage(""), 5000);
    } catch (err) {
      setError(err.message || "Failed to verify clinician");
    } finally {
      setActionLoading(null);
    }
  };

  const handleReject = async (id, name) => {
    if (!window.confirm(`Reject clinical registration for ${name}?`)) {
      return;
    }
    try {
      setActionLoading(id);
      setError("");
      await api.rejectClinician(id);
      setMessage(`Clinician registration for ${name} has been rejected.`);
      await loadPending();
      setTimeout(() => setMessage(""), 5000);
    } catch (err) {
      setError(err.message || "Failed to reject clinician");
    } finally {
      setActionLoading(null);
    }
  };

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
              Administrative Governance
            </span>
          </div>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 800, color: "var(--text-main)" }}>
            Clinician Verification & Oversight
          </h1>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
            Verify medical practitioner credentials before enabling clinical decision support and case triage access
          </p>
        </div>

        <button
          onClick={loadPending}
          disabled={loading}
          className="btn btn-secondary btn-sm"
        >
          <RefreshCw size={14} className={loading ? "spin" : ""} />
          <span>Refresh Queue</span>
        </button>
      </div>

      {message && (
        <div
          style={{
            padding: "12px 16px",
            background: "var(--success-bg)",
            border: "1px solid var(--success-border)",
            borderRadius: "var(--radius-md)",
            color: "var(--success-text)",
            fontSize: "0.85rem",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <ShieldCheck size={16} />
          <span>{message}</span>
        </div>
      )}

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
            gap: 8,
          }}
        >
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Pending Clinicians List */}
      <div className="medical-card" style={{ padding: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
            Pending Practitioner Applications ({pendingClinicians.length})
          </h2>
        </div>

        {loading ? (
          <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text-muted)", fontSize: "0.85rem" }}>
            <RefreshCw size={22} className="spin" style={{ margin: "0 auto 8px" }} />
            <p>Loading pending verification requests...</p>
          </div>
        ) : pendingClinicians.length === 0 ? (
          <div
            style={{
              padding: "44px 20px",
              textAlign: "center",
              background: "var(--bg-surface-subtle)",
              borderRadius: "var(--radius-md)",
              border: "1px dashed var(--border-color)",
            }}
          >
            <ShieldCheck size={32} color="var(--primary-hover)" style={{ margin: "0 auto 10px" }} />
            <h4 style={{ fontSize: "1.0rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 4 }}>
              No Pending Verification Requests
            </h4>
            <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", maxWidth: 420, margin: "0 auto" }}>
              All registered clinicians are verified. Newly registered practitioner accounts will appear here for verification.
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {pendingClinicians.map((clinician) => (
              <div
                key={clinician.id}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  flexWrap: "wrap",
                  gap: 14,
                  padding: "14px 18px",
                  background: "var(--bg-surface-subtle)",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-color)",
                }}
              >
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <Stethoscope size={16} color="var(--primary-hover)" />
                    <strong style={{ fontSize: "0.95rem", color: "var(--text-main)" }}>
                      {clinician.full_name}
                    </strong>
                    <span className="badge badge-awaiting" style={{ fontSize: "0.7rem" }}>
                      Pending Approval
                    </span>
                  </div>

                  <div style={{ display: "flex", gap: 16, fontSize: "0.78rem", color: "var(--text-secondary)", flexWrap: "wrap" }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <Mail size={13} /> {clinician.email}
                    </span>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <Building2 size={13} /> {clinician.institution || "General Practice"}
                    </span>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <Clock size={13} /> Registered: {new Date(clinician.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <button
                    onClick={() => handleVerify(clinician.id, clinician.full_name)}
                    disabled={actionLoading === clinician.id}
                    className="btn btn-primary btn-sm"
                  >
                    <UserCheck size={14} />
                    <span>{actionLoading === clinician.id ? "Approving..." : "Approve & Verify"}</span>
                  </button>

                  <button
                    onClick={() => handleReject(clinician.id, clinician.full_name)}
                    disabled={actionLoading === clinician.id}
                    className="btn btn-secondary btn-sm"
                    style={{ color: "var(--error-text)" }}
                  >
                    <UserX size={14} />
                    <span>Reject</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
