import React, { useState } from "react";
import {
  X,
  Microscope,
  UserPlus,
  LogIn,
  AlertCircle,
  Eye,
  EyeOff,
  Activity,
  RefreshCw
} from "lucide-react";
import { api } from "../api";

export default function LoginModal({ isOpen, onClose, onAuthSuccess, initialRegister = false }) {
  const [isRegister, setIsRegister] = useState(initialRegister);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  // Form inputs
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("clinician");
  const [institution, setInstitution] = useState("");

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!username.trim() || !password) {
      setError("Please fill in all required credentials.");
      return;
    }

    setLoading(true);
    try {
      if (isRegister) {
        if (!fullName.trim() || !email.trim()) {
          setError("Full name and email address are required for registration.");
          setLoading(false);
          return;
        }

        const res = await api.register({
          username: username.trim(),
          password,
          full_name: fullName.trim(),
          email: email.trim(),
          role,
          institution: institution.trim() || "General Dermatology Practice",
        });
        if (onAuthSuccess) onAuthSuccess(res);
      } else {
        const res = await api.login(username.trim(), password);
        if (onAuthSuccess) onAuthSuccess(res);
      }
      onClose();
    } catch (err) {
      setError(err.message || "Authentication failed. Check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 460 }}
      >
        {/* Modal Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 20,
            paddingBottom: 14,
            borderBottom: "1px solid var(--border-color)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 34,
                height: 34,
                borderRadius: "var(--radius-md)",
                background: "var(--primary-accent)",
                color: "var(--text-main)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Activity size={18} strokeWidth={2.4} />
            </div>
            <div>
              <h2 style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-main)" }}>
                {isRegister ? "Register Clinical Account" : "Sign In to DermAI"}
              </h2>
              <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: 1 }}>
                Clinical Decision Support Platform
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="btn btn-ghost btn-sm"
            style={{ padding: 6, color: "var(--text-muted)" }}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        {/* Tab Toggle (Sign In vs Register) */}
        <div
          style={{
            display: "flex",
            background: "var(--bg-surface-subtle)",
            padding: 3,
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border-color)",
            marginBottom: 20,
          }}
        >
          <button
            type="button"
            onClick={() => {
              setIsRegister(false);
              setError("");
            }}
            style={{
              flex: 1,
              padding: "7px 12px",
              fontSize: "0.82rem",
              fontWeight: !isRegister ? 700 : 500,
              background: !isRegister ? "#FFFFFF" : "transparent",
              color: !isRegister ? "var(--text-main)" : "var(--text-secondary)",
              border: !isRegister ? "1px solid var(--border-color)" : "1px solid transparent",
              borderRadius: "var(--radius-sm)",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => {
              setIsRegister(true);
              setError("");
            }}
            style={{
              flex: 1,
              padding: "7px 12px",
              fontSize: "0.82rem",
              fontWeight: isRegister ? 700 : 500,
              background: isRegister ? "#FFFFFF" : "transparent",
              color: isRegister ? "var(--text-main)" : "var(--text-secondary)",
              border: isRegister ? "1px solid var(--border-color)" : "1px solid transparent",
              borderRadius: "var(--radius-sm)",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            Register Practitioner
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
              marginBottom: 16,
            }}
          >
            <AlertCircle size={16} style={{ flexShrink: 0 }} />
            <span>{error}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {isRegister && (
            <>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Full Name & Credentials</label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="e.g. Dr. Alex Morgan, MD"
                  className="form-control"
                  required
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Professional Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@hospital.org"
                  className="form-control"
                  required
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Role Category</label>
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="form-control"
                >
                  <option value="clinician">Attending Clinician / Dermatologist (MD)</option>
                  <option value="patient">Patient Self-Screening</option>
                </select>
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Institution / Hospital (Optional)</label>
                <input
                  type="text"
                  value={institution}
                  onChange={(e) => setInstitution(e.target.value)}
                  placeholder="e.g. Metro Dermatology Clinic"
                  className="form-control"
                />
              </div>
            </>
          )}

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. dr_smith"
              className="form-control"
              autoComplete="username"
              required
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Password</label>
            <div style={{ position: "relative" }}>
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="form-control"
                style={{ paddingRight: 36 }}
                autoComplete={isRegister ? "new-password" : "current-password"}
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: "absolute",
                  right: 10,
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: "var(--text-muted)",
                  padding: 2,
                }}
                aria-label="Toggle password visibility"
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary btn-lg"
            style={{ width: "100%", marginTop: 6 }}
          >
            {loading ? (
              <>
                <RefreshCw size={16} className="spin" />
                <span>Authenticating...</span>
              </>
            ) : isRegister ? (
              <>
                <UserPlus size={16} />
                <span>Create Practitioner Account</span>
              </>
            ) : (
              <>
                <LogIn size={16} />
                <span>Sign In</span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
