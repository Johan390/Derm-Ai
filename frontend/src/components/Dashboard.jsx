import React, { useState, useEffect } from "react";
import {
  FolderOpen,
  Plus,
  Database,
  Image as ImageIcon,
  ChevronRight,
  Clock,
  Sparkles,
  MessageSquare,
  Activity
} from "lucide-react";
import { api } from "../api";

export default function Dashboard({
  user,
  cases,
  loading,
  onOpenNewCase,
  onNavigateTab,
  onSelectCase,
}) {
  const [edaData, setEdaData] = useState(null);

  useEffect(() => {
    let mounted = true;
    async function fetchEda() {
      try {
        const data = await api.getEdaSummary();
        if (mounted && data && data.total_records) {
          setEdaData(data);
        }
      } catch (err) {
        console.warn("EDA summary notice:", err);
      }
    }
    fetchEda();
    return () => {
      mounted = false;
    };
  }, []);

  // Compute metrics from genuine cases
  const totalCases = cases.length;
  const analyzedCases = cases.filter((c) => c.status === "completed" || (c.predictions && c.predictions.length > 0)).length;
  const pendingReviewCases = cases.filter((c) => !c.review_status || c.review_status === "pending_review").length;
  const followUpCases = cases.filter((c) => c.review_status === "follow_up_required").length;

  // Recent 5 cases
  const recentCases = [...cases]
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 5);

  // Recent images
  const recentImages = [];
  cases.forEach((c) => {
    if (c.images && c.images.length > 0) {
      c.images.forEach((img) => {
        recentImages.push({
          ...img,
          caseCode: c.case_code,
          bodyLocation: c.body_location,
          caseObj: c,
        });
      });
    }
  });
  recentImages.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  const latestFourImages = recentImages.slice(0, 4);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Executive Header */}
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
            <span className="badge badge-teal">
              Clinical Command Center
            </span>
            {user && (
              <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                Dr. {user.full_name || user.username}
              </span>
            )}
          </div>
          <h1 style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)" }}>
            Dermatological Screening Dashboard
          </h1>
          <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", marginTop: 2 }}>
            Real-time overview of active clinical cases, deep learning inferences, and review triage
          </p>
        </div>

        {/* Consolidated Actions */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button
            onClick={() => onNavigateTab("cases")}
            className="btn btn-secondary"
          >
            <FolderOpen size={16} />
            <span>Case Review Queue</span>
          </button>

          <button
            onClick={onOpenNewCase}
            className="btn btn-primary"
          >
            <Plus size={16} />
            <span>Intake New Case</span>
          </button>
        </div>
      </div>

      {/* Triage Metric Tiles */}
      <div className="metric-strip">
        <div
          onClick={() => onNavigateTab("cases")}
          className="metric-tile metric-tile-interactive"
          style={{ borderLeft: "4px solid var(--primary-accent)" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
              Total Cases
            </span>
            <FolderOpen size={16} color="var(--primary-hover)" />
          </div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--text-main)", margin: "6px 0 2px" }}>
            {totalCases}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Registered screening cases
          </div>
        </div>

        <div
          onClick={() => onNavigateTab("cases")}
          className="metric-tile metric-tile-interactive"
          style={{ borderLeft: "4px solid var(--primary-hover)" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
              AI Inferences
            </span>
            <Sparkles size={16} color="var(--primary-hover)" />
          </div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--text-main)", margin: "6px 0 2px" }}>
            {analyzedCases}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            MobileNetV2 & ViT predictions
          </div>
        </div>

        <div
          onClick={() => onNavigateTab("cases")}
          className="metric-tile metric-tile-interactive"
          style={{ borderLeft: "4px solid var(--warning)" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
              Pending Review
            </span>
            <Clock size={16} color="var(--warning)" />
          </div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--warning)", margin: "6px 0 2px" }}>
            {pendingReviewCases}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Awaiting physician evaluation
          </div>
        </div>

        <div
          onClick={() => onNavigateTab("cases")}
          className="metric-tile metric-tile-interactive"
          style={{ borderLeft: "4px solid var(--error)" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
              Follow-Up Required
            </span>
            <MessageSquare size={16} color="var(--error)" />
          </div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--error-text)", margin: "6px 0 2px" }}>
            {followUpCases}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Priority clinical attention
          </div>
        </div>
      </div>

      {/* Main Grid: Recent Cases & Recent Imagery */}
      <div className="dashboard-columns">
        {/* Left: Recent Cases Worklist */}
        <div className="medical-card" style={{ padding: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Activity size={18} color="var(--primary-hover)" />
              <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
                Recent Screening Cases
              </h2>
            </div>
            <button
              onClick={() => onNavigateTab("cases")}
              className="btn btn-ghost btn-sm"
              style={{ fontSize: "0.78rem" }}
            >
              <span>View All</span>
              <ChevronRight size={14} />
            </button>
          </div>

          {loading ? (
            <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)", fontSize: "0.85rem" }}>
              Loading clinical cases...
            </div>
          ) : recentCases.length === 0 ? (
            <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)" }}>
              <FolderOpen size={32} style={{ margin: "0 auto 8px", opacity: 0.5 }} />
              <p style={{ fontSize: "0.85rem" }}>No screening cases found.</p>
              <button onClick={onOpenNewCase} className="btn btn-primary btn-sm" style={{ marginTop: 12 }}>
                <Plus size={14} />
                <span>Register First Case</span>
              </button>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {recentCases.map((c) => {
                const hasPredictions = c.predictions && c.predictions.length > 0;
                const latestPred = hasPredictions ? c.predictions[c.predictions.length - 1] : null;

                return (
                  <div
                    key={c.id}
                    onClick={() => onSelectCase(c)}
                    className="medical-card medical-card-interactive"
                    style={{
                      padding: "12px 16px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 12,
                    }}
                  >
                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.88rem", fontWeight: 700, color: "var(--text-main)" }}>
                          {c.case_code}
                        </span>
                        {latestPred && (
                          <span
                            className={
                              latestPred.risk_level === "High Risk"
                                ? "badge badge-high"
                                : latestPred.risk_level === "Moderate Risk"
                                ? "badge badge-moderate"
                                : "badge badge-low"
                            }
                            style={{ fontSize: "0.68rem" }}
                          >
                            {latestPred.predicted_disease_name || latestPred.predicted_class} ({(latestPred.confidence_score * 100).toFixed(0)}%)
                          </span>
                        )}
                        {c.review_status === "reviewed" ? (
                          <span className="badge badge-low" style={{ fontSize: "0.68rem" }}>Reviewed</span>
                        ) : c.review_status === "follow_up_required" ? (
                          <span className="badge badge-high" style={{ fontSize: "0.68rem" }}>Follow-up</span>
                        ) : (
                          <span className="badge badge-awaiting" style={{ fontSize: "0.68rem" }}>Pending</span>
                        )}
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "flex", gap: 12 }}>
                        <span>Loc: <strong style={{ textTransform: "capitalize" }}>{c.body_location}</strong></span>
                        <span>•</span>
                        <span>{c.patient_age ? `${c.patient_age}y` : "Age N/A"} • {c.patient_gender || "Unknown"}</span>
                      </div>
                    </div>

                    <ChevronRight size={16} color="var(--text-muted)" />
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: Recent Lesion Photography & Benchmark Overview */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Recent Images Strip */}
          <div className="medical-card" style={{ padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <ImageIcon size={17} color="var(--primary-hover)" />
                <h3 style={{ fontSize: "0.98rem", fontWeight: 700, color: "var(--text-main)" }}>
                  Recent Lesion Uploads
                </h3>
              </div>
              <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                {recentImages.length} total
              </span>
            </div>

            {latestFourImages.length === 0 ? (
              <div style={{ padding: "20px 0", textAlign: "center", color: "var(--text-muted)", fontSize: "0.82rem" }}>
                No lesion photographs uploaded yet.
              </div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                {latestFourImages.map((img, idx) => (
                  <div
                    key={idx}
                    onClick={() => onSelectCase(img.caseObj)}
                    style={{
                      aspectRatio: "1/1",
                      borderRadius: "var(--radius-sm)",
                      overflow: "hidden",
                      border: "1px solid var(--border-color)",
                      cursor: "pointer",
                      background: "var(--bg-surface-subtle)",
                      position: "relative",
                    }}
                    title={`Case: ${img.caseCode}`}
                  >
                    <img
                      src={api.getImageUrl(img.preview_url || img.original_url)}
                      alt="Lesion"
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* HAM10000 Research Highlights */}
          <div className="medical-card" style={{ padding: 20, background: "var(--bg-surface-subtle)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
              <Database size={16} color="var(--primary-hover)" />
              <h3 style={{ fontSize: "0.92rem", fontWeight: 700, color: "var(--text-main)" }}>
                HAM10000 Dataset Benchmark
              </h3>
            </div>
            <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.45, marginBottom: 12 }}>
              Trained on {edaData?.total_records ? edaData.total_records.toLocaleString() : "10,015"} dermatoscopic images across 7 neoplasm categories with multi-center ViDIR verification.
            </p>
            <button
              onClick={() => onNavigateTab("research-dashboard")}
              className="btn btn-secondary btn-sm"
              style={{ width: "100%", justifyContent: "center" }}
            >
              <span>Explore Benchmark & Metrics</span>
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
