import React, { useState } from "react";
import {
  History,
  Calendar,
  ChevronRight
} from "lucide-react";

export default function PredictionHistory({
  predictions = [],
  selectedPredictionId,
  onSelectPrediction,
}) {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  if (!predictions || predictions.length === 0) {
    return (
      <div
        className="medical-card"
        style={{
          padding: 20,
          textAlign: "center",
          color: "var(--text-secondary)",
        }}
      >
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: "50%",
            background: "var(--primary-light)",
            color: "var(--text-main)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 8px",
          }}
        >
          <History size={18} />
        </div>
        <h4 style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-main)" }}>
          No Prior Inferences
        </h4>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: 2 }}>
          Past deep learning evaluations will appear here chronologically for longitudinal comparison.
        </p>
      </div>
    );
  }

  // Filter predictions by date
  const filtered = predictions.filter((p) => {
    const timestamp = new Date(p.prediction_timestamp || p.created_at || 0).getTime();
    if (startDate) {
      const s = new Date(startDate).getTime();
      if (timestamp < s) return false;
    }
    if (endDate) {
      const e = new Date(endDate + "T23:59:59").getTime();
      if (timestamp > e) return false;
    }
    return true;
  });

  // Sort descending by created_at (newest first)
  const sorted = [...filtered].sort((a, b) => {
    const da = new Date(a.prediction_timestamp || a.created_at || 0).getTime();
    const db = new Date(b.prediction_timestamp || b.created_at || 0).getTime();
    return db - da;
  });

  return (
    <div className="medical-card" style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, flexWrap: "wrap", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <History size={18} color="var(--primary-hover)" />
          <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
            Inference Timeline ({predictions.length})
          </h3>
        </div>

        {/* Date Filters */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>From:</span>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="form-control"
              style={{ padding: "4px 8px", fontSize: "0.75rem", height: 28 }}
            />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>To:</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="form-control"
              style={{ padding: "4px 8px", fontSize: "0.75rem", height: 28 }}
            />
          </div>
          {(startDate || endDate) && (
            <button
              onClick={() => {
                setStartDate("");
                setEndDate("");
              }}
              className="btn btn-ghost btn-sm"
              style={{ fontSize: "0.72rem", padding: "2px 6px", height: 28 }}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {sorted.map((pred) => {
          const isSelected = selectedPredictionId === pred.id;
          const conf = (pred.confidence_score * 100).toFixed(1);
          const dt = new Date(pred.prediction_timestamp || pred.created_at || 0);

          return (
            <div
              key={pred.id}
              onClick={() => onSelectPrediction && onSelectPrediction(pred)}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "10px 14px",
                borderRadius: "var(--radius-sm)",
                background: isSelected ? "var(--primary-light)" : "var(--bg-surface-subtle)",
                border: isSelected ? "1px solid var(--primary-light-border)" : "1px solid var(--border-color)",
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span
                  className={
                    pred.risk_level === "High Risk"
                      ? "badge badge-high"
                      : pred.risk_level === "Moderate Risk"
                      ? "badge badge-moderate"
                      : "badge badge-low"
                  }
                  style={{ fontSize: "0.68rem" }}
                >
                  {pred.predicted_disease_name || pred.predicted_class}
                </span>

                <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                  Model: <strong>{pred.model_name || "Baseline"}</strong>
                </span>

                <span style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--text-main)" }}>
                  {conf}%
                </span>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <Calendar size={12} />
                  {dt.toLocaleDateString()} {dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
                <ChevronRight size={14} color={isSelected ? "var(--text-main)" : "var(--text-muted)"} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
