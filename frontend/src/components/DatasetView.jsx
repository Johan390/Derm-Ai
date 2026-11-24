import React, { useState, useEffect } from "react";
import {
  Database,
  Download,
  Activity,
  ShieldCheck,
  Filter,
  ChevronLeft,
  ChevronRight
} from "lucide-react";
import { api } from "../api";

export default function DatasetView({ initialSection = "benchmark" }) {
  const [activeSection, setActiveSection] = useState(
    initialSection === "model-evaluation"
      ? "metrics"
      : initialSection === "anonymized-cases"
      ? "anonymized"
      : "benchmark"
  );
  const [edaSummary, setEdaSummary] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [_qualityStats, setQualityStats] = useState(null);
  const [anonymizedCases, setAnonymizedCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [exportLoading, setExportLoading] = useState(false);
  const [exportSuccess, setExportSuccess] = useState("");
  const [caseFilter, setCaseFilter] = useState("all");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 15;

  useEffect(() => {
    if (initialSection) {
      if (initialSection === "model-evaluation") {
        setActiveSection("metrics");
      } else if (initialSection === "anonymized-cases") {
        setActiveSection("anonymized");
      } else {
        setActiveSection("benchmark");
      }
    }
  }, [initialSection]);

  useEffect(() => {
    let mounted = true;
    async function loadAllResearchData() {
      try {
        const [eda, metrics, quality, anon] = await Promise.all([
          api.getEdaSummary().catch(() => null),
          api.getResearchMetrics().catch(() => null),
          api.getImageQualityStats().catch(() => null),
          api.getAnonymizedCases(100).catch(() => []),
        ]);

        if (mounted) {
          if (eda) setEdaSummary(eda);
          if (metrics) setModelMetrics(metrics);
          if (quality) setQualityStats(quality);
          if (anon) setAnonymizedCases(anon);
          setLoading(false);
        }
      } catch (err) {
        console.warn("Research data load error:", err);
        if (mounted) setLoading(false);
      }
    }
    loadAllResearchData();
    return () => {
      mounted = false;
    };
  }, []);

  const handleExportData = async (format = "json") => {
    try {
      setExportLoading(true);
      setExportSuccess("");
      if (format === "csv") {
        const blob = await api.exportAnonymizedData("csv");
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `dermai_deidentified_research_export_${new Date().toISOString().slice(0, 10)}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } else {
        const data = await api.exportAnonymizedData("json");
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `dermai_deidentified_research_export_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      }
      setExportSuccess(`Cohort successfully exported as ${format.toUpperCase()}`);
      setTimeout(() => setExportSuccess(""), 4000);
    } catch (err) {
      console.warn("Export failed:", err);
    } finally {
      setExportLoading(false);
    }
  };

  const filteredAnonymizedCases = anonymizedCases.filter((row) => {
    if (caseFilter === "suitable") return row.is_suitable_for_ai === true;
    if (caseFilter === "malignant") return row.risk_level === "High Risk";
    return true;
  });

  const totalPages = Math.ceil(filteredAnonymizedCases.length / pageSize) || 1;
  const paginatedCases = filteredAnonymizedCases.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
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
              Scientific Evaluation & Benchmarking
            </span>
          </div>
          <h1 style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)" }}>
            Research & Model Evaluation Hub
          </h1>
          <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", marginTop: 2 }}>
            Empirical validation on the HAM10000 dataset, class-wise performance metrics, and de-identified case analytics
          </p>
        </div>

        {/* Section Sub-Navigation Tabs */}
        <div
          style={{
            display: "flex",
            background: "var(--bg-surface-subtle)",
            padding: 3,
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border-color)",
          }}
        >
          <button
            onClick={() => setActiveSection("benchmark")}
            className={`btn btn-sm ${activeSection === "benchmark" ? "btn-secondary" : "btn-ghost"}`}
          >
            <Database size={15} />
            <span>HAM10000 EDA</span>
          </button>

          <button
            onClick={() => setActiveSection("metrics")}
            className={`btn btn-sm ${activeSection === "metrics" ? "btn-secondary" : "btn-ghost"}`}
          >
            <Activity size={15} />
            <span>Model Evaluation</span>
          </button>

          <button
            onClick={() => setActiveSection("anonymized")}
            className={`btn btn-sm ${activeSection === "anonymized" ? "btn-secondary" : "btn-ghost"}`}
          >
            <ShieldCheck size={15} />
            <span>De-Identified Data</span>
          </button>
        </div>
      </div>

      {loading ? (
        <div className="medical-card" style={{ padding: 48, textAlign: "center", color: "var(--text-muted)" }}>
          <p style={{ fontSize: "0.85rem" }}>Loading research and benchmark datasets...</p>
        </div>
      ) : activeSection === "benchmark" ? (
        /* SECTION 1: HAM10000 BENCHMARK & EDA */
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Key Metrics Strip */}
          <div className="metric-strip">
            <div className="metric-tile" style={{ borderLeft: "4px solid var(--primary-accent)" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
                Total Dataset Size
              </span>
              <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--text-main)", margin: "4px 0 2px" }}>
                {edaSummary ? edaSummary.total_records.toLocaleString() : "10,015"}
              </div>
              <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>Dermatoscopic images</span>
            </div>

            <div className="metric-tile" style={{ borderLeft: "4px solid var(--primary-hover)" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
                Unique Patients
              </span>
              <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--text-main)", margin: "4px 0 2px" }}>
                {edaSummary?.unique_lesions ? edaSummary.unique_lesions.toLocaleString() : "7,470"}
              </div>
              <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>Lesion tracking IDs</span>
            </div>

            <div className="metric-tile" style={{ borderLeft: "4px solid var(--success)" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
                Diagnostic Categories
              </span>
              <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--text-main)", margin: "4px 0 2px" }}>
                7
              </div>
              <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>HAM10000 taxonomy</span>
            </div>

            <div className="metric-tile" style={{ borderLeft: "4px solid var(--info)" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
                Verification Modality
              </span>
              <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-main)", margin: "8px 0 4px" }}>
                Histopathology
              </div>
              <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>Gold-standard biopsy</span>
            </div>
          </div>

          {/* Class Distribution Table & Chart */}
          <div className="medical-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 4 }}>
              Diagnostic Neoplasm Class Distribution
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 16 }}>
              Representation across the 10,015 images in the benchmark training and validation split
            </p>

            <div className="table-container">
              <table className="medical-table">
                <thead>
                  <tr>
                    <th>Class Code</th>
                    <th>Full Neoplasm Name</th>
                    <th>Clinical Risk Tier</th>
                    <th style={{ textAlign: "right" }}>Sample Count</th>
                    <th style={{ textAlign: "right" }}>Dataset Share</th>
                    <th style={{ width: 180 }}>Representation Bar</th>
                  </tr>
                </thead>
                <tbody>
                  {(edaSummary?.class_summary || (edaSummary?.class_distribution ? Object.entries(edaSummary.class_distribution).map(([code, count]) => ({ code, count, percentage: ((count / edaSummary.total_records) * 100) })) : null)) ? (
                    (edaSummary?.class_summary || Object.entries(edaSummary.class_distribution).map(([code, count]) => ({ code, count, percentage: ((count / edaSummary.total_records) * 100) }))).map((item) => {
                      const code = item.code;
                      const count = item.count;
                      const share = item.percentage != null ? Number(item.percentage).toFixed(1) : ((count / (edaSummary.total_records || 10015)) * 100).toFixed(1);
                      const isHigh = item.risk_level === "High Risk" || code === "mel" || code === "bcc";
                      const isMod = item.risk_level === "Moderate Risk" || code === "akiec";

                      return (
                        <tr key={code}>
                          <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                            {code.toUpperCase()}
                          </td>
                          <td style={{ fontWeight: 600 }}>
                            {item.name || (code === "nv" ? "Melanocytic Nevus" :
                             code === "mel" ? "Malignant Melanoma" :
                             code === "bkl" ? "Benign Keratosis" :
                             code === "bcc" ? "Basal Cell Carcinoma" :
                             code === "akiec" ? "Actinic Keratosis" :
                             code === "vasc" ? "Vascular Lesion" : "Dermatofibroma")}
                          </td>
                          <td>
                            <span className={isHigh ? "badge badge-high" : isMod ? "badge badge-moderate" : "badge badge-low"}>
                              {item.risk_level || (isHigh ? "High Risk" : isMod ? "Moderate Risk" : "Low Risk")}
                            </span>
                          </td>
                          <td style={{ textAlign: "right", fontFamily: "var(--font-mono)" }}>
                            {count != null ? count.toLocaleString() : "-"}
                          </td>
                          <td style={{ textAlign: "right", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                            {share}%
                          </td>
                          <td>
                            <div style={{ height: 6, background: "var(--bg-surface-subtle)", borderRadius: 3, overflow: "hidden", border: "1px solid var(--border-color)" }}>
                              <div style={{ height: "100%", width: `${share}%`, background: isHigh ? "var(--error)" : isMod ? "var(--warning)" : "var(--primary-hover)" }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={6} style={{ textAlign: "center", padding: 24, color: "var(--text-muted)" }}>
                        Loading class distribution data...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : activeSection === "metrics" ? (
        /* SECTION 2: MODEL EVALUATION & BENCHMARKS */
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Performance Summary Cards */}
          <div className="metric-strip">
            <div className="metric-tile" style={{ borderLeft: "4px solid var(--primary-accent)" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
                Overall Accuracy
              </span>
              <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--text-main)", margin: "4px 0 2px" }}>
                {modelMetrics ? `${(modelMetrics.overall_accuracy * 100).toFixed(2)}%` : "70.22%"}
              </div>
              <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>1,501 test images</span>
            </div>

            <div className="metric-tile" style={{ borderLeft: "4px solid var(--primary-hover)" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
                Macro ROC-AUC
              </span>
              <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--text-main)", margin: "4px 0 2px" }}>
                {modelMetrics ? modelMetrics.macro_roc_auc?.toFixed(4) : "0.9053"}
              </div>
              <span style={{ fontSize: "0.74rem", color: "var(--success-text)" }}>High multi-class discrimination</span>
            </div>

            <div className="metric-tile" style={{ borderLeft: "4px solid var(--success)" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
                Balanced Accuracy
              </span>
              <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--text-main)", margin: "4px 0 2px" }}>
                {modelMetrics ? `${(modelMetrics.balanced_accuracy * 100).toFixed(2)}%` : "60.85%"}
              </div>
              <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>Unweighted mean recall</span>
            </div>

            <div className="metric-tile" style={{ borderLeft: "4px solid var(--info)" }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
                Weighted F1-Score
              </span>
              <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--text-main)", margin: "4px 0 2px" }}>
                {modelMetrics ? modelMetrics.weighted_f1?.toFixed(4) : "0.7213"}
              </div>
              <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>Support-weighted harmonic mean</span>
            </div>
          </div>

          {/* Model Class-Wise Performance Table */}
          <div className="medical-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 4 }}>
              Class-Wise Test Set Performance (MobileNetV2 Transfer Learning)
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 16 }}>
              Precision, Sensitivity (Recall), F1-Score, and ROC-AUC evaluated on 1,501 held-out test images
            </p>

            <div className="table-container">
              <table className="medical-table">
                <thead>
                  <tr>
                    <th>Class Code</th>
                    <th>Diagnostic Entity</th>
                    <th>Risk Tier</th>
                    <th style={{ textAlign: "right" }}>Support</th>
                    <th style={{ textAlign: "right" }}>Precision</th>
                    <th style={{ textAlign: "right" }}>Recall (Sensitivity)</th>
                    <th style={{ textAlign: "right" }}>F1-Score</th>
                    <th style={{ textAlign: "right" }}>ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {modelMetrics?.per_class &&
                    Object.entries(modelMetrics.per_class).map(([code, metrics]) => (
                      <tr key={code}>
                        <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                          {code.toUpperCase()}
                        </td>
                        <td style={{ fontWeight: 600 }}>{metrics.name}</td>
                        <td>
                          <span
                            className={
                              metrics.risk_level === "High Risk"
                                ? "badge badge-high"
                                : metrics.risk_level === "Moderate Risk"
                                ? "badge badge-moderate"
                                : "badge badge-low"
                            }
                          >
                            {metrics.risk_level}
                          </span>
                        </td>
                        <td style={{ textAlign: "right", fontFamily: "var(--font-mono)" }}>
                          {metrics.support}
                        </td>
                        <td style={{ textAlign: "right", fontFamily: "var(--font-mono)" }}>
                          {(metrics.precision * 100).toFixed(1)}%
                        </td>
                        <td
                          style={{
                            textAlign: "right",
                            fontFamily: "var(--font-mono)",
                            fontWeight: 700,
                            color: metrics.recall > 0.6 ? "var(--success-text)" : "var(--error-text)",
                          }}
                        >
                          {(metrics.recall * 100).toFixed(1)}%
                        </td>
                        <td style={{ textAlign: "right", fontFamily: "var(--font-mono)" }}>
                          {metrics.f1_score.toFixed(3)}
                        </td>
                        <td style={{ textAlign: "right", fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--text-main)" }}>
                          {metrics.roc_auc.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        /* SECTION 3: DE-IDENTIFIED CASES COHORT */
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="toolbar-row">
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
              <Filter size={15} color="var(--text-secondary)" />
              <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", fontWeight: 600 }}>Cohort Filter:</span>
              <button
                onClick={() => {
                  setCaseFilter("all");
                  setCurrentPage(1);
                }}
                className={`btn btn-sm ${caseFilter === "all" ? "btn-secondary" : "btn-ghost"}`}
              >
                All ({anonymizedCases.length})
              </button>
              <button
                onClick={() => {
                  setCaseFilter("suitable");
                  setCurrentPage(1);
                }}
                className={`btn btn-sm ${caseFilter === "suitable" ? "btn-secondary" : "btn-ghost"}`}
              >
                Suitable Only
              </button>
              <button
                onClick={() => {
                  setCaseFilter("malignant");
                  setCurrentPage(1);
                }}
                className={`btn btn-sm ${caseFilter === "malignant" ? "btn-secondary" : "btn-ghost"}`}
              >
                High Risk
              </button>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {exportSuccess && (
                <span style={{ fontSize: "0.75rem", color: "var(--success-text)", fontWeight: 600 }}>
                  {exportSuccess}
                </span>
              )}
              <button
                onClick={() => handleExportData("csv")}
                disabled={exportLoading}
                className="btn btn-secondary btn-sm"
              >
                <Download size={14} />
                <span>Export CSV</span>
              </button>
              <button
                onClick={() => handleExportData("json")}
                disabled={exportLoading}
                className="btn btn-secondary btn-sm"
              >
                <Download size={14} />
                <span>Export JSON</span>
              </button>
            </div>
          </div>

          <div className="table-container">
            <table className="medical-table">
              <thead>
                <tr>
                  <th>Research ID</th>
                  <th>Age Cohort</th>
                  <th>Biological Sex</th>
                  <th>Anatomical Location</th>
                  <th>IQA Status</th>
                  <th>Predicted Class</th>
                  <th>Risk Tier</th>
                  <th style={{ textAlign: "right" }}>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {paginatedCases.length === 0 ? (
                  <tr>
                    <td colSpan={8} style={{ textAlign: "center", padding: 32, color: "var(--text-muted)" }}>
                      No de-identified records match current filter.
                    </td>
                  </tr>
                ) : (
                  paginatedCases.map((row, idx) => (
                    <tr key={idx}>
                      <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--text-main)" }}>
                        {row.research_id}
                      </td>
                      <td>{row.patient_age_group}</td>
                      <td style={{ textTransform: "capitalize" }}>{row.patient_gender}</td>
                      <td style={{ textTransform: "capitalize" }}>{row.body_location}</td>
                      <td>
                        <span className={`badge ${row.is_suitable_for_ai ? "badge-low" : "badge-high"}`}>
                          {row.image_quality_status || (row.is_suitable_for_ai ? "Suitable" : "Suboptimal")}
                        </span>
                      </td>
                      <td style={{ fontWeight: 600 }}>
                        <span style={{ fontFamily: "var(--font-mono)", marginRight: 6 }}>
                          {row.predicted_class?.toUpperCase()}
                        </span>
                        {row.predicted_disease_name}
                      </td>
                      <td>
                        <span
                          className={`badge ${
                            row.risk_level === "High Risk"
                              ? "badge-high"
                              : row.risk_level === "Moderate Risk"
                              ? "badge-moderate"
                              : "badge-low"
                          }`}
                        >
                          {row.risk_level}
                        </span>
                      </td>
                      <td style={{ textAlign: "right", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                        {row.confidence_score ? `${(row.confidence_score * 100).toFixed(1)}%` : "--"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 14px", background: "#FFFFFF", borderRadius: "var(--radius-md)", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
                Showing {(currentPage - 1) * pageSize + 1} to {Math.min(currentPage * pageSize, filteredAnonymizedCases.length)} of {filteredAnonymizedCases.length} records
              </span>

              <div style={{ display: "flex", gap: 6 }}>
                <button
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: "4px 8px" }}
                >
                  <ChevronLeft size={14} />
                  <span>Prev</span>
                </button>
                <span style={{ fontSize: "0.8rem", padding: "4px 8px", fontWeight: 600, color: "var(--text-main)" }}>
                  {currentPage} / {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: "4px 8px" }}
                >
                  <span>Next</span>
                  <ChevronRight size={14} />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
