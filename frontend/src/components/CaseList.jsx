import React, { useState } from "react";
import {
  Search,
  Calendar,
  MapPin,
  ChevronRight,
  Image as ImageIcon,
  Plus,
  FolderOpen,
  CheckCircle2,
  Clock,
  Sparkles,
  MessageSquare,
  UserCheck,
  Download,
  LayoutGrid,
  ListFilter,
  X
} from "lucide-react";
import { api } from "../api";

export default function CaseList({
  cases,
  loading,
  onSelectCase,
  onOpenNewCaseWorkflow,
  user
}) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [reviewFilter, setReviewFilter] = useState("all");
  const [assignmentFilter, setAssignmentFilter] = useState("all");
  const [viewMode, setViewMode] = useState("table"); // 'table' | 'cards'
  const [isExportingCsv, setIsExportingCsv] = useState(false);
  const [isExportingJson, setIsExportingJson] = useState(false);
  const [exportFeedback, setExportFeedback] = useState("");

  const handleExportCohortCsv = async () => {
    setIsExportingCsv(true);
    setExportFeedback("");
    try {
      await api.exportAllCasesCsv();
      setExportFeedback("Cohort CSV downloaded successfully.");
      setTimeout(() => setExportFeedback(""), 3000);
    } catch (e) {
      setExportFeedback(e.message || "Failed to export CSV");
      setTimeout(() => setExportFeedback(""), 4000);
    } finally {
      setIsExportingCsv(false);
    }
  };

  const handleExportCohortJson = async () => {
    setIsExportingJson(true);
    setExportFeedback("");
    try {
      await api.exportAllCasesJson();
      setExportFeedback("Cohort JSON downloaded successfully.");
      setTimeout(() => setExportFeedback(""), 3000);
    } catch (e) {
      setExportFeedback(e.message || "Failed to export JSON");
      setTimeout(() => setExportFeedback(""), 4000);
    } finally {
      setIsExportingJson(false);
    }
  };

  const isClinician = user && (user.role === "clinician" || user.role === "admin");
  const currentUserId = user?.id || user?.user_id;

  // Filter cases
  const filteredCases = cases.filter((c) => {
    const query = search.toLowerCase();
    const matchesSearch =
      c.case_code.toLowerCase().includes(query) ||
      (c.body_location && c.body_location.toLowerCase().includes(query)) ||
      (c.patient_gender && c.patient_gender.toLowerCase().includes(query)) ||
      (c.clinical_notes && c.clinical_notes.toLowerCase().includes(query));

    const matchesStatus =
      statusFilter === "all" ||
      (statusFilter === "completed" && c.status === "completed") ||
      (statusFilter === "preprocessed" && c.status === "preprocessed") ||
      (statusFilter === "created" && c.status === "created");

    const matchesReview =
      reviewFilter === "all" ||
      c.review_status === reviewFilter;

    const matchesAssignment =
      assignmentFilter === "all" ||
      (assignmentFilter === "assigned_to_me" && (c.assigned_doctor_id === currentUserId || (currentUserId && c.assigned_doctor_id === Number(currentUserId)))) ||
      (assignmentFilter === "unassigned" && !c.assigned_doctor_id);

    return matchesSearch && matchesStatus && matchesReview && matchesAssignment;
  });

  const getStatusBadge = (status) => {
    switch (status) {
      case "completed":
        return <span className="badge badge-teal">AI Analyzed</span>;
      case "preprocessed":
        return <span className="badge badge-preprocessed">Preprocessed</span>;
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

  const pendingReviewCount = cases.filter((c) => !c.review_status || c.review_status === "pending_review").length;
  const underReviewCount = cases.filter((c) => c.review_status === "under_review").length;
  const reviewedCount = cases.filter((c) => c.review_status === "reviewed").length;
  const followUpCount = cases.filter((c) => c.review_status === "follow_up_required").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Top Header */}
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
              {isClinician ? "Clinical Diagnostic Queue" : "Patient Screening Portal"}
            </span>
            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              {cases.length} {cases.length === 1 ? "case registered" : "cases registered"}
            </span>
          </div>
          <h1 style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)" }}>
            {isClinician ? "Clinical Case Review Queue" : "My Skin Screening Cases"}
          </h1>
          <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", marginTop: 2 }}>
            {isClinician
              ? "Review screening cases, inspect standardized lesion tensors, execute AI predictions, and document physician observations."
              : "Track your uploaded lesion photography, automated quality scores, AI decision support, and doctor review guidance."}
          </p>
        </div>

        {/* Header Action Buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          {exportFeedback && (
            <span style={{ fontSize: "0.78rem", color: "var(--text-main)", fontWeight: 600 }}>
              {exportFeedback}
            </span>
          )}
          {cases.length > 0 && isClinician && (
            <div style={{ display: "flex", gap: 6 }}>
              <button
                onClick={handleExportCohortCsv}
                disabled={isExportingCsv}
                className="btn btn-secondary btn-sm"
                title="Export all accessible cases as CSV"
              >
                <Download size={14} />
                <span>{isExportingCsv ? "CSV..." : "Export CSV"}</span>
              </button>
              <button
                onClick={handleExportCohortJson}
                disabled={isExportingJson}
                className="btn btn-secondary btn-sm"
                title="Export all accessible cases as JSON"
              >
                <Download size={14} />
                <span>{isExportingJson ? "JSON..." : "Export JSON"}</span>
              </button>
            </div>
          )}
          <button
            onClick={onOpenNewCaseWorkflow}
            className="btn btn-primary btn-lg"
          >
            <Plus size={18} strokeWidth={2.4} />
            <span>{isClinician ? "Intake New Case" : "New Screening Case"}</span>
          </button>
        </div>
      </div>

      {/* Triage Metrics Strip */}
      <div className="metric-strip">
        <div
          onClick={() => setReviewFilter(reviewFilter === "pending_review" ? "all" : "pending_review")}
          className="metric-tile metric-tile-interactive"
          style={{
            borderLeft: "4px solid var(--warning)",
            background: reviewFilter === "pending_review" ? "var(--warning-bg)" : "#FFFFFF",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
              Pending Review
            </span>
            <Clock size={16} color="var(--warning)" />
          </div>
          <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--warning)", margin: "4px 0 2px" }}>
            {pendingReviewCount}
          </div>
          <div style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
            Awaiting physician evaluation
          </div>
        </div>

        <div
          onClick={() => setReviewFilter(reviewFilter === "under_review" ? "all" : "under_review")}
          className="metric-tile metric-tile-interactive"
          style={{
            borderLeft: "4px solid var(--primary-accent)",
            background: reviewFilter === "under_review" ? "var(--primary-light)" : "#FFFFFF",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
              Under Review
            </span>
            <UserCheck size={16} color="var(--primary-hover)" />
          </div>
          <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--text-main)", margin: "4px 0 2px" }}>
            {underReviewCount}
          </div>
          <div style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
            Claimed by clinician
          </div>
        </div>

        <div
          onClick={() => setReviewFilter(reviewFilter === "reviewed" ? "all" : "reviewed")}
          className="metric-tile metric-tile-interactive"
          style={{
            borderLeft: "4px solid var(--success)",
            background: reviewFilter === "reviewed" ? "var(--success-bg)" : "#FFFFFF",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
              Reviewed
            </span>
            <CheckCircle2 size={16} color="var(--success)" />
          </div>
          <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--success-text)", margin: "4px 0 2px" }}>
            {reviewedCount}
          </div>
          <div style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
            Physician evaluation recorded
          </div>
        </div>

        <div
          onClick={() => setReviewFilter(reviewFilter === "follow_up_required" ? "all" : "follow_up_required")}
          className="metric-tile metric-tile-interactive"
          style={{
            borderLeft: "4px solid var(--error)",
            background: reviewFilter === "follow_up_required" ? "var(--error-bg)" : "#FFFFFF",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 700 }}>
              Follow-Up Required
            </span>
            <MessageSquare size={16} color="var(--error)" />
          </div>
          <div style={{ fontSize: "1.7rem", fontWeight: 800, color: "var(--error-text)", margin: "4px 0 2px" }}>
            {followUpCount}
          </div>
          <div style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
            Priority clinical attention
          </div>
        </div>
      </div>

      {/* Filter and Search Toolbar */}
      <div className="toolbar-row">
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "1 1 260px", maxWidth: 380 }}>
          <div style={{ position: "relative", width: "100%" }}>
            <Search
              size={15}
              color="var(--text-muted)"
              style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)" }}
            />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search code, location, notes..."
              className="form-control"
              style={{ paddingLeft: 34, height: 38 }}
            />
            {search && (
              <button
                onClick={() => setSearch("")}
                style={{
                  position: "absolute",
                  right: 10,
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: "var(--text-muted)",
                }}
              >
                <X size={14} />
              </button>
            )}
          </div>
        </div>

        {/* Filter Selects & View Toggle */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <select
            value={reviewFilter}
            onChange={(e) => setReviewFilter(e.target.value)}
            className="form-control"
            style={{ width: "auto", minWidth: 140, height: 38 }}
          >
            <option value="all">All Reviews</option>
            <option value="pending_review">Pending Review</option>
            <option value="under_review">Under Review</option>
            <option value="reviewed">Reviewed</option>
            <option value="follow_up_required">Follow-up Required</option>
          </select>

          {isClinician && (
            <select
              value={assignmentFilter}
              onChange={(e) => setAssignmentFilter(e.target.value)}
              className="form-control"
              style={{ width: "auto", minWidth: 140, height: 38 }}
            >
              <option value="all">All Assignments</option>
              <option value="assigned_to_me">Assigned to Me</option>
              <option value="unassigned">Unassigned Triage</option>
            </select>
          )}

          {(search || statusFilter !== "all" || reviewFilter !== "all" || assignmentFilter !== "all") && (
            <button
              onClick={() => {
                setSearch("");
                setStatusFilter("all");
                setReviewFilter("all");
                setAssignmentFilter("all");
              }}
              className="btn btn-secondary btn-sm"
              style={{ height: 38 }}
            >
              Reset
            </button>
          )}

          {/* View Toggle */}
          <div
            style={{
              display: "flex",
              background: "var(--bg-surface-subtle)",
              padding: 2,
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--border-color)",
              marginLeft: 4,
            }}
          >
            <button
              type="button"
              onClick={() => setViewMode("table")}
              className={`btn btn-sm ${viewMode === "table" ? "btn-secondary" : "btn-ghost"}`}
              style={{ padding: "6px 10px", height: 32 }}
              title="Table worklist view"
            >
              <ListFilter size={15} />
              <span className="hide-mobile">Table</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode("cards")}
              className={`btn btn-sm ${viewMode === "cards" ? "btn-secondary" : "btn-ghost"}`}
              style={{ padding: "6px 10px", height: 32 }}
              title="Card grid view"
            >
              <LayoutGrid size={15} />
              <span className="hide-mobile">Cards</span>
            </button>
          </div>
        </div>
      </div>

      {/* Case Content */}
      {loading ? (
        <div className="medical-card" style={{ padding: 48, textAlign: "center", color: "var(--text-muted)" }}>
          <p style={{ fontSize: "0.85rem" }}>Loading screening cases...</p>
        </div>
      ) : filteredCases.length === 0 ? (
        <div
          className="medical-card"
          style={{
            padding: "56px 24px",
            textAlign: "center",
          }}
        >
          <FolderOpen size={40} color="var(--text-muted)" style={{ margin: "0 auto 12px", opacity: 0.6 }} />
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 6 }}>
            {cases.length === 0 ? "No Screening Cases Registered" : "No Cases Match Selected Filters"}
          </h3>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", maxWidth: 440, margin: "0 auto 20px", lineHeight: 1.5 }}>
            {cases.length === 0
              ? "Use the clinical intake workflow to register a screening case and upload its standardized skin lesion photograph."
              : "Try adjusting your search query, review status filter, or assignment criteria."}
          </p>

          {cases.length === 0 ? (
            <button
              onClick={onOpenNewCaseWorkflow}
              className="btn btn-primary btn-lg"
            >
              <Plus size={16} />
              <span>Intake First Case</span>
            </button>
          ) : (
            <button
              onClick={() => {
                setSearch("");
                setStatusFilter("all");
                setReviewFilter("all");
                setAssignmentFilter("all");
              }}
              className="btn btn-secondary btn-sm"
            >
              Reset Filters
            </button>
          )}
        </div>
      ) : viewMode === "table" ? (
        /* High-Density Medical Table View */
        <div className="table-container">
          <table className="medical-table">
            <thead>
              <tr>
                <th style={{ width: 60 }}>Photo</th>
                <th>Case Code</th>
                <th>Patient Details</th>
                <th>Anatomical Site</th>
                <th>AI Screening Finding</th>
                <th>Assignment</th>
                <th>Review Status</th>
                <th>Date</th>
                <th style={{ textAlign: "right" }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredCases.map((c) => {
                const hasImg = c.images && c.images.length > 0;
                const latestImg = hasImg ? c.images[c.images.length - 1] : null;
                const hasPredictions = c.predictions && c.predictions.length > 0;
                const latestPred = hasPredictions ? c.predictions[c.predictions.length - 1] : null;

                return (
                  <tr
                    key={c.id}
                    onClick={() => onSelectCase(c)}
                    style={{ cursor: "pointer" }}
                  >
                    <td>
                      <div
                        style={{
                          width: 40,
                          height: 40,
                          borderRadius: "var(--radius-sm)",
                          background: "var(--bg-surface-subtle)",
                          border: "1px solid var(--border-color)",
                          overflow: "hidden",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        {latestImg ? (
                          <img
                            src={api.getImageUrl(latestImg.preview_url || latestImg.original_url)}
                            alt="Lesion"
                            style={{ width: "100%", height: "100%", objectFit: "cover" }}
                          />
                        ) : (
                          <ImageIcon size={18} color="var(--text-muted)" />
                        )}
                      </div>
                    </td>
                    <td>
                      <div style={{ fontFamily: "var(--font-mono)", fontWeight: 700, fontSize: "0.85rem", color: "var(--text-main)" }}>
                        {c.case_code}
                      </div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                        {getStatusBadge(c.status)}
                      </div>
                    </td>
                    <td>
                      <div style={{ fontSize: "0.82rem", fontWeight: 600 }}>
                        {c.patient_age ? `${c.patient_age} yrs` : "Age N/A"}
                      </div>
                      <div style={{ fontSize: "0.74rem", color: "var(--text-secondary)", textTransform: "capitalize" }}>
                        {c.patient_gender || "Unknown"}
                      </div>
                    </td>
                    <td>
                      <div style={{ display: "inline-flex", alignItems: "center", gap: 4, textTransform: "capitalize", fontSize: "0.82rem" }}>
                        <MapPin size={13} color="var(--primary-hover)" />
                        <span>{c.body_location}</span>
                      </div>
                    </td>
                    <td>
                      {latestPred ? (
                        <div>
                          <span
                            className={
                              latestPred.risk_level === "High Risk"
                                ? "badge badge-high"
                                : latestPred.risk_level === "Moderate Risk"
                                ? "badge badge-moderate"
                                : "badge badge-low"
                            }
                            style={{ fontSize: "0.7rem", marginBottom: 2 }}
                          >
                            <Sparkles size={11} />
                            {latestPred.predicted_disease_name || latestPred.predicted_class}
                          </span>
                          <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                            Confidence: {(latestPred.confidence_score * 100).toFixed(1)}%
                          </div>
                        </div>
                      ) : (
                        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Not analyzed</span>
                      )}
                    </td>
                    <td>
                      {c.assigned_doctor_name ? (
                        <span
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--text-main)",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                          }}
                        >
                          <UserCheck size={13} color="var(--primary-hover)" />
                          {(c.assigned_doctor_id === currentUserId || (currentUserId && c.assigned_doctor_id === Number(currentUserId))) ? "Assigned to You" : `Dr. ${c.assigned_doctor_name}`}
                        </span>
                      ) : (
                        <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Unassigned</span>
                      )}
                    </td>
                    <td>
                      {getReviewStatusBadge(c.review_status)}
                    </td>
                    <td>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                        {new Date(c.created_at).toLocaleDateString()}
                      </div>
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectCase(c);
                        }}
                        className="btn btn-secondary btn-sm"
                        style={{ fontSize: "0.78rem", padding: "4px 10px" }}
                      >
                        <span>{isClinician ? "Review" : "View"}</span>
                        <ChevronRight size={14} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        /* Card Grid View */
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 14 }}>
          {filteredCases.map((c) => {
            const hasImg = c.images && c.images.length > 0;
            const latestImg = hasImg ? c.images[c.images.length - 1] : null;
            const hasPredictions = c.predictions && c.predictions.length > 0;
            const latestPred = hasPredictions ? c.predictions[c.predictions.length - 1] : null;

            return (
              <div
                key={c.id}
                onClick={() => onSelectCase(c)}
                className="medical-card medical-card-interactive"
                style={{
                  padding: "16px 18px",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  gap: 12,
                  background: "#FFFFFF",
                }}
              >
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10, marginBottom: 10 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div
                        style={{
                          width: 48,
                          height: 48,
                          borderRadius: "var(--radius-sm)",
                          background: "var(--bg-surface-subtle)",
                          border: "1px solid var(--border-color)",
                          overflow: "hidden",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          flexShrink: 0,
                        }}
                      >
                        {latestImg ? (
                          <img
                            src={api.getImageUrl(latestImg.preview_url || latestImg.original_url)}
                            alt="Lesion"
                            style={{ width: "100%", height: "100%", objectFit: "cover" }}
                          />
                        ) : (
                          <ImageIcon size={20} color="var(--text-muted)" />
                        )}
                      </div>
                      <div>
                        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.95rem", fontWeight: 700, color: "var(--text-main)" }}>
                          {c.case_code}
                        </div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "flex", gap: 8, marginTop: 2 }}>
                          <span>{c.patient_age ? `${c.patient_age}y` : "Age N/A"} • {c.patient_gender || "Unknown"}</span>
                          <span>•</span>
                          <span style={{ textTransform: "capitalize" }}>{c.body_location}</span>
                        </div>
                      </div>
                    </div>

                    {getReviewStatusBadge(c.review_status)}
                  </div>

                  {latestPred && (
                    <div
                      style={{
                        padding: "8px 12px",
                        background: "var(--bg-surface-subtle)",
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid var(--border-color)",
                        marginBottom: 10,
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span
                        className={
                          latestPred.risk_level === "High Risk"
                            ? "badge badge-high"
                            : latestPred.risk_level === "Moderate Risk"
                            ? "badge badge-moderate"
                            : "badge badge-low"
                        }
                        style={{ fontSize: "0.7rem" }}
                      >
                        <Sparkles size={11} />
                        {latestPred.predicted_disease_name || latestPred.predicted_class}
                      </span>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-main)" }}>
                        {(latestPred.confidence_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}

                  {c.clinical_notes && (
                    <p
                      style={{
                        fontSize: "0.78rem",
                        color: "var(--text-muted)",
                        lineHeight: 1.4,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        display: "-webkit-box",
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: "vertical",
                      }}
                    >
                      {c.clinical_notes}
                    </p>
                  )}
                </div>

                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    paddingTop: 10,
                    borderTop: "1px solid var(--border-subtle)",
                    fontSize: "0.75rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <Calendar size={13} />
                    {new Date(c.created_at).toLocaleDateString()}
                  </span>

                  <span style={{ display: "flex", alignItems: "center", gap: 4, fontWeight: 600, color: "var(--text-main)" }}>
                    <span>{isClinician ? "Review Case" : "Details"}</span>
                    <ChevronRight size={15} />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <style>{`
        @media (max-width: 640px) {
          .hide-mobile { display: none !important; }
        }
      `}</style>
    </div>
  );
}
