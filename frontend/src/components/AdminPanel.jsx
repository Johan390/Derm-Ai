import React, { useState, useEffect, useCallback } from "react";
import {
  LayoutDashboard,
  Users,
  User,
  Stethoscope,
  Activity,
  Clock,
  Settings as SettingsIcon,
  LogOut,
  Search,
  Edit2,
  Trash2,
  ShieldCheck,
  ShieldAlert,
  AlertCircle,
  CheckCircle2,
  X,
  RefreshCw,
  Mail,
  Building2,
  Calendar,
  Lock,
  ChevronRight,
  Database,
  FileText
} from "lucide-react";
import { api } from "../api";

export default function AdminPanel({ user, onLogout }) {
  const [activeSection, setActiveSection] = useState("dashboard");

  // Dashboard Stats state
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);

  // Patients state
  const [patients, setPatients] = useState([]);
  const [patientsLoading, setPatientsLoading] = useState(false);
  const [patientSearch, setPatientSearch] = useState("");

  // Doctors state
  const [doctors, setDoctors] = useState([]);
  const [doctorsLoading, setDoctorsLoading] = useState(false);
  const [doctorSearch, setDoctorSearch] = useState("");

  // Analyses state
  const [analyses, setAnalyses] = useState([]);
  const [analysesLoading, setAnalysesLoading] = useState(false);

  // Activity Logs state
  const [logs, setLogs] = useState([]);
  const [logsLoading, setLogsLoading] = useState(false);

  // Settings state
  const [settingsData, setSettingsData] = useState(null);
  const [settingsLoading, setSettingsLoading] = useState(false);

  // Modals & Notifications
  const [editModalUser, setEditModalUser] = useState(null);
  const [editForm, setEditForm] = useState({ full_name: "", email: "", password: "", institution: "" });
  const [editSubmitting, setEditSubmitting] = useState(false);

  const [deleteModalUser, setDeleteModalUser] = useState(null);
  const [deleteSubmitting, setDeleteSubmitting] = useState(false);

  const [notification, setNotification] = useState(null); // { type: 'success'|'error', message: '' }

  const showNotification = (type, message) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 4000);
  };

  // 1. Fetch Dashboard Statistics
  const loadDashboardStats = useCallback(async () => {
    try {
      setStatsLoading(true);
      const data = await api.getAdminStats();
      setStats(data);
    } catch (err) {
      showNotification("error", err.message || "Failed to load dashboard statistics");
    } finally {
      setStatsLoading(false);
    }
  }, []);

  // 2. Fetch Patients
  const loadPatients = useCallback(async (searchQuery = "") => {
    try {
      setPatientsLoading(true);
      const data = await api.getAdminPatients(searchQuery);
      setPatients(data || []);
    } catch (err) {
      showNotification("error", err.message || "Failed to load patient records");
    } finally {
      setPatientsLoading(false);
    }
  }, []);

  // 3. Fetch Doctors
  const loadDoctors = useCallback(async (searchQuery = "") => {
    try {
      setDoctorsLoading(true);
      const data = await api.getAdminDoctors(searchQuery);
      setDoctors(data || []);
    } catch (err) {
      showNotification("error", err.message || "Failed to load doctor records");
    } finally {
      setDoctorsLoading(false);
    }
  }, []);

  // 4. Fetch Analyses
  const loadAnalyses = useCallback(async () => {
    try {
      setAnalysesLoading(true);
      const data = await api.getAdminAnalyses(50);
      setAnalyses(data || []);
    } catch (err) {
      showNotification("error", err.message || "Failed to load analyses");
    } finally {
      setAnalysesLoading(false);
    }
  }, []);

  // 5. Fetch Activity Logs
  const loadActivityLogs = useCallback(async () => {
    try {
      setLogsLoading(true);
      const data = await api.getAdminActivityLogs(50);
      setLogs(data || []);
    } catch (err) {
      showNotification("error", err.message || "Failed to load activity logs");
    } finally {
      setLogsLoading(false);
    }
  }, []);

  // 6. Fetch Settings
  const loadSettings = useCallback(async () => {
    try {
      setSettingsLoading(true);
      const data = await api.getAdminSettings();
      setSettingsData(data);
    } catch (err) {
      showNotification("error", err.message || "Failed to load system settings");
    } finally {
      setSettingsLoading(false);
    }
  }, []);

  // Load initial data based on active section
  useEffect(() => {
    if (activeSection === "dashboard") {
      loadDashboardStats();
    } else if (activeSection === "patients") {
      loadPatients(patientSearch);
    } else if (activeSection === "doctors") {
      loadDoctors(doctorSearch);
    } else if (activeSection === "analyses") {
      loadAnalyses();
    } else if (activeSection === "logs") {
      loadActivityLogs();
    } else if (activeSection === "settings") {
      loadSettings();
    }
  }, [activeSection, loadDashboardStats, loadPatients, loadDoctors, loadAnalyses, loadActivityLogs, loadSettings, patientSearch, doctorSearch]);

  // Handle Edit User Modal Open
  const handleOpenEdit = (targetUser) => {
    setEditModalUser(targetUser);
    setEditForm({
      full_name: targetUser.full_name || "",
      email: targetUser.email || "",
      password: "",
      institution: targetUser.institution || ""
    });
  };

  // Submit Edit User
  const handleSaveEdit = async (e) => {
    e.preventDefault();
    if (!editModalUser) return;
    if (!editForm.full_name.trim() || !editForm.email.trim()) {
      showNotification("error", "Full name and email address are required.");
      return;
    }

    try {
      setEditSubmitting(true);
      const payload = {
        full_name: editForm.full_name.trim(),
        email: editForm.email.trim(),
        institution: editForm.institution ? editForm.institution.trim() : null
      };
      if (editForm.password && editForm.password.trim()) {
        payload.password = editForm.password.trim();
      }

      await api.updateAdminUser(editModalUser.id, payload);
      showNotification("success", `Account details for ${editForm.full_name} updated successfully.`);
      setEditModalUser(null);

      // Refresh current list
      if (activeSection === "patients") loadPatients(patientSearch);
      if (activeSection === "doctors") loadDoctors(doctorSearch);
    } catch (err) {
      showNotification("error", err.message || "Failed to update account details");
    } finally {
      setEditSubmitting(false);
    }
  };

  // Submit Delete User
  const handleConfirmDelete = async () => {
    if (!deleteModalUser) return;
    try {
      setDeleteSubmitting(true);
      await api.deleteAdminUser(deleteModalUser.id);
      showNotification("success", `Account for ${deleteModalUser.full_name} (${deleteModalUser.role}) deleted successfully.`);
      setDeleteModalUser(null);

      // Refresh data
      if (activeSection === "patients") loadPatients(patientSearch);
      if (activeSection === "doctors") loadDoctors(doctorSearch);
      if (activeSection === "dashboard") loadDashboardStats();
    } catch (err) {
      showNotification("error", err.message || "Failed to delete account");
    } finally {
      setDeleteSubmitting(false);
    }
  };

  // Quick Doctor Verification Toggle
  const handleVerifyDoctor = async (doctorId, doctorName) => {
    try {
      await api.verifyClinician(doctorId);
      showNotification("success", `Dr. ${doctorName} has been verified for clinical practice.`);
      loadDoctors(doctorSearch);
    } catch (err) {
      showNotification("error", err.message || "Failed to verify clinician");
    }
  };

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "patients", label: "Patients", icon: Users },
    { id: "doctors", label: "Doctors", icon: Stethoscope },
    { id: "analyses", label: "Analyses", icon: Activity },
    { id: "logs", label: "Activity Logs", icon: Clock },
    { id: "settings", label: "Settings", icon: SettingsIcon },
  ];

  return (
    <div style={{ display: "flex", minHeight: "calc(100vh - 64px)", background: "var(--bg-page)", width: "100%" }}>
      {/* --------------------------------------------------------------------- */}
      {/* 2. Admin Sidebar (Left Navigation) */}
      {/* --------------------------------------------------------------------- */}
      <aside
        style={{
          width: 240,
          background: "#FFFFFF",
          borderRight: "1px solid var(--border-color)",
          display: "flex",
          flexDirection: "column",
          flexShrink: 0,
        }}
      >
        {/* Sidebar Brand Header */}
        <div
          style={{
            padding: "20px 18px",
            borderBottom: "1px solid var(--border-subtle)",
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: "var(--radius-md)",
              background: "var(--primary-light)",
              color: "var(--text-main)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "1px solid var(--primary-light-border)",
            }}
          >
            <ShieldCheck size={20} />
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: "0.95rem", color: "var(--text-main)", letterSpacing: "-0.01em" }}>
              Admin Panel
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", fontWeight: 500 }}>
              DermAI Platform
            </div>
          </div>
        </div>

        {/* Sidebar Navigation Items */}
        <nav style={{ padding: "12px 10px", display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveSection(item.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "10px 14px",
                  borderRadius: "var(--radius-sm)",
                  border: "none",
                  background: isActive ? "var(--primary-light)" : "transparent",
                  color: isActive ? "var(--text-main)" : "var(--text-secondary)",
                  fontWeight: isActive ? 700 : 500,
                  fontSize: "0.875rem",
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "all 0.15s ease",
                  borderLeft: isActive ? "3px solid var(--primary-hover)" : "3px solid transparent",
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.background = "var(--bg-surface-subtle)";
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.background = "transparent";
                }}
              >
                <Icon size={18} color={isActive ? "var(--text-main)" : "var(--text-secondary)"} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Sidebar Footer with Logout */}
        <div style={{ padding: "14px 10px", borderTop: "1px solid var(--border-subtle)" }}>
          <button
            onClick={onLogout}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              padding: "10px 14px",
              width: "100%",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-color)",
              background: "#FFFFFF",
              color: "var(--error-text)",
              fontWeight: 600,
              fontSize: "0.875rem",
              cursor: "pointer",
              transition: "background 0.15s ease",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--error-bg)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "#FFFFFF")}
          >
            <LogOut size={18} color="var(--error-text)" />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* --------------------------------------------------------------------- */}
      {/* Main Content Area */}
      {/* --------------------------------------------------------------------- */}
      <main style={{ flex: 1, padding: "24px 32px", overflowY: "auto" }}>
        {/* Top Alert Notification Banner */}
        {notification && (
          <div
            style={{
              marginBottom: 20,
              padding: "12px 18px",
              borderRadius: "var(--radius-sm)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              background: notification.type === "success" ? "var(--success-bg)" : "var(--error-bg)",
              border: `1px solid ${notification.type === "success" ? "var(--success-border)" : "var(--error-border)"}`,
              color: notification.type === "success" ? "var(--success-text)" : "var(--error-text)",
              fontSize: "0.875rem",
              fontWeight: 500,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              {notification.type === "success" ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
              <span>{notification.message}</span>
            </div>
            <button
              onClick={() => setNotification(null)}
              style={{ background: "none", border: "none", cursor: "pointer", color: "inherit", padding: 4 }}
            >
              <X size={16} />
            </button>
          </div>
        )}

        {/* =================================================================== */}
        {/* 1. SECTION: DASHBOARD */}
        {/* =================================================================== */}
        {activeSection === "dashboard" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <h1 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-main)" }}>
                  Administrative Dashboard
                </h1>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  System-wide clinical screening metrics, user distribution and real-time audit feed
                </p>
              </div>
              <button
                onClick={loadDashboardStats}
                disabled={statsLoading}
                className="btn btn-secondary btn-sm"
              >
                <RefreshCw size={14} className={statsLoading ? "spin" : ""} />
                <span>Refresh</span>
              </button>
            </div>

            {/* Statistic Cards Grid */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: 16,
              }}
            >
              {/* Total Users */}
              <div
                className="medical-card"
                style={{ padding: "20px 22px", background: "#FFFFFF", display: "flex", alignItems: "center", gap: 16 }}
              >
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: "var(--radius-md)",
                    background: "#EBF3FF",
                    color: "#2563EB",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Users size={22} />
                </div>
                <div>
                  <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase" }}>
                    Total Users
                  </div>
                  <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)", marginTop: 2 }}>
                    {stats ? stats.total_users : "—"}
                  </div>
                </div>
              </div>

              {/* Total Patients */}
              <div
                className="medical-card"
                style={{ padding: "20px 22px", background: "#FFFFFF", display: "flex", alignItems: "center", gap: 16 }}
              >
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
                  }}
                >
                  <User size={22} />
                </div>
                <div>
                  <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase" }}>
                    Total Patients
                  </div>
                  <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)", marginTop: 2 }}>
                    {stats ? stats.total_patients : "—"}
                  </div>
                </div>
              </div>

              {/* Total Doctors */}
              <div
                className="medical-card"
                style={{ padding: "20px 22px", background: "#FFFFFF", display: "flex", alignItems: "center", gap: 16 }}
              >
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: "var(--radius-md)",
                    background: "var(--success-bg)",
                    color: "var(--success)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Stethoscope size={22} />
                </div>
                <div>
                  <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase" }}>
                    Total Doctors
                  </div>
                  <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)", marginTop: 2 }}>
                    {stats ? stats.total_doctors : "—"}
                  </div>
                </div>
              </div>

              {/* Total Analyses */}
              <div
                className="medical-card"
                style={{ padding: "20px 22px", background: "#FFFFFF", display: "flex", alignItems: "center", gap: 16 }}
              >
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: "var(--radius-md)",
                    background: "#F3E8FF",
                    color: "#9333EA",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Activity size={22} />
                </div>
                <div>
                  <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase" }}>
                    Total Analyses
                  </div>
                  <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-main)", marginTop: 2 }}>
                    {stats ? stats.total_analyses : "—"}
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Activity Section */}
            <div className="medical-card" style={{ padding: "22px 26px", background: "#FFFFFF" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <div>
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
                    Recent Activity
                  </h2>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 1 }}>
                    Chronological activity log from the live database
                  </p>
                </div>
                <button
                  onClick={() => setActiveSection("logs")}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--text-main)",
                    fontWeight: 600,
                    fontSize: "0.8rem",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                  }}
                >
                  <span>View All Logs</span>
                  <ChevronRight size={14} />
                </button>
              </div>

              {statsLoading ? (
                <div style={{ padding: "32px 0", textAlign: "center", color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                  Loading recent activity records...
                </div>
              ) : !stats || !stats.recent_activity || stats.recent_activity.length === 0 ? (
                <div style={{ padding: "32px 0", textAlign: "center", color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                  No recent activity recorded yet.
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {stats.recent_activity.map((item, idx) => (
                    <div
                      key={item.id || idx}
                      style={{
                        padding: "10px 14px",
                        borderRadius: "var(--radius-sm)",
                        background: "var(--bg-surface-subtle)",
                        border: "1px solid var(--border-subtle)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        flexWrap: "wrap",
                        gap: 12,
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                        <span
                          className={`badge ${
                            item.action.includes("DELETED")
                              ? "badge-rose"
                              : item.action.includes("UPDATED") || item.action.includes("RESET")
                              ? "badge-amber"
                              : "badge-teal"
                          }`}
                          style={{ fontSize: "0.7rem", fontWeight: 700 }}
                        >
                          {item.action.replace(/_/g, " ")}
                        </span>
                        <span style={{ fontSize: "0.85rem", color: "var(--text-main)", fontWeight: 500 }}>
                          {item.details}
                        </span>
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 6 }}>
                        <Clock size={12} />
                        <span>{item.created_at ? new Date(item.created_at).toLocaleString() : "Just now"}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* =================================================================== */}
        {/* 2. SECTION: PATIENTS */}
        {/* =================================================================== */}
        {activeSection === "patients" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Header & Search */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: 16,
              }}
            >
              <div>
                <h1 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-main)" }}>
                  Patient Management
                </h1>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Search, review details, change passwords and manage patient accounts
                </p>
              </div>

              {/* Search Bar */}
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ position: "relative" }}>
                  <Search size={16} color="var(--text-muted)" style={{ position: "absolute", left: 12, top: 10 }} />
                  <input
                    type="text"
                    placeholder="Search name or email..."
                    value={patientSearch}
                    onChange={(e) => setPatientSearch(e.target.value)}
                    style={{
                      padding: "8px 12px 8px 34px",
                      borderRadius: "var(--radius-sm)",
                      border: "1px solid var(--border-color)",
                      background: "#FFFFFF",
                      fontSize: "0.85rem",
                      width: 240,
                      outline: "none",
                    }}
                  />
                </div>
                <button
                  onClick={() => loadPatients(patientSearch)}
                  disabled={patientsLoading}
                  className="btn btn-secondary btn-sm"
                >
                  <RefreshCw size={14} className={patientsLoading ? "spin" : ""} />
                </button>
              </div>
            </div>

            {/* Patients Table */}
            <div className="medical-card" style={{ background: "#FFFFFF", overflow: "hidden" }}>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                  <thead>
                    <tr style={{ background: "var(--bg-surface-subtle)", borderBottom: "1px solid var(--border-color)" }}>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        PATIENT NAME
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        EMAIL ADDRESS
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        REGISTRATION DATE
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)", textAlign: "right" }}>
                        ACTIONS
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {patientsLoading ? (
                      <tr>
                        <td colSpan="4" style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>
                          Loading patient records...
                        </td>
                      </tr>
                    ) : patients.length === 0 ? (
                      <tr>
                        <td colSpan="4" style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>
                          No patients found matching your search.
                        </td>
                      </tr>
                    ) : (
                      patients.map((pat) => (
                        <tr
                          key={pat.id}
                          style={{ borderBottom: "1px solid var(--border-subtle)", transition: "background 0.1s ease" }}
                          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-surface-subtle)")}
                          onMouseLeave={(e) => (e.currentTarget.style.background = "#FFFFFF")}
                        >
                          <td style={{ padding: "12px 18px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                              <div
                                style={{
                                  width: 32,
                                  height: 32,
                                  borderRadius: "50%",
                                  background: "var(--primary-light)",
                                  color: "var(--text-main)",
                                  fontWeight: 700,
                                  fontSize: "0.8rem",
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                }}
                              >
                                {pat.full_name ? pat.full_name[0].toUpperCase() : "P"}
                              </div>
                              <div>
                                <div style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-main)" }}>
                                  {pat.full_name}
                                </div>
                                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                  @{pat.username}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.85rem", color: "var(--text-main)" }}>
                            {pat.email}
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                            {new Date(pat.created_at).toLocaleDateString(undefined, {
                              year: "numeric",
                              month: "short",
                              day: "numeric",
                            })}
                          </td>
                          <td style={{ padding: "12px 18px", textAlign: "right" }}>
                            <div style={{ display: "inline-flex", gap: 8 }}>
                              <button
                                onClick={() => handleOpenEdit(pat)}
                                className="btn btn-secondary btn-sm"
                                style={{ padding: "6px 10px", fontSize: "0.78rem" }}
                                title="Edit patient details or change password"
                              >
                                <Edit2 size={13} />
                                <span>Edit</span>
                              </button>
                              <button
                                onClick={() => setDeleteModalUser(pat)}
                                className="btn btn-danger btn-sm"
                                style={{ padding: "6px 10px", fontSize: "0.78rem" }}
                                title="Delete patient account"
                              >
                                <Trash2 size={13} />
                                <span>Delete</span>
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* =================================================================== */}
        {/* 3. SECTION: DOCTORS */}
        {/* =================================================================== */}
        {activeSection === "doctors" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Header & Search */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: 16,
              }}
            >
              <div>
                <h1 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-main)" }}>
                  Doctor Management
                </h1>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Oversee clinician practitioner accounts, credentials, and verification statuses
                </p>
              </div>

              {/* Search Bar */}
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ position: "relative" }}>
                  <Search size={16} color="var(--text-muted)" style={{ position: "absolute", left: 12, top: 10 }} />
                  <input
                    type="text"
                    placeholder="Search name or email..."
                    value={doctorSearch}
                    onChange={(e) => setDoctorSearch(e.target.value)}
                    style={{
                      padding: "8px 12px 8px 34px",
                      borderRadius: "var(--radius-sm)",
                      border: "1px solid var(--border-color)",
                      background: "#FFFFFF",
                      fontSize: "0.85rem",
                      width: 240,
                      outline: "none",
                    }}
                  />
                </div>
                <button
                  onClick={() => loadDoctors(doctorSearch)}
                  disabled={doctorsLoading}
                  className="btn btn-secondary btn-sm"
                >
                  <RefreshCw size={14} className={doctorsLoading ? "spin" : ""} />
                </button>
              </div>
            </div>

            {/* Doctors Table */}
            <div className="medical-card" style={{ background: "#FFFFFF", overflow: "hidden" }}>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                  <thead>
                    <tr style={{ background: "var(--bg-surface-subtle)", borderBottom: "1px solid var(--border-color)" }}>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        DOCTOR NAME & INSTITUTION
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        EMAIL ADDRESS
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        VERIFICATION STATUS
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        REGISTRATION DATE
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)", textAlign: "right" }}>
                        ACTIONS
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {doctorsLoading ? (
                      <tr>
                        <td colSpan="5" style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>
                          Loading doctor records...
                        </td>
                      </tr>
                    ) : doctors.length === 0 ? (
                      <tr>
                        <td colSpan="5" style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>
                          No doctors found matching your search.
                        </td>
                      </tr>
                    ) : (
                      doctors.map((doc) => (
                        <tr
                          key={doc.id}
                          style={{ borderBottom: "1px solid var(--border-subtle)", transition: "background 0.1s ease" }}
                          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-surface-subtle)")}
                          onMouseLeave={(e) => (e.currentTarget.style.background = "#FFFFFF")}
                        >
                          <td style={{ padding: "12px 18px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                              <div
                                style={{
                                  width: 32,
                                  height: 32,
                                  borderRadius: "50%",
                                  background: "var(--success-bg)",
                                  color: "var(--success-text)",
                                  fontWeight: 700,
                                  fontSize: "0.8rem",
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                }}
                              >
                                {doc.full_name ? doc.full_name[0].toUpperCase() : "D"}
                              </div>
                              <div>
                                <div style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-main)" }}>
                                  {doc.full_name}
                                </div>
                                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                  {doc.institution || "Independent Practice"}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.85rem", color: "var(--text-main)" }}>
                            {doc.email}
                          </td>
                          <td style={{ padding: "12px 18px" }}>
                            {doc.is_verified ? (
                              <span className="badge badge-teal" style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                                <CheckCircle2 size={12} />
                                <span>Verified</span>
                              </span>
                            ) : (
                              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <span className="badge badge-amber" style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                                  <Clock size={12} />
                                  <span>Pending</span>
                                </span>
                                <button
                                  onClick={() => handleVerifyDoctor(doc.id, doc.full_name)}
                                  className="btn btn-outline btn-sm"
                                  style={{ padding: "3px 8px", fontSize: "0.72rem" }}
                                >
                                  Approve
                                </button>
                              </div>
                            )}
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                            {new Date(doc.created_at).toLocaleDateString(undefined, {
                              year: "numeric",
                              month: "short",
                              day: "numeric",
                            })}
                          </td>
                          <td style={{ padding: "12px 18px", textAlign: "right" }}>
                            <div style={{ display: "inline-flex", gap: 8 }}>
                              <button
                                onClick={() => handleOpenEdit(doc)}
                                className="btn btn-secondary btn-sm"
                                style={{ padding: "6px 10px", fontSize: "0.78rem" }}
                                title="Edit doctor details or change password"
                              >
                                <Edit2 size={13} />
                                <span>Edit</span>
                              </button>
                              <button
                                onClick={() => setDeleteModalUser(doc)}
                                className="btn btn-danger btn-sm"
                                style={{ padding: "6px 10px", fontSize: "0.78rem" }}
                                title="Delete doctor account"
                              >
                                <Trash2 size={13} />
                                <span>Delete</span>
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* =================================================================== */}
        {/* 4. SECTION: ANALYSES */}
        {/* =================================================================== */}
        {activeSection === "analyses" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <h1 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-main)" }}>
                  Clinical Analyses
                </h1>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  System-wide automated deep learning screenings, confidence scores and review outcomes
                </p>
              </div>
              <button
                onClick={loadAnalyses}
                disabled={analysesLoading}
                className="btn btn-secondary btn-sm"
              >
                <RefreshCw size={14} className={analysesLoading ? "spin" : ""} />
                <span>Refresh</span>
              </button>
            </div>

            <div className="medical-card" style={{ background: "#FFFFFF", overflow: "hidden" }}>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                  <thead>
                    <tr style={{ background: "var(--bg-surface-subtle)", borderBottom: "1px solid var(--border-color)" }}>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        CASE CODE
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        PATIENT
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        ANATOMICAL SITE
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        PREDICTED DIAGNOSIS
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        CONFIDENCE & RISK
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        MODEL
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        DATE
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysesLoading ? (
                      <tr>
                        <td colSpan="7" style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>
                          Loading diagnostic analyses...
                        </td>
                      </tr>
                    ) : analyses.length === 0 ? (
                      <tr>
                        <td colSpan="7" style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>
                          No clinical analyses found in the database.
                        </td>
                      </tr>
                    ) : (
                      analyses.map((an) => (
                        <tr
                          key={an.id}
                          style={{ borderBottom: "1px solid var(--border-subtle)", transition: "background 0.1s ease" }}
                          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-surface-subtle)")}
                          onMouseLeave={(e) => (e.currentTarget.style.background = "#FFFFFF")}
                        >
                          <td style={{ padding: "12px 18px", fontWeight: 700, fontSize: "0.85rem", color: "var(--text-main)" }}>
                            {an.case_code}
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.85rem", color: "var(--text-main)" }}>
                            {an.patient_name}
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.82rem", color: "var(--text-secondary)", textTransform: "capitalize" }}>
                            {an.body_location}
                          </td>
                          <td style={{ padding: "12px 18px" }}>
                            <span style={{ fontWeight: 600, fontSize: "0.85rem", color: "var(--text-main)" }}>
                              {an.predicted_disease_name}
                            </span>
                            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginLeft: 6 }}>
                              ({an.predicted_class.toUpperCase()})
                            </span>
                          </td>
                          <td style={{ padding: "12px 18px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                              <span style={{ fontWeight: 700, fontSize: "0.85rem" }}>
                                {(an.confidence_score * 100).toFixed(1)}%
                              </span>
                              <span
                                className={`badge ${
                                  an.risk_level === "High Risk"
                                    ? "badge-rose"
                                    : an.risk_level === "Moderate Risk"
                                    ? "badge-amber"
                                    : "badge-teal"
                                }`}
                                style={{ fontSize: "0.7rem" }}
                              >
                                {an.risk_level}
                              </span>
                            </div>
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                            {an.model_name.replace("DermAI_", "")}
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                            {an.created_at ? new Date(an.created_at).toLocaleDateString() : "—"}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* =================================================================== */}
        {/* 5. SECTION: ACTIVITY LOGS */}
        {/* =================================================================== */}
        {activeSection === "logs" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <h1 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-main)" }}>
                  Activity Logs
                </h1>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Comprehensive audit trail of registrations, updates, deletions and clinical events
                </p>
              </div>
              <button
                onClick={loadActivityLogs}
                disabled={logsLoading}
                className="btn btn-secondary btn-sm"
              >
                <RefreshCw size={14} className={logsLoading ? "spin" : ""} />
                <span>Refresh</span>
              </button>
            </div>

            <div className="medical-card" style={{ background: "#FFFFFF", overflow: "hidden" }}>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                  <thead>
                    <tr style={{ background: "var(--bg-surface-subtle)", borderBottom: "1px solid var(--border-color)" }}>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        TIMESTAMP
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        ACTION
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        ACTOR
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        DETAILS
                      </th>
                      <th style={{ padding: "12px 18px", fontSize: "0.78rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        IP ADDRESS
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {logsLoading ? (
                      <tr>
                        <td colSpan="5" style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>
                          Loading audit records...
                        </td>
                      </tr>
                    ) : logs.length === 0 ? (
                      <tr>
                        <td colSpan="5" style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>
                          No activity logs found.
                        </td>
                      </tr>
                    ) : (
                      logs.map((lg) => (
                        <tr
                          key={lg.id}
                          style={{ borderBottom: "1px solid var(--border-subtle)" }}
                        >
                          <td style={{ padding: "12px 18px", fontSize: "0.8rem", color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
                            {lg.created_at ? new Date(lg.created_at).toLocaleString() : "—"}
                          </td>
                          <td style={{ padding: "12px 18px" }}>
                            <span
                              className={`badge ${
                                lg.action.includes("DELETED")
                                  ? "badge-rose"
                                  : lg.action.includes("UPDATED") || lg.action.includes("RESET")
                                  ? "badge-amber"
                                  : "badge-teal"
                              }`}
                              style={{ fontSize: "0.7rem", fontWeight: 700 }}
                            >
                              {lg.action.replace(/_/g, " ")}
                            </span>
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.85rem", fontWeight: 600, color: "var(--text-main)" }}>
                            {lg.actor_name || "System"}
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.85rem", color: "var(--text-main)" }}>
                            {lg.details}
                          </td>
                          <td style={{ padding: "12px 18px", fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                            {lg.ip_address || "127.0.0.1"}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* =================================================================== */}
        {/* 6. SECTION: SETTINGS */}
        {/* =================================================================== */}
        {activeSection === "settings" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <h1 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-main)" }}>
                  System Settings
                </h1>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Configuration parameters, cryptographic security settings and admin profile
                </p>
              </div>
              <button
                onClick={loadSettings}
                disabled={settingsLoading}
                className="btn btn-secondary btn-sm"
              >
                <RefreshCw size={14} className={settingsLoading ? "spin" : ""} />
                <span>Refresh</span>
              </button>
            </div>

            {settingsLoading || !settingsData ? (
              <div className="medical-card" style={{ padding: 40, textAlign: "center", color: "var(--text-secondary)" }}>
                Loading system settings...
              </div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
                {/* Admin Profile Card */}
                <div className="medical-card" style={{ padding: "24px 26px", background: "#FFFFFF" }}>
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 16 }}>
                    Administrator Profile
                  </h2>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>FULL NAME</div>
                      <div style={{ fontSize: "0.9rem", color: "var(--text-main)", fontWeight: 700, marginTop: 2 }}>
                        {settingsData.admin_user?.full_name || user?.full_name}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>USERNAME</div>
                      <div style={{ fontSize: "0.9rem", color: "var(--text-main)", marginTop: 2 }}>
                        @{settingsData.admin_user?.username || user?.username}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>EMAIL ADDRESS</div>
                      <div style={{ fontSize: "0.9rem", color: "var(--text-main)", marginTop: 2 }}>
                        {settingsData.admin_user?.email || user?.email}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>SYSTEM ROLE</div>
                      <div style={{ marginTop: 4 }}>
                        <span className="badge badge-teal" style={{ textTransform: "uppercase" }}>
                          {settingsData.admin_user?.role || "ADMIN"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* System & Security Parameters */}
                <div className="medical-card" style={{ padding: "24px 26px", background: "#FFFFFF" }}>
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)", marginBottom: 16 }}>
                    Security & Engine Runtime
                  </h2>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>PROJECT NAME</div>
                      <div style={{ fontSize: "0.9rem", color: "var(--text-main)", fontWeight: 600, marginTop: 2 }}>
                        {settingsData.project_name} (v{settingsData.version})
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>ENVIRONMENT MODE</div>
                      <div style={{ fontSize: "0.9rem", color: "var(--text-main)", marginTop: 2, textTransform: "capitalize" }}>
                        {settingsData.environment}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>JWT TOKEN SIGNING</div>
                      <div style={{ fontSize: "0.9rem", color: "var(--text-main)", marginTop: 2 }}>
                        Algorithm: {settingsData.algorithm} | Expiration: {settingsData.token_expire_minutes} minutes
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>MAX IMAGE UPLOAD PAYLOAD</div>
                      <div style={{ fontSize: "0.9rem", color: "var(--text-main)", marginTop: 2 }}>
                        {settingsData.max_file_size_mb} MB
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>IQA BLUR LAPLACIAN REJECTION THRESHOLD</div>
                      <div style={{ fontSize: "0.9rem", color: "var(--text-main)", marginTop: 2 }}>
                        {settingsData.iqa_blur_threshold}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* --------------------------------------------------------------------- */}
      {/* Edit User Modal Dialog */}
      {/* --------------------------------------------------------------------- */}
      {editModalUser && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(23, 59, 69, 0.4)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 999,
            padding: 16,
          }}
        >
          <div
            className="medical-card"
            style={{
              width: "100%",
              maxWidth: 460,
              background: "#FFFFFF",
              padding: "24px 28px",
              boxShadow: "0 10px 25px rgba(23, 59, 69, 0.15)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
              <div>
                <h3 style={{ fontSize: "1.15rem", fontWeight: 800, color: "var(--text-main)" }}>
                  Edit {editModalUser.role === "clinician" ? "Doctor" : "Patient"} Details
                </h3>
                <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Account: @{editModalUser.username}
                </p>
              </div>
              <button
                onClick={() => setEditModalUser(null)}
                style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)" }}
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleSaveEdit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Full Name */}
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-main)", marginBottom: 4 }}>
                  Full Name *
                </label>
                <input
                  type="text"
                  required
                  value={editForm.full_name}
                  onChange={(e) => setEditForm({ ...editForm, full_name: e.target.value })}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid var(--border-color)",
                    fontSize: "0.875rem",
                    boxSizing: "border-box",
                  }}
                />
              </div>

              {/* Email Address */}
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-main)", marginBottom: 4 }}>
                  Email Address *
                </label>
                <input
                  type="email"
                  required
                  value={editForm.email}
                  onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid var(--border-color)",
                    fontSize: "0.875rem",
                    boxSizing: "border-box",
                  }}
                />
              </div>

              {/* Institution (for Doctors only) */}
              {editModalUser.role === "clinician" && (
                <div>
                  <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-main)", marginBottom: 4 }}>
                    Hospital or Institution
                  </label>
                  <input
                    type="text"
                    value={editForm.institution}
                    onChange={(e) => setEditForm({ ...editForm, institution: e.target.value })}
                    style={{
                      width: "100%",
                      padding: "8px 12px",
                      borderRadius: "var(--radius-sm)",
                      border: "1px solid var(--border-color)",
                      fontSize: "0.875rem",
                      boxSizing: "border-box",
                    }}
                  />
                </div>
              )}

              {/* New Password (Optional) */}
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 600, color: "var(--text-main)", marginBottom: 4 }}>
                  New Password (optional)
                </label>
                <input
                  type="password"
                  placeholder="Leave empty to keep existing password"
                  value={editForm.password}
                  onChange={(e) => setEditForm({ ...editForm, password: e.target.value })}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid var(--border-color)",
                    fontSize: "0.875rem",
                    boxSizing: "border-box",
                  }}
                />
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>
                  If left empty, the current password will remain unchanged. Minimum 6 characters if provided.
                </div>
              </div>

              {/* Action Buttons */}
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 10 }}>
                <button
                  type="button"
                  onClick={() => setEditModalUser(null)}
                  disabled={editSubmitting}
                  className="btn btn-secondary"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={editSubmitting}
                  className="btn btn-primary"
                >
                  {editSubmitting ? "Saving..." : "Save Changes"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* --------------------------------------------------------------------- */}
      {/* Delete User Confirmation Modal */}
      {/* --------------------------------------------------------------------- */}
      {deleteModalUser && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(23, 59, 69, 0.4)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 999,
            padding: 16,
          }}
        >
          <div
            className="medical-card"
            style={{
              width: "100%",
              maxWidth: 440,
              background: "#FFFFFF",
              padding: "24px 28px",
              boxShadow: "0 10px 25px rgba(23, 59, 69, 0.15)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
              <div
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: "50%",
                  background: "var(--error-bg)",
                  color: "var(--error-text)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Trash2 size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-main)" }}>
                  Confirm Account Deletion
                </h3>
                <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
                  Irreversible administrative operation
                </div>
              </div>
            </div>

            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 16 }}>
              Are you sure you want to permanently delete the {deleteModalUser.role} account for{" "}
              <strong style={{ color: "var(--text-main)" }}>{deleteModalUser.full_name}</strong> (
              <code style={{ fontSize: "0.8rem", color: "var(--text-main)" }}>{deleteModalUser.email}</code>)?
            </p>

            {deleteModalUser.role === "clinician" ? (
              <div
                style={{
                  padding: "10px 14px",
                  background: "var(--bg-surface-subtle)",
                  borderRadius: "var(--radius-sm)",
                  fontSize: "0.78rem",
                  color: "var(--text-secondary)",
                  marginBottom: 18,
                  border: "1px solid var(--border-color)",
                }}
              >
                Safe Deletion: Any patient cases assigned to this doctor will be safely returned to the unassigned triage queue. Patient screening cases will not be deleted.
              </div>
            ) : (
              <div
                style={{
                  padding: "10px 14px",
                  background: "var(--warning-bg)",
                  borderRadius: "var(--radius-sm)",
                  fontSize: "0.78rem",
                  color: "var(--warning-text)",
                  marginBottom: 18,
                  border: "1px solid var(--warning-border)",
                }}
              >
                Patient Data Notice: Deleting this patient account will remove their associated uploaded photos and screening records.
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
              <button
                type="button"
                onClick={() => setDeleteModalUser(null)}
                disabled={deleteSubmitting}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmDelete}
                disabled={deleteSubmitting}
                className="btn btn-danger"
              >
                {deleteSubmitting ? "Deleting..." : "Delete Account"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
