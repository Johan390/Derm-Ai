import React, { useState, useEffect, useCallback } from "react";
import "./App.css";
import { Lock, ArrowRight, Activity, ShieldAlert, Database, Clock } from "lucide-react";
import Navbar from "./components/Navbar";
import Dashboard from "./components/Dashboard";
import CaseList from "./components/CaseList";
import CaseDetail from "./components/CaseDetail";
import CreateCaseAndUpload from "./components/CreateCaseAndUpload";
import DatasetView from "./components/DatasetView";
import AdminCliniciansView from "./components/AdminCliniciansView";
import AdminPanel from "./components/AdminPanel";
import ClinicalGuidelines from "./components/ClinicalGuidelines";
import LandingPage from "./components/LandingPage";
import LoginModal from "./components/LoginModal";
import { api, getCurrentUser, getAuthToken } from "./api";

export default function App() {
  const [user, setUser] = useState(() => {
    const token = getAuthToken();
    return token ? getCurrentUser() : null;
  });

  const [activeTab, setActiveTab] = useState(() => {
    const params = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
    if (params && params.get("tab")) {
      return params.get("tab");
    }
    const token = getAuthToken();
    const u = token ? getCurrentUser() : null;
    if (!u) return "landing";
    if (u.role === "admin") return "admin-panel";
    if (u.role === "clinician") return "dashboard";
    return "cases";
  });

  const [cases, setCases] = useState([]);
  const [selectedCase, setSelectedCase] = useState(null);
  const [isLoginOpen, setIsLoginOpen] = useState(() => {
    const params = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
    return params ? params.get("login") === "true" : false;
  });
  const [isRegisterMode, setIsRegisterMode] = useState(false);
  const [loading, setLoading] = useState(false);

  const isDedicatedResearcher = false;
  const isUnverifiedClinician = user && user.role === "clinician" && user.is_verified === false;

  // Fetch Cases handler (Clinical roles only)
  const loadCases = useCallback(async () => {
    const currentUser = getCurrentUser();
    if (
      !currentUser ||
      (currentUser.role === "clinician" && currentUser.is_verified === false)
    ) {
      return;
    }
    try {
      setLoading(true);
      const data = await api.getCases();
      setCases(data);
    } catch (err) {
      console.warn("Failed to fetch cases:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Synchronize case data when user session changes
  useEffect(() => {
    let ignore = false;
    if (user && !(user.role === "clinician" && user.is_verified === false)) {
      setLoading(true);
      api.getCases()
        .then((data) => {
          if (!ignore) {
            setCases(data);
            setLoading(false);
          }
        })
        .catch((err) => {
          if (!ignore) {
            console.warn("Failed to load initial cases:", err);
            setLoading(false);
          }
        });
    }
    return () => {
      ignore = true;
    };
  }, [user]);

  const handleAuthSuccess = (userData) => {
    setUser(userData);
    setSelectedCase(null);
    if (userData.role === "admin") {
      setActiveTab("admin-panel");
    } else if (userData.role === "clinician") {
      loadCases();
      setActiveTab("dashboard");
    } else {
      loadCases();
      setActiveTab("cases");
    }
  };

  const handleLogout = () => {
    api.logout();
    setUser(null);
    setCases([]);
    setSelectedCase(null);
    setActiveTab("landing");
  };

  const handleCaseCreatedAndProcessed = (newCase) => {
    setCases((prev) => [newCase, ...prev.filter((c) => c.id !== newCase.id)]);
  };

  const handleCaseUpdated = (updatedCase) => {
    setCases((prev) => prev.map((c) => (c.id === updatedCase.id ? updatedCase : c)));
    setSelectedCase(updatedCase);
  };

  const openSignInModal = () => {
    setIsRegisterMode(false);
    setIsLoginOpen(true);
  };

  const openRegisterModal = () => {
    setIsRegisterMode(true);
    setIsLoginOpen(true);
  };

  return (
    <div className="app-wrapper">
      {/* Global Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          if (tab !== "cases") setSelectedCase(null);
        }}
        user={user}
        onOpenLogin={openSignInModal}
        onOpenRegister={openRegisterModal}
        onLogout={handleLogout}
        onSelectCase={async (caseId) => {
          try {
            const c = await api.getCase(caseId);
            setSelectedCase(c);
            setActiveTab("cases");
          } catch (e) {
            console.warn("Could not navigate to case from notification:", e);
          }
        }}
      />

      {/* Main Content Area */}
      <main className="main-content">
        <div className="app-container">
          {/* TAB: PUBLIC LANDING PAGE */}
          {activeTab === "landing" && (
            <LandingPage
              onGetStarted={() => {
                if (user) {
                  setActiveTab(user.role === "admin" ? "admin-panel" : user.role === "clinician" ? "dashboard" : "cases");
                } else {
                  openSignInModal();
                }
              }}
              onExploreResearch={() => setActiveTab("research-dashboard")}
            />
          )}

          {/* TAB: CLINICAL GUIDELINES & REFERENCE */}
          {activeTab === "guidelines" && (
            <ClinicalGuidelines />
          )}

          {/* TAB: DASHBOARD (Clinician / Admin Overview) */}
          {activeTab === "dashboard" && (
            <>
              {!user ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 480,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--primary-light)",
                      color: "var(--text-main)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 16px",
                    }}
                  >
                    <Lock size={22} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Clinical Authentication Required
                  </h2>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.5 }}>
                    Please sign in with a verified clinician or administrator account to access the diagnostic dashboard.
                  </p>
                  <button onClick={openSignInModal} className="btn btn-primary btn-lg">
                    <span>Sign In to Clinical Portal</span>
                    <ArrowRight size={16} />
                  </button>
                </div>
              ) : isUnverifiedClinician ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 520,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--warning-bg)",
                      color: "var(--warning-text)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 14px",
                    }}
                  >
                    <Clock size={24} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Clinician Account Pending Verification
                  </h2>
                  <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: 16, lineHeight: 1.5 }}>
                    Your clinician account is awaiting administrative verification. Clinical access will be enabled after approval.
                  </p>
                  <div
                    style={{
                      padding: "10px 14px",
                      background: "var(--bg-surface-subtle)",
                      borderRadius: "var(--radius-sm)",
                      border: "1px solid var(--border-color)",
                      fontSize: "0.78rem",
                      color: "var(--text-muted)",
                      maxWidth: 420,
                      margin: "0 auto",
                    }}
                  >
                    Access to patient clinical records and diagnostic decision support requires administrator approval.
                  </div>
                </div>
              ) : isDedicatedResearcher ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 500,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--warning-bg)",
                      color: "var(--warning-text)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 14px",
                    }}
                  >
                    <ShieldAlert size={24} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Clinical Privacy Boundary Enforced
                  </h2>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.5 }}>
                    Clinical dashboard access is reserved for attending practitioners. Please proceed to the research dashboard.
                  </p>
                  <button onClick={() => setActiveTab("research-dashboard")} className="btn btn-primary">
                    <Database size={15} />
                    <span>Return to Research Dashboard</span>
                  </button>
                </div>
              ) : (
                <Dashboard
                  user={user}
                  cases={cases}
                  loading={loading}
                  onOpenNewCase={() => setActiveTab("new-case-upload")}
                  onNavigateTab={(tab) => {
                    setActiveTab(tab);
                    if (tab !== "cases") setSelectedCase(null);
                  }}
                  onSelectCase={(c) => {
                    setSelectedCase(c);
                    setActiveTab("cases");
                  }}
                />
              )}
            </>
          )}

          {/* TAB: CASES (Queue & Workstation) */}
          {activeTab === "cases" && (
            <>
              {!user ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 480,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--primary-light)",
                      color: "var(--text-main)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 16px",
                    }}
                  >
                    <Lock size={22} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Clinical Authentication Required
                  </h2>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.5 }}>
                    Sign in to your practitioner account or patient portal to manage screening cases and diagnostic evaluations.
                  </p>
                  <button onClick={openSignInModal} className="btn btn-primary btn-lg">
                    <span>Sign In to Portal</span>
                    <ArrowRight size={16} />
                  </button>
                </div>
              ) : isUnverifiedClinician ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 520,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--warning-bg)",
                      color: "var(--warning-text)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 14px",
                    }}
                  >
                    <Clock size={24} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Clinician Account Pending Verification
                  </h2>
                  <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: 16, lineHeight: 1.5 }}>
                    Your clinician account is awaiting administrative verification. Clinical access will be enabled after approval.
                  </p>
                </div>
              ) : isDedicatedResearcher ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 500,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--warning-bg)",
                      color: "var(--warning-text)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 14px",
                    }}
                  >
                    <ShieldAlert size={24} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Clinical Privacy Boundary Enforced
                  </h2>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.5 }}>
                    Direct access to patient medical records is restricted to attending clinicians and patients under HIPAA/GDPR clinical data privacy boundaries.
                  </p>
                  <button onClick={() => setActiveTab("research-dashboard")} className="btn btn-primary">
                    <Database size={15} />
                    <span>Return to Research Dashboard</span>
                  </button>
                </div>
              ) : selectedCase ? (
                <CaseDetail
                  currentCase={selectedCase}
                  onBack={() => {
                    setSelectedCase(null);
                    loadCases();
                  }}
                  onCaseUpdated={handleCaseUpdated}
                  user={user}
                />
              ) : (
                <CaseList
                  cases={cases}
                  loading={loading}
                  onSelectCase={(c) => setSelectedCase(c)}
                  onOpenNewCaseWorkflow={() => {
                    setSelectedCase(null);
                    setActiveTab("new-case-upload");
                  }}
                  user={user}
                />
              )}
            </>
          )}

          {/* TAB: COMBINED NEW CASE & IMAGE UPLOAD */}
          {activeTab === "new-case-upload" && (
            <>
              {!user ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 480,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--primary-light)",
                      color: "var(--text-main)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 16px",
                    }}
                  >
                    <Lock size={22} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Sign In to Create Screening Cases
                  </h2>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.5 }}>
                    Please sign in to register screening cases and upload lesion photography for automated quality assessment.
                  </p>
                  <button onClick={openSignInModal} className="btn btn-primary btn-lg">
                    <span>Sign In to Portal</span>
                    <ArrowRight size={16} />
                  </button>
                </div>
              ) : isUnverifiedClinician ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 520,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--warning-bg)",
                      color: "var(--warning-text)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 14px",
                    }}
                  >
                    <Clock size={24} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Clinical Intake Disabled Pending Approval
                  </h2>
                  <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    Your clinician account is awaiting administrative verification. Clinical intake will be enabled after approval.
                  </p>
                </div>
              ) : isDedicatedResearcher ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 500,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--warning-bg)",
                      color: "var(--warning-text)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 14px",
                    }}
                  >
                    <ShieldAlert size={24} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Clinical Intake Restricted
                  </h2>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.5 }}>
                    Patient screening intake and clinical image uploading are reserved for patient self-screening and clinician examination workflows.
                  </p>
                  <button onClick={() => setActiveTab("research-dashboard")} className="btn btn-primary">
                    <Database size={15} />
                    <span>Go to Research Dashboard</span>
                  </button>
                </div>
              ) : (
                <CreateCaseAndUpload
                  onCaseCreatedAndProcessed={handleCaseCreatedAndProcessed}
                  onViewCaseDetails={(newCase) => {
                    setSelectedCase(newCase);
                    setActiveTab("cases");
                  }}
                />
              )}
            </>
          )}

          {/* TAB: RESEARCH DASHBOARD / MODEL EVALUATION / ANONYMIZED DATA */}
          {(activeTab === "research-dashboard" || activeTab === "model-evaluation" || activeTab === "anonymized-cases") && (
            <DatasetView initialSection={activeTab} />
          )}

          {/* TAB: ADMIN MANAGEMENT PANEL */}
          {activeTab === "admin-panel" && (
            <>
              {user?.role !== "admin" ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 480,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--error-bg)",
                      color: "var(--error-text)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 14px",
                    }}
                  >
                    <ShieldAlert size={24} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Administrator Access Required
                  </h2>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.5 }}>
                    Access to the DermAI Administrative Console is strictly restricted to system administrators.
                  </p>
                  <button onClick={() => setActiveTab("dashboard")} className="btn btn-primary">
                    <span>Return to Dashboard</span>
                  </button>
                </div>
              ) : (
                <AdminPanel user={user} onLogout={handleLogout} />
              )}
            </>
          )}

          {/* TAB: ADMIN CLINICIAN VERIFICATIONS */}
          {activeTab === "admin-verifications" && (
            <>
              {user?.role !== "admin" ? (
                <div
                  className="medical-card"
                  style={{
                    padding: "48px 24px",
                    textAlign: "center",
                    maxWidth: 480,
                    margin: "40px auto",
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: "var(--radius-md)",
                      background: "var(--error-bg)",
                      color: "var(--error-text)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 14px",
                    }}
                  >
                    <ShieldAlert size={24} />
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-main)", marginBottom: 8 }}>
                    Administrator Access Required
                  </h2>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.5 }}>
                    Clinician account verification and registration oversight are strictly restricted to system administrators.
                  </p>
                  <button onClick={() => setActiveTab("dashboard")} className="btn btn-primary">
                    <span>Return to Dashboard</span>
                  </button>
                </div>
              ) : (
                <AdminCliniciansView />
              )}
            </>
          )}
        </div>
      </main>

      {/* Clinical Footer */}
      <footer className="app-footer">
        <div className="app-container" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 12,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: "var(--radius-xs)",
                  background: "var(--primary-accent)",
                  color: "var(--text-main)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Activity size={13} strokeWidth={2.4} />
              </div>
              <strong style={{ color: "var(--text-main)" }}>DermAI</strong>
              <span>• Deep Learning Skin Lesion Screening & Decision Support</span>
            </div>

            <div style={{ display: "flex", gap: 12, alignItems: "center", fontSize: "0.75rem", color: "var(--text-muted)" }}>
              <span>HAM10000 MobileNetV2 & ViT</span>
              <span>•</span>
              <span>Role-Differentiated Workflows</span>
            </div>
          </div>

          <div
            style={{
              paddingTop: 8,
              borderTop: "1px solid var(--border-subtle)",
              fontSize: "0.74rem",
              color: "var(--text-muted)",
              lineHeight: 1.5,
            }}
          >
            <strong style={{ color: "var(--text-secondary)" }}>Medical Disclaimer: </strong>
            DermAI is an investigational deep-learning clinical decision-support platform powered by MobileNetV2 and Vision Transformer models. Softmax scores represent relative model confidence across 7 skin lesion categories, not calibrated clinical likelihoods. DermAI does not provide autonomous definitive diagnoses or replace in-person dermatological examination, dermoscopy, and histopathological biopsy.
          </div>
        </div>
      </footer>

      {/* Authentication Modal */}
      <LoginModal
        isOpen={isLoginOpen}
        onClose={() => setIsLoginOpen(false)}
        onAuthSuccess={handleAuthSuccess}
        initialRegister={isRegisterMode}
      />
    </div>
  );
}
