import React from "react";
import { Activity, LogIn, UserPlus } from "lucide-react";

export default function LandingPage({ onGetStarted, onRegister }) {
  const handleSignIn = () => {
    if (typeof onGetStarted === "function") {
      onGetStarted();
      setTimeout(() => {
        const tabs = Array.from(document.querySelectorAll(".modal-content button"));
        const signinTab = tabs.find((b) => b.textContent && b.textContent.trim() === "Sign In");
        if (signinTab) signinTab.click();
      }, 30);
    } else {
      const signInBtn = Array.from(document.querySelectorAll("header button, .navbar-header button"))
        .find((b) => b.textContent && b.textContent.includes("Sign In"));
      if (signInBtn) signInBtn.click();
    }
  };

  const handleRegister = () => {
    if (typeof onRegister === "function") {
      onRegister();
      return;
    }
    if (typeof onGetStarted === "function") {
      onGetStarted();
      setTimeout(() => {
        const tabs = Array.from(document.querySelectorAll(".modal-content button"));
        const regTab = tabs.find((b) => b.textContent && b.textContent.includes("Register"));
        if (regTab) regTab.click();
      }, 30);
      return;
    }
    const registerBtn = Array.from(document.querySelectorAll("header button, .navbar-header button"))
      .find((b) => b.textContent && b.textContent.includes("Register"));
    if (registerBtn) {
      registerBtn.click();
      setTimeout(() => {
        const tabs = Array.from(document.querySelectorAll(".modal-content button"));
        const regTab = tabs.find((b) => b.textContent && b.textContent.includes("Register"));
        if (regTab) regTab.click();
      }, 30);
    }
  };

  return (
    <div className="dermai-welcome-screen">
      <div className="dermai-welcome-card">
        {/* DermAI Branding */}
        <div className="dermai-welcome-brand">
          <div className="dermai-welcome-logo">
            <Activity size={32} strokeWidth={2.4} />
          </div>
          <h1 className="dermai-welcome-title">DermAI</h1>
        </div>

        {/* Action Buttons: Sign In and Register */}
        <div className="dermai-welcome-actions">
          <button
            type="button"
            onClick={handleSignIn}
            className="btn btn-primary dermai-btn-signin"
            id="welcome-signin-btn"
          >
            <LogIn size={18} />
            <span>Sign In</span>
          </button>

          <button
            type="button"
            onClick={handleRegister}
            className="btn btn-secondary dermai-btn-register"
            id="welcome-register-btn"
          >
            <UserPlus size={18} />
            <span>Register</span>
          </button>
        </div>
      </div>

      <style>{`
        .dermai-welcome-screen {
          position: fixed;
          inset: 0;
          z-index: 500;
          background: #F8FAFC;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
        }

        .dermai-welcome-card {
          width: 100%;
          max-width: 420px;
          background: #FFFFFF;
          border: 1px solid var(--border-color, #E2E8F0);
          border-radius: var(--radius-xl, 16px);
          box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.08), 0 8px 10px -6px rgba(15, 23, 42, 0.04);
          padding: 48px 36px;
          text-align: center;
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .dermai-welcome-brand {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin-bottom: 32px;
        }

        .dermai-welcome-logo {
          width: 64px;
          height: 64px;
          border-radius: var(--radius-lg, 14px);
          background: var(--primary-accent, #0D9488);
          color: #FFFFFF;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 16px;
          box-shadow: 0 4px 12px rgba(13, 148, 136, 0.25);
        }

        .dermai-welcome-title {
          font-size: 2.25rem;
          font-weight: 800;
          color: var(--text-main, #0F172A);
          letter-spacing: -0.03em;
          line-height: 1.1;
          margin: 0;
        }

        .dermai-welcome-actions {
          display: flex;
          flex-direction: column;
          gap: 12px;
          width: 100%;
        }

        .dermai-btn-signin,
        .dermai-btn-register {
          width: 100%;
          height: 48px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          font-size: 0.95rem;
          font-weight: 600;
          border-radius: var(--radius-md, 8px);
          cursor: pointer;
          transition: all 0.15s ease-in-out;
        }

        .dermai-btn-signin {
          background: var(--primary-accent, #0D9488);
          color: #FFFFFF;
          border: 1px solid transparent;
        }

        .dermai-btn-signin:hover {
          background: var(--primary-hover, #0F766E);
        }

        .dermai-btn-register {
          background: #FFFFFF;
          color: var(--text-main, #0F172A);
          border: 1px solid var(--border-color, #E2E8F0);
        }

        .dermai-btn-register:hover {
          background: var(--bg-surface-subtle, #F1F5F9);
          border-color: var(--border-color-hover, #CBD5E1);
        }
      `}</style>
    </div>
  );
}
