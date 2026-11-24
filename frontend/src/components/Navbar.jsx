import React, { useState, useEffect } from "react";
import {
  Activity,
  FolderOpen,
  PlusCircle,
  Database,
  LogOut,
  User,
  Shield,
  Menu,
  X,
  Bell,
  CheckCheck,
  Clock,
  BookOpen,
  LayoutDashboard
} from "lucide-react";
import { api } from "../api";

export default function Navbar({
  activeTab,
  setActiveTab,
  user,
  onOpenLogin,
  onOpenRegister,
  onLogout,
  onSelectCase
}) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);

  const fetchNotifications = async () => {
    if (!user) return;
    try {
      const data = await api.getNotifications();
      setNotifications(data.notifications || []);
      setUnreadCount(data.unread_count || 0);
    } catch (e) {
      console.warn("Could not fetch notifications:", e);
    }
  };

  useEffect(() => {
    if (user) {
      fetchNotifications();
      const interval = setInterval(fetchNotifications, 15000);
      return () => clearInterval(interval);
    } else {
      setNotifications([]);
      setUnreadCount(0);
    }
  }, [user]);

  const handleMarkRead = async (id, caseId) => {
    try {
      await api.markNotificationRead(id);
      setNotifications((prev) =>
        prev.map((n) => (n.id === id ? { ...n, is_read: true } : n))
      );
      setUnreadCount((c) => Math.max(0, c - 1));
      if (caseId && onSelectCase) {
        onSelectCase(caseId);
        setShowNotifications(false);
      }
    } catch (e) {
      console.warn("Failed to mark notification read:", e);
    }
  };

  const handleMarkAllRead = async () => {
    try {
      await api.markAllNotificationsRead();
      setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
      setUnreadCount(0);
    } catch (e) {
      console.warn("Failed to mark all notifications read:", e);
    }
  };

  const handleNavClick = (tab) => {
    setActiveTab(tab);
    setMobileMenuOpen(false);
  };

  let navItems = [];
  if (!user) {
    navItems = [
      { id: "landing", label: "Overview", icon: Activity },
      { id: "guidelines", label: "Clinical Guidelines", icon: BookOpen },
    ];
  } else if (user.role === "patient") {
    navItems = [
      { id: "cases", label: "My Screening Cases", icon: FolderOpen },
      { id: "new-case-upload", label: "New Screening Case", icon: PlusCircle },
      { id: "guidelines", label: "Guidelines", icon: BookOpen },
    ];
  } else if (user.role === "clinician") {
    navItems = [
      { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
      { id: "cases", label: "Case Queue", icon: FolderOpen },
      { id: "new-case-upload", label: "Intake Case", icon: PlusCircle },
      { id: "guidelines", label: "Guidelines", icon: BookOpen },
    ];
  } else if (user.role === "admin") {
    navItems = [
      { id: "admin-panel", label: "Admin Panel", icon: Shield },
      { id: "dashboard", label: "Clinical Dashboard", icon: LayoutDashboard },
      { id: "cases", label: "Clinical Cases", icon: FolderOpen },
      { id: "new-case-upload", label: "Intake Case", icon: PlusCircle },
      { id: "guidelines", label: "Guidelines", icon: BookOpen },
    ];
  }

  const getRoleBadgeLabel = (role) => {
    switch (role) {
      case "clinician":
        return "Clinician (MD)";
      case "patient":
        return "Patient Portal";
      case "admin":
        return "System Admin";
      default:
        return role || "User";
    }
  };

  return (
    <header className="navbar-header">
      <div className="app-container navbar-container">
        {/* Brand */}
        <div
          onClick={() => handleNavClick(user ? (user.role === "admin" ? "admin-panel" : user.role === "clinician" ? "dashboard" : "cases") : "landing")}
          className="navbar-brand"
        >
          <div className="navbar-brand-logo">
            <Activity size={20} strokeWidth={2.4} />
          </div>
          <div>
            <div className="navbar-brand-text">
              <span>DermAI</span>
              <span className="badge badge-teal" style={{ fontSize: "0.68rem", padding: "1px 6px" }}>
                Clinical Decision Support
              </span>
            </div>
            <p className="navbar-brand-subtitle">
              Deep Learning Skin Lesion Screening
            </p>
          </div>
        </div>

        {/* Desktop Navigation Links */}
        <nav className="navbar-nav desktop-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNavClick(item.id)}
                className={`navbar-item ${isActive ? "navbar-item-active" : ""}`}
              >
                <Icon size={16} strokeWidth={isActive ? 2.2 : 1.8} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Desktop Actions & User Controls */}
        <div className="navbar-actions desktop-auth">
          {user ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              {/* Notifications Popover */}
              <div style={{ position: "relative" }}>
                <button
                  onClick={() => {
                    setShowNotifications(!showNotifications);
                    if (!showNotifications) fetchNotifications();
                  }}
                  className="btn btn-secondary btn-sm"
                  style={{
                    position: "relative",
                    padding: "7px 10px",
                    background: showNotifications ? "var(--primary-light)" : "#FFFFFF",
                    borderColor: showNotifications ? "var(--border-focus)" : "var(--border-color)",
                  }}
                  title="Notifications"
                  aria-label="View notifications"
                >
                  <Bell size={15} />
                  {unreadCount > 0 && (
                    <span
                      style={{
                        position: "absolute",
                        top: -4,
                        right: -4,
                        background: "var(--error)",
                        color: "#FFFFFF",
                        borderRadius: "50%",
                        fontSize: "0.62rem",
                        fontWeight: 700,
                        width: 16,
                        height: 16,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        border: "2px solid #FFFFFF",
                      }}
                    >
                      {unreadCount > 9 ? "9+" : unreadCount}
                    </span>
                  )}
                </button>

                {showNotifications && (
                  <div className="notification-popover">
                    <div className="notification-header">
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-main)" }}>
                          Notifications
                        </span>
                        {unreadCount > 0 && (
                          <span className="badge badge-teal" style={{ fontSize: "0.68rem" }}>
                            {unreadCount} new
                          </span>
                        )}
                      </div>
                      {unreadCount > 0 && (
                        <button
                          onClick={handleMarkAllRead}
                          className="btn btn-ghost btn-sm"
                          style={{ fontSize: "0.72rem", padding: "2px 6px" }}
                        >
                          <CheckCheck size={13} />
                          <span>Mark all read</span>
                        </button>
                      )}
                    </div>

                    <div style={{ overflowY: "auto", maxHeight: 360 }}>
                      {notifications.length === 0 ? (
                        <div style={{ padding: "28px 16px", textAlign: "center", color: "var(--text-muted)" }}>
                          <Clock size={22} style={{ margin: "0 auto 6px", opacity: 0.5 }} />
                          <p style={{ fontSize: "0.82rem" }}>No notifications at this time.</p>
                        </div>
                      ) : (
                        notifications.map((n) => (
                          <div
                            key={n.id}
                            onClick={() => handleMarkRead(n.id, n.case_id)}
                            style={{
                              padding: "10px 14px",
                              borderBottom: "1px solid var(--border-subtle)",
                              background: n.is_read ? "#FFFFFF" : "var(--primary-subtle)",
                              cursor: "pointer",
                              transition: "background 0.15s ease",
                            }}
                          >
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
                              <strong style={{ fontSize: "0.82rem", color: "var(--text-main)" }}>
                                {n.title}
                              </strong>
                              {!n.is_read && (
                                <span
                                  style={{
                                    width: 7,
                                    height: 7,
                                    borderRadius: "50%",
                                    background: "var(--primary-hover)",
                                    flexShrink: 0,
                                    marginTop: 4,
                                  }}
                                />
                              )}
                            </div>
                            <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 2, lineHeight: 1.4 }}>
                              {n.message}
                            </p>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 4 }}>
                              {new Date(n.created_at).toLocaleDateString()} {new Date(n.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* User Profile Pill */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  background: "var(--bg-surface-subtle)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "var(--radius-pill)",
                  padding: "4px 12px 4px 6px",
                }}
              >
                <div
                  style={{
                    width: 26,
                    height: 26,
                    borderRadius: "50%",
                    background: "var(--primary-accent)",
                    color: "var(--text-main)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 700,
                    fontSize: "0.75rem",
                  }}
                >
                  {(user.full_name || user.username || "U")[0].toUpperCase()}
                </div>
                <div style={{ lineHeight: 1.2 }}>
                  <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-main)" }}>
                    {user.full_name || user.username}
                  </div>
                  <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                    {getRoleBadgeLabel(user.role)}
                  </div>
                </div>
              </div>

              {/* Sign Out Action */}
              <button
                onClick={onLogout}
                className="btn btn-secondary btn-sm"
                title="Sign out of DermAI"
              >
                <LogOut size={14} />
                <span>Sign Out</span>
              </button>
            </div>
          ) : (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <button
                onClick={onOpenLogin}
                className="btn btn-secondary btn-sm"
              >
                <User size={14} />
                <span>Sign In</span>
              </button>
              {onOpenRegister && (
                <button
                  onClick={onOpenRegister}
                  className="btn btn-primary btn-sm"
                >
                  <span>Register</span>
                </button>
              )}
            </div>
          )}
        </div>

        {/* Mobile Hamburger Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="btn btn-secondary btn-sm mobile-toggle"
          aria-label="Toggle navigation menu"
        >
          {mobileMenuOpen ? <X size={18} /> : <Menu size={18} />}
        </button>
      </div>

      {/* Mobile Off-Canvas Drawer */}
      {mobileMenuOpen && (
        <>
          <div className="mobile-nav-overlay" onClick={() => setMobileMenuOpen(false)} />
          <div className="mobile-nav-drawer">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div className="navbar-brand-logo" style={{ width: 30, height: 30 }}>
                  <Activity size={16} />
                </div>
                <span style={{ fontWeight: 800, fontSize: "1.1rem" }}>DermAI</span>
              </div>
              <button
                onClick={() => setMobileMenuOpen(false)}
                className="btn btn-ghost btn-sm"
                style={{ padding: 4 }}
              >
                <X size={18} />
              </button>
            </div>

            {user && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 12px",
                  background: "var(--primary-light)",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--primary-light-border)",
                  marginBottom: 16,
                }}
              >
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    background: "var(--primary-accent)",
                    color: "var(--text-main)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 700,
                  }}
                >
                  {(user.full_name || user.username || "U")[0].toUpperCase()}
                </div>
                <div>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-main)" }}>
                    {user.full_name || user.username}
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>
                    {getRoleBadgeLabel(user.role)}
                  </div>
                </div>
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavClick(item.id)}
                    className={`navbar-item ${isActive ? "navbar-item-active" : ""}`}
                    style={{ width: "100%", justifyContent: "flex-start", padding: "10px 14px" }}
                  >
                    <Icon size={18} />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>

            <div style={{ marginTop: "auto", paddingTop: 16, borderTop: "1px solid var(--border-color)" }}>
              {user ? (
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    onLogout();
                  }}
                  className="btn btn-secondary"
                  style={{ width: "100%", justifyContent: "center" }}
                >
                  <LogOut size={16} />
                  <span>Sign Out</span>
                </button>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <button
                    onClick={() => {
                      setMobileMenuOpen(false);
                      onOpenLogin();
                    }}
                    className="btn btn-secondary"
                    style={{ width: "100%", justifyContent: "center" }}
                  >
                    <User size={16} />
                    <span>Sign In</span>
                  </button>
                  {onOpenRegister && (
                    <button
                      onClick={() => {
                        setMobileMenuOpen(false);
                        onOpenRegister();
                      }}
                      className="btn btn-primary"
                      style={{ width: "100%", justifyContent: "center" }}
                    >
                      <span>Register Practitioner</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </header>
  );
}
