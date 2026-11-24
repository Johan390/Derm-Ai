// Dynamic Backend Configuration supporting Environment Variables
export const BACKEND_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE_URL) ||
  "";

const API_BASE = `/api/v1`;

export function getAuthToken() {
  return localStorage.getItem("dermai_token");
}

export function setAuthToken(token) {
  if (token) {
    localStorage.setItem("dermai_token", token);
  } else {
    localStorage.removeItem("dermai_token");
  }
}

export function getCurrentUser() {
  const user = localStorage.getItem("dermai_user");
  return user ? JSON.parse(user) : null;
}

export function setCurrentUser(user) {
  if (user) {
    localStorage.setItem("dermai_user", JSON.stringify(user));
  } else {
    localStorage.removeItem("dermai_user");
  }
}

export async function apiRequest(endpoint, options = {}) {
  const token = getAuthToken();
  const headers = options.headers ? { ...options.headers } : {};

  if (token && !headers["Authorization"]) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // If not FormData, default to application/json
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    // If unauthorized, clean up expired token
    if (response.status === 401) {
      setAuthToken(null);
      setCurrentUser(null);
    }
    const errorData = await response.json().catch(() => ({ detail: "Unknown server error" }));
    throw new Error(errorData.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export const api = {
  // Auth
  login: async (username, password) => {
    const data = await apiRequest("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    const userPayload = {
      ...data,
      id: data.id || data.user_id,
      user_id: data.user_id || data.id,
    };
    setAuthToken(userPayload.access_token);
    setCurrentUser(userPayload);
    return userPayload;
  },

  register: async (userData) => {
    const data = await apiRequest("/auth/register", {
      method: "POST",
      body: JSON.stringify(userData),
    });
    const userPayload = {
      ...data,
      id: data.id || data.user_id,
      user_id: data.user_id || data.id,
    };
    setAuthToken(userPayload.access_token);
    setCurrentUser(userPayload);
    return userPayload;
  },

  logout: () => {
    setAuthToken(null);
    setCurrentUser(null);
  },

  getMe: async () => {
    return apiRequest("/auth/me");
  },

  // Cases
  getCases: async (statusFilter = "", limit = 500) => {
    const params = new URLSearchParams();
    if (statusFilter) params.append("status_filter", statusFilter);
    if (limit) params.append("limit", limit);
    const queryString = params.toString() ? `?${params.toString()}` : "";
    return apiRequest(`/cases/${queryString}`);
  },

  getCase: async (id) => {
    return apiRequest(`/cases/${id}`);
  },

  createCase: async (caseData) => {
    return apiRequest("/cases/", {
      method: "POST",
      body: JSON.stringify(caseData),
    });
  },

  // Image upload
  uploadCaseImage: async (caseId, file) => {
    const formData = new FormData();
    formData.append("file", file);
    return apiRequest(`/cases/${caseId}/upload-image`, {
      method: "POST",
      body: formData,
    });
  },

  // Milestone 2 - Deep Learning Model Predictions
  predictCase: async (caseId, model = "mobilenet") => {
    return apiRequest(`/cases/${caseId}/predict?model=${encodeURIComponent(model)}`, {
      method: "POST",
    });
  },

  getCasePredictions: async (caseId) => {
    return apiRequest(`/cases/${caseId}/predictions`);
  },

  // Dataset & EDA
  getDatasetInfo: async () => {
    return apiRequest("/dataset/info");
  },

  getEdaSummary: async () => {
    return apiRequest("/dataset/eda-summary");
  },

  // Doctor Case Reviews & Comments
  addCaseReview: async (caseId, reviewData) => {
    return apiRequest(`/cases/${caseId}/reviews`, {
      method: "POST",
      body: JSON.stringify(reviewData),
    });
  },

  getCaseReviews: async (caseId) => {
    return apiRequest(`/cases/${caseId}/reviews`);
  },

  updateReviewStatus: async (caseId, reviewStatus) => {
    return apiRequest(`/cases/${caseId}/review-status`, {
      method: "PATCH",
      body: JSON.stringify({ review_status: reviewStatus }),
    });
  },

  assignCaseDoctor: async (caseId, doctorId = null) => {
    return apiRequest(`/cases/${caseId}/assign`, {
      method: "PUT",
      body: JSON.stringify({ doctor_id: doctorId }),
    });
  },

  releaseCase: async (caseId) => {
    return apiRequest(`/cases/${caseId}/release`, {
      method: "POST",
    });
  },

  deleteCase: async (caseId) => {
    return apiRequest(`/cases/${caseId}`, {
      method: "DELETE",
    });
  },

  // Admin Clinician Verification
  getPendingClinicians: async () => {
    return apiRequest("/admin/clinicians/pending");
  },

  verifyClinician: async (clinicianId) => {
    return apiRequest(`/admin/clinicians/${clinicianId}/verify`, {
      method: "PATCH",
    });
  },

  rejectClinician: async (clinicianId) => {
    return apiRequest(`/admin/clinicians/${clinicianId}/reject`, {
      method: "PATCH",
    });
  },

  // Admin Panel Management APIs
  getAdminStats: async () => {
    return apiRequest("/admin/stats");
  },

  getAdminPatients: async (search = "") => {
    const q = search ? `?search=${encodeURIComponent(search)}` : "";
    return apiRequest(`/admin/patients${q}`);
  },

  getAdminDoctors: async (search = "") => {
    const q = search ? `?search=${encodeURIComponent(search)}` : "";
    return apiRequest(`/admin/doctors${q}`);
  },

  updateAdminUser: async (userId, userData) => {
    return apiRequest(`/admin/users/${userId}`, {
      method: "PUT",
      body: JSON.stringify(userData),
    });
  },

  deleteAdminUser: async (userId) => {
    return apiRequest(`/admin/users/${userId}`, {
      method: "DELETE",
    });
  },

  getAdminAnalyses: async (limit = 50) => {
    return apiRequest(`/admin/analyses?limit=${limit}`);
  },

  getAdminActivityLogs: async (limit = 50) => {
    return apiRequest(`/admin/activity-logs?limit=${limit}`);
  },

  getAdminSettings: async () => {
    return apiRequest("/admin/settings");
  },

  // Research & Model Evaluation Endpoints
  getResearchMetrics: async () => {
    return apiRequest("/research/metrics");
  },

  getImageQualityStats: async () => {
    return apiRequest("/research/image-quality-stats");
  },

  getAnonymizedCases: async (limit = 100) => {
    return apiRequest(`/research/anonymized-cases?limit=${limit}`);
  },

  exportAnonymizedData: async (format = "json") => {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch(`${BACKEND_URL}/api/v1/research/export-anonymized?format=${format}`, {
      headers,
    });
    if (!response.ok) {
      throw new Error(`Export failed with status ${response.status}`);
    }
    if (format === "csv") {
      return response.blob();
    }
    return response.json();
  },

  // Milestone 4 - Notifications System
  getNotifications: async (unreadOnly = false) => {
    return apiRequest(`/notifications?unread_only=${unreadOnly}`);
  },

  markNotificationRead: async (notificationId) => {
    return apiRequest(`/notifications/${notificationId}/read`, {
      method: "PATCH",
    });
  },

  markAllNotificationsRead: async () => {
    return apiRequest("/notifications/read-all", {
      method: "POST",
    });
  },

  // Milestone 4 - Clinical Reports & Data Exports
  downloadCasePdf: async (caseId, caseCode = "Report") => {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch(`${API_BASE}/cases/${caseId}/report/pdf`, { headers });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "PDF generation failed" }));
      throw new Error(err.detail || "Failed to download PDF report");
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `DermAI_Report_${caseCode}.pdf`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  },

  exportCaseCsv: async (caseId, caseCode = "Case") => {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch(`${API_BASE}/cases/${caseId}/export/csv`, { headers });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "CSV export failed" }));
      throw new Error(err.detail || "Failed to export case CSV");
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `DermAI_Case_${caseCode}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  },

  exportCaseJson: async (caseId, caseCode = "Case") => {
    const data = await apiRequest(`/cases/${caseId}/export/json`);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `DermAI_Case_${caseCode}.json`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  },

  exportAllCasesCsv: async () => {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch(`${API_BASE}/cases/export/csv`, { headers });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "Cohort export failed" }));
      throw new Error(err.detail || "Failed to export cases CSV");
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "DermAI_Clinical_Cases_Cohort.csv";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  },

  exportAllCasesJson: async () => {
    const data = await apiRequest("/cases/export/json");
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "DermAI_Clinical_Cases_Cohort.json";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  },

  // Health and System Telemetry
  getHealth: async () => {
    return apiRequest("/health");
  },

  // Independent Image Quality Assessment
  assessImageQuality: async (formData) => {
    return apiRequest("/image-quality/assess", {
      method: "POST",
      body: formData,
    });
  },

  // Profile and Password Management
  updateProfile: async (profileData) => {
    return apiRequest("/auth/profile", {
      method: "PUT",
      body: JSON.stringify(profileData),
    });
  },

  changePassword: async (passwordData) => {
    return apiRequest("/auth/change-password", {
      method: "POST",
      body: JSON.stringify(passwordData),
    });
  },

  // Longitudinal Case Analysis Comparison
  compareCaseAnalyses: async (caseId, earlierId = null, newerId = null) => {
    const params = new URLSearchParams();
    if (earlierId) params.append("earlier_id", earlierId);
    if (newerId) params.append("newer_id", newerId);
    const queryString = params.toString() ? `?${params.toString()}` : "";
    return apiRequest(`/cases/${caseId}/compare${queryString}`);
  },

  // Model Registry, Training History & Rollback
  getModelRegistry: async () => {
    return apiRequest("/models/registry");
  },

  getModelTrainingHistory: async () => {
    return apiRequest("/models/training-history");
  },

  setActiveModel: async (rollbackData) => {
    return apiRequest("/models/active", {
      method: "POST",
      body: JSON.stringify(rollbackData),
    });
  },

  // Follow-Up Reminders
  createFollowUpReminder: async (caseId, reminderData) => {
    return apiRequest(`/cases/${caseId}/reminders`, {
      method: "POST",
      body: JSON.stringify(reminderData),
    });
  },

  getCaseReminders: async (caseId) => {
    return apiRequest(`/cases/${caseId}/reminders`);
  },

  updateReminderStatus: async (reminderId, status) => {
    return apiRequest(`/reminders/${reminderId}/status?status_update=${status}`, {
      method: "PATCH",
    });
  },

  getImageUrl: (relativeUrl) => {
    if (!relativeUrl) return "";
    const token = getAuthToken();
    const separator = relativeUrl.includes("?") ? "&" : "?";
    const authParam = token ? `${separator}token=${encodeURIComponent(token)}` : "";
    return `${BACKEND_URL}${relativeUrl}${authParam}`;
  },
};
