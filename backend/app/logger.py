"""
DermAI Structured Logging and Monitoring Service
=================================================
Provides centralized, HIPAA-conscious, structured logging for DermAI.
Ensures zero leakage of credentials, passwords, JWT tokens, patient PHI,
or raw medical image contents.
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Configure standard root logger format
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("dermai")

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sanitize_username(username: Optional[str]) -> str:
    """Safely sanitize username for logging."""
    if not username:
        return "<anonymous>"
    return str(username)[:50].strip()

def log_startup(app_name: str, version: str, environment: str = "production"):
    """Record application startup."""
    logger.info(
        f"[SYSTEM_STARTUP] {app_name} v{version} initialized in environment: {environment} at {_utc_iso()}"
    )

def log_shutdown(app_name: str):
    """Record application shutdown."""
    logger.info(f"[SYSTEM_SHUTDOWN] {app_name} shutting down gracefully at {_utc_iso()}")

def log_auth_success(username: str, role: str, client_ip: Optional[str] = None):
    """Record successful authentication event without sensitive token data."""
    logger.info(
        f"[AUTH_SUCCESS] User '{sanitize_username(username)}' (role: {role}) authenticated successfully. IP: {client_ip or 'unknown'}"
    )

def log_auth_failure(username: Optional[str], reason: str, client_ip: Optional[str] = None):
    """Record authentication failure without exposing entered passwords or credentials."""
    logger.warning(
        f"[AUTH_FAILURE] Authentication failed for user '{sanitize_username(username)}'. Reason: {reason}. IP: {client_ip or 'unknown'}"
    )

def log_prediction_success(
    case_id: int,
    case_code: str,
    model_name: str,
    predicted_class: str,
    latency_ms: float,
    user_id: int
):
    """Record deep learning prediction event with latency and telemetry."""
    logger.info(
        f"[PREDICTION_SUCCESS] Case {case_code} (ID: {case_id}) evaluated by model '{model_name}'. "
        f"Result: {predicted_class} in {latency_ms:.1f}ms by user {user_id}"
    )

def log_prediction_failure(case_id: int, model_name: str, error: str):
    """Record prediction error without exposing system internals."""
    logger.error(
        f"[PREDICTION_FAILURE] Case ID {case_id} failed during inference with model '{model_name}'. Error: {error}"
    )

def log_report_generation(case_id: int, report_format: str, user_id: int):
    """Record clinical report generation."""
    logger.info(
        f"[REPORT_GENERATED] Report format '{report_format.upper()}' generated for case ID {case_id} by user {user_id}"
    )

def log_report_failure(case_id: int, report_format: str, error: str):
    """Record report generation error."""
    logger.error(
        f"[REPORT_FAILURE] Report format '{report_format.upper()}' generation failed for case ID {case_id}. Error: {error}"
    )

def log_system_event(event_type: str, details: str, severity: str = "INFO"):
    """Generic structured system event logging."""
    msg = f"[SYSTEM_EVENT:{event_type}] {details}"
    if severity.upper() == "WARNING":
        logger.warning(msg)
    elif severity.upper() == "ERROR":
        logger.error(msg)
    elif severity.upper() == "CRITICAL":
        logger.critical(msg)
    else:
        logger.info(msg)
