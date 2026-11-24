import os
from dataclasses import dataclass, field
from typing import Set, Tuple, List

DEFAULT_DEV_SECRET = "dermai-dev-jwt-super-secret-key-2026-springboard"

@dataclass
class Settings:
    PROJECT_NAME: str = "DermAI Clinical Screening API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    
    # Security & JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", DEFAULT_DEV_SECRET)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # CORS Origins (Explicit list, no wildcards with credentials)
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: [
        origin.strip() for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
        ).split(",") if origin.strip()
    ])
    
    # Database
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dermai.db')}"
    )
    
    # Upload Directories
        # Upload Directories
    UPLOAD_DIR: str = os.getenv(
        "UPLOAD_DIR",
        os.path.join(BASE_DIR, "uploads")
    )
    ORIGINAL_DIR: str = os.path.join(UPLOAD_DIR, "original")
    PREVIEW_DIR: str = os.path.join(UPLOAD_DIR, "preview")
    PREPROCESSED_DIR: str = os.path.join(UPLOAD_DIR, "preprocessed")
    
    # Upload Constraints
    MAX_FILE_SIZE_MB: int = 15
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {"jpg", "jpeg", "png", "webp"})
    TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)
    
    # Configurable Image Quality Assessment (IQA) Engineering Heuristic Thresholds
    # Calibrated against empirical HAM10000 dermoscopy distribution (Median blur: 50.2, contrast: 24.8)
    IQA_BLUR_THRESHOLD_POOR: float = 20.0
    IQA_BLUR_THRESHOLD_ACCEPTABLE: float = 40.0
    IQA_BRIGHTNESS_MIN: float = 40.0
    IQA_BRIGHTNESS_MAX: float = 225.0
    IQA_CONTRAST_MIN: float = 12.0
    IQA_MIN_DIMENSION: int = 150
    
    # Demo Data Configuration
    SEED_DEMO_USERS: bool = field(default_factory=lambda: (
        os.getenv("SEED_DEMO_USERS", "true" if os.getenv("ENVIRONMENT", "development") != "production" else "false").lower() == "true"
    ))

    def __post_init__(self):
        # Fail safely if in production mode and secret key is unset or using development fallback
        if self.ENVIRONMENT == "production":
            if not self.SECRET_KEY or self.SECRET_KEY == DEFAULT_DEV_SECRET:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: "
                    "SECRET_KEY environment variable must be explicitly configured with a strong secret in production."
                )

settings = Settings()

# Ensure directories exist
for path in [settings.UPLOAD_DIR, settings.ORIGINAL_DIR, settings.PREVIEW_DIR, settings.PREPROCESSED_DIR]:
    os.makedirs(path, exist_ok=True)
