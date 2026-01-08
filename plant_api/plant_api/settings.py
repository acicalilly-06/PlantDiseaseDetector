"""
Django settings for plant_api project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# ===========================================
# 🔧 BASE CONFIG
# ===========================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "django-insecure-dev-key")

# ✅ Always True for offline local development
DEBUG = True

# ✅ Allow all local connections (for emulator, LAN, etc.)
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "0.0.0.0"]

# ===========================================
# 📦 INSTALLED APPS
# ===========================================

INSTALLED_APPS = [
    # Core Django apps
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Third-party
    "rest_framework",
    "corsheaders",

    # Local apps
    "api",
]

# ===========================================
# ⚙️ MIDDLEWARE
# ===========================================

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",  # must be first for CORS
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "plant_api.urls"

# ===========================================
# 🎨 TEMPLATES
# ===========================================

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "plant_api.wsgi.application"

# ===========================================
# 🗄️ DATABASE (SQLite only — offline)
# ===========================================

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ===========================================
# 🔒 PASSWORD VALIDATION
# ===========================================

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ===========================================
# 🌍 INTERNATIONALIZATION
# ===========================================

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Kolkata")
USE_I18N = True
USE_TZ = True

# ===========================================
# 🖼️ STATIC & MEDIA (Local storage only)
# ===========================================

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ✅ Always use local file storage — no Cloudinary or external service
DEFAULT_FILE_STORAGE = "django.core.files.storage.FileSystemStorage"

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
}

CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {
        "handlers": ["console"],
        "level": "DEBUG",  # changed to DEBUG for local visibility
    },
}
