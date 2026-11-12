import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    _secret     = os.environ.get('SECRET_KEY')     or 'shieldvision-dev-secret-key-change-in-prod!!'
    _jwt_secret = os.environ.get('JWT_SECRET_KEY') or 'shieldvision-jwt-secret-key-change-in-prod!!'

    SECRET_KEY     = _secret
    JWT_SECRET_KEY = _jwt_secret

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'postgresql://postgres:password@localhost:5432/shieldvision'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)

    # CSV upload limits
    CSV_MAX_BYTES = 1 * 1024 * 1024        # 1 MB max CSV size
    CSV_MAX_ROWS = 500                      # max users per import
    ALLOWED_CSV_EXTENSIONS = {'csv'}

    # Rate limiting
    RATELIMIT_DEFAULT = '1000 per day;200 per hour;60 per minute'
    RATELIMIT_STORAGE_URI = 'memory://'    # swap to redis:// in production

    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MODELS_FOLDER = os.path.join(os.path.dirname(__file__), 'models')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB (video uploads)
