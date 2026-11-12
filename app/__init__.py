from flask import Flask, redirect, url_for
from app.models.database import db
from config import Config
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=[])

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    db.init_app(app)
    limiter.init_app(app)

    from app.api.auth import auth_bp
    from app.api.cameras import cameras_bp
    from app.api.alerts import alerts_bp
    from app.api.pages import admin_bp, user_bp
    from app.api.gpu import gpu_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(cameras_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(gpu_bp)

    @app.route('/')
    def index():
        from flask import session
        if 'user_id' in session:
            if session.get('role') == 'admin':
                return redirect(url_for('admin.dashboard'))
            return redirect(url_for('user.dashboard'))
        return redirect(url_for('auth.login_page'))

    with app.app_context():
        db.create_all()
        _seed_admin(app)
        _autostart_streams(app)

    return app


def _seed_admin(app):
    from app.models.database import User
    if not User.query.filter_by(role='admin').first():
        admin = User(username='admin', email='admin@shieldvision.local', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("[Setup] Default admin created: admin / admin123")


def _autostart_streams(app):
    """Start AI inference streams for all active cameras on server startup."""
    from app.models.database import Camera
    from app.services.stream_service import stream_manager
    from app.services.alert_service import broadcaster
    from app.models.database import Alert
    from datetime import datetime, timezone

    cameras = Camera.query.filter_by(is_active=True).all()
    for cam in cameras:
        model_types = [m.model_type for m in cam.model_mappings if m.is_enabled]
        roi_map     = {m.model_type: m.roi for m in cam.model_mappings if m.roi}
        if not model_types:
            continue

        def make_alert_callback(camera):
            def on_alert(camera_id, alert_type, confidence, snapshot_path):
                broadcaster.broadcast({
                    'type':        'ALERT',
                    'camera_id':   camera_id,
                    'camera_name': camera.name,
                    'alert_type':  alert_type,
                    'confidence':  round(confidence, 3),
                    'timestamp':   datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                })
                try:
                    with app.app_context():
                        alert = Alert(
                            camera_id=camera_id,
                            alert_type=alert_type,
                            confidence=confidence,
                            snapshot_path=snapshot_path,
                            roi_triggered=True,
                        )
                        db.session.add(alert)
                        db.session.commit()
                except Exception as e:
                    print(f"[Alert] DB write failed: {e}")
            return on_alert

        stream_manager.start_camera(
            cam.id, cam.rtsp_url, model_types, roi_map,
            alert_callback=make_alert_callback(cam)
        )
        print(f"[Setup] Auto-started stream for camera: {cam.name} (id={cam.id})")
