# ShieldVision Pro — Intelligent Surveillance System

A full-stack, role-based AI surveillance platform built with Flask, PostgreSQL, OpenCV, and YOLOv8.

---

## Project Structure

```
shieldvision_pro/
├── app/
│   ├── api/
│   │   ├── auth.py          # Login, logout, user CRUD, bulk CSV import
│   │   ├── cameras.py       # Camera CRUD, stream start/stop, ROI, user assignment
│   │   ├── alerts.py        # Alert CRUD, SSE real-time events, filtering
│   │   └── pages.py         # HTML page routes (admin + user)
│   ├── models/
│   │   └── database.py      # SQLAlchemy models: User, Camera, Alert, mappings
│   ├── services/
│   │   ├── stream_service.py  # RTSP capture, YOLOv8 inference, multi-camera manager
│   │   └── alert_service.py   # SSE broadcaster for real-time alerts
│   ├── static/
│   │   ├── css/main.css     # Dark glassmorphism UI theme
│   │   └── sounds/alarm.ogg # Alert sound (copy from fight_detection_1)
│   └── templates/
│       ├── base.html
│       ├── auth/login.html
│       ├── admin/
│       │   ├── dashboard.html
│       │   ├── cameras.html
│       │   ├── users.html
│       │   └── alerts.html
│       └── user/
│           ├── dashboard.html
│           └── camera_view.html
├── models/                  # AI model weights (.pt files)
├── uploads/                 # Snapshots from alerts
├── config.py
├── run.py
├── requirements.txt
└── sample_users.csv
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- PostgreSQL 14+
- (Optional) NVIDIA GPU for faster inference

### 2. Create PostgreSQL Database
```sql
CREATE DATABASE shieldvision;
```

### 3. Create Virtual Environment
```bash
cd shieldvision_pro
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

> For GPU support replace `torch==2.3.0` with the CUDA version from https://pytorch.org

### 5. Configure Environment
```bash
copy .env.example .env
# Edit .env with your PostgreSQL credentials
```

Or set environment variables directly:
```bash
set DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/shieldvision
set SECRET_KEY=your-random-secret
```

### 6. Copy AI Models
Copy your `.pt` model files into the `models/` folder:
```
models/
├── yolov8n.pt              # Intrusion detection (YOLOv8)
├── fire_detect_model.pt    # Fire detection
└── violence_detection.pt   # Violence detection
```

Copy the alarm sound:
```bash
copy ..\fight_detection_1\static\alarm.ogg app\static\sounds\alarm.ogg
```

### 7. Run the Application
```bash
python run.py
```

Open: http://localhost:5001

---

## Default Credentials

| Role  | Username | Password  |
|-------|----------|-----------|
| Admin | admin    | admin123  |

> Change the admin password after first login via the Users page.

---

## Features

### Admin
- **Dashboard**: Stats overview + live alert feed
- **Cameras**: Add/edit/delete cameras with RTSP URLs, assign AI models (intrusion/fire/violence), assign users
- **Users**: Create users individually or bulk import via CSV (`username,email,password`)
- **Alerts**: Filter by camera, type, date range; real-time updates

### User
- **Dashboard**: View assigned cameras + global alert table with live SSE updates
- **Camera View**: Live MJPEG stream, draw ROI zones per model, per-camera alert log with alarm sound

### AI Processing
- Multi-camera concurrent streaming via `StreamManager`
- YOLOv8 inference every 3rd frame (configurable)
- ROI intersection check triggers alerts
- 5-second cooldown per alert type per camera
- Snapshots saved on alert trigger

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/login` | Login |
| POST | `/api/logout` | Logout |
| GET/POST | `/api/users` | List / create users |
| POST | `/api/users/bulk` | Bulk CSV import |
| DELETE | `/api/users/<id>` | Delete user |
| GET/POST | `/api/cameras` | List / add cameras |
| PUT | `/api/cameras/<id>` | Update camera |
| DELETE | `/api/cameras/<id>` | Delete camera |
| POST | `/api/cameras/<id>/assign` | Assign users to camera |
| POST | `/api/cameras/<id>/roi` | Set ROI for model |
| POST | `/api/cameras/<id>/stream/start` | Start AI stream |
| POST | `/api/cameras/<id>/stream/stop` | Stop stream |
| GET | `/api/cameras/<id>/feed` | MJPEG video feed |
| GET | `/api/alerts` | Get alerts (with filters) |
| GET | `/api/events` | SSE stream (all alerts) |
| GET | `/api/events/camera/<id>` | SSE stream (per camera) |

---

## Database Schema

```
users              cameras              alerts
─────────────      ──────────────       ──────────────────
id                 id                   id
username           name                 camera_id → cameras
email              rtsp_url             alert_type
password_hash      location             confidence
role               is_active            timestamp
is_active          created_by → users   snapshot_path
created_at         created_at           roi_triggered

user_camera_mapping          camera_model_mapping
───────────────────          ────────────────────
user_id → users              camera_id → cameras
camera_id → cameras          model_type
assigned_at                  roi (JSON)
assigned_by → users          is_enabled
```

---

## Scalability Notes

- `StreamManager` is a thread-safe singleton; each camera runs in its own daemon thread
- AI models are lazy-loaded and cached in memory
- SSE broadcaster supports unlimited concurrent clients
- To scale horizontally: replace SSE broadcaster with Redis Pub/Sub, use Celery for inference workers
