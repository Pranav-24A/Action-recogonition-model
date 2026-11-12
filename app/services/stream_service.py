"""
stream_service.py
─────────────────
Manages concurrent RTSP camera streams with per-camera AI inference.

GPU support
───────────
- YOLO models  : device=0 (cuda:0) + half=True when FP16 is enabled
- Violence LSTM: tensors cast to .half() on GPU; softmax kept in FP32
- MobileNetV2  : .half() on GPU; torch.cuda.amp.autocast() wraps forward
- CPU fallback : automatic when no CUDA device is detected
"""

import cv2
import os
import time
import queue
import threading
import logging
from datetime import datetime, timezone
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1280, 720

INFER_EVERY = {
    'intrusion': 3,
    'fire':      3,
    'violence':  2,
}

ALERT_COOLDOWN = 5.0

CONF_THRESHOLD = {
    'intrusion': 0.45,
    'fire':      0.40,
    'violence':  0.60,
}

# Consecutive detections required before triggering an alert
DETECT_CONFIRM_FRAMES = {
    'intrusion': 3,
    'fire':      4,
    'violence':  3,
}

# Consecutive non-detections required before the event is considered cleared
CLEAR_FRAMES = {
    'intrusion': 10,
    'fire':      15,
    'violence':  10,
}

VIOLENCE_SEQ_LEN    = 30
VIOLENCE_TARGET_FPS = 10   # process every Nth frame
VIOLENCE_VOTE_LEN   = 20   # majority vote window

RECONNECT_INIT    = 2.0
RECONNECT_MAX     = 30.0
RECONNECT_BACKOFF = 2.0

OVERLAY_COLORS = {
    'intrusion': (0, 255, 255),
    'fire':      (0, 100, 255),
    'violence':  (0, 0, 255),
}

SNAP_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'uploads', 'snapshots')
)


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """Stateful per-camera inference engine with GPU/FP16 support."""

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self._frame_counters: dict[str, int] = {}
        self._violence_buffer: deque = deque(maxlen=VIOLENCE_SEQ_LEN)
        self._violence_votes:  deque = deque(maxlen=VIOLENCE_VOTE_LEN)
        self._violence_last_ts: float = 0.0
        self._inference_enabled: dict[str, bool] = {
            'intrusion': False,
            'fire':      True,
            'violence':  True,
        }
        # Consecutive detection / clear streak counters per model
        self._detect_streak: dict[str, int] = {}
        self._clear_streak:  dict[str, int] = {}
        self._event_active:  dict[str, bool] = {}  # True = event currently ongoing

    def enable_inference(self, mtype: str) -> None:
        """Called when ROI is set for a model — unblocks inference."""
        self._inference_enabled[mtype] = True
        logger.info(f"[Cam {self.camera_id}] Inference ENABLED for '{mtype}'")

    def disable_inference(self, mtype: str) -> None:
        """Called when ROI is cleared — blocks intrusion inference again."""
        if mtype == 'intrusion':
            self._inference_enabled[mtype] = False
            logger.info(f"[Cam {self.camera_id}] Inference PAUSED for '{mtype}' (ROI cleared)")

    def run(
        self,
        frame: np.ndarray,
        model_types: list[str],
        roi_map: dict,
    ) -> list[tuple[str, float, list]]:
        alerts = []
        for mtype in model_types:
            if not self._inference_enabled.get(mtype, True):
                if mtype == 'intrusion':
                    cv2.putText(frame, "DRAW ROI ZONE TO START DETECTION",
                                (TARGET_W // 2 - 280, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                continue

            self._frame_counters[mtype] = self._frame_counters.get(mtype, 0) + 1
            if self._frame_counters[mtype] % INFER_EVERY.get(mtype, 3) != 0:
                continue

            try:
                detected = False
                conf     = 0.0
                box      = []

                if mtype in ('intrusion', 'fire'):
                    results = self._run_yolo(frame, mtype, roi_map.get(mtype))
                    if results:
                        detected = True
                        conf     = results[0][1]
                        box      = results[0][2]
                elif mtype == 'violence':
                    result = self._run_violence(frame, roi_map.get(mtype))
                    if result:
                        detected = True
                        conf     = result[1]
                        box      = result[2]

                # Update streaks
                if detected:
                    self._detect_streak[mtype] = self._detect_streak.get(mtype, 0) + 1
                    self._clear_streak[mtype]  = 0
                else:
                    self._clear_streak[mtype]  = self._clear_streak.get(mtype, 0) + 1
                    self._detect_streak[mtype] = 0

                # Clear event if absent long enough
                if self._event_active.get(mtype) and \
                        self._clear_streak.get(mtype, 0) >= CLEAR_FRAMES.get(mtype, 10):
                    self._event_active[mtype] = False
                    logger.info(f"[Cam {self.camera_id}] {mtype} event CLEARED")

                # Trigger alert only after enough consecutive detections
                # and only if event is not already active
                if detected and not self._event_active.get(mtype) and \
                        self._detect_streak.get(mtype, 0) >= DETECT_CONFIRM_FRAMES.get(mtype, 3):
                    self._event_active[mtype] = True
                    alerts.append((mtype, conf, box))

            except Exception as e:
                logger.error(f"[Cam {self.camera_id}] Inference error ({mtype}): {e}")
        return alerts

    # ── YOLO pipeline ─────────────────────────────────────────────────────────

    def _run_yolo(
        self,
        frame: np.ndarray,
        mtype: str,
        roi: list | None,
    ) -> list[tuple[str, float, list]]:
        from app.services.model_registry import registry, INTRUSION_CLASSES, FIRE_CLASSES

        model = registry.get(mtype)
        if model is None:
            return []

        classes     = INTRUSION_CLASSES if mtype == 'intrusion' else FIRE_CLASSES
        conf_thresh = CONF_THRESHOLD[mtype]
        gpu_info    = registry.gpu_info

        # half=False: torch 2.4.0 on Windows has a known FP16 BatchNorm bug
        # (ultralytics issue #15049). Run YOLO in FP32 on GPU — still much
        # faster than CPU. FP16 is only used for the PyTorch BiLSTM pipeline.
        results = model.predict(
            frame,
            classes=classes,
            conf=conf_thresh,
            device=0 if gpu_info['available'] else 'cpu',
            half=False,
            verbose=False,
        )

        alerts = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                color = OVERLAY_COLORS[mtype]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{mtype} {conf:.2f}", (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if roi is None or _boxes_intersect(xyxy, roi, TARGET_W, TARGET_H):
                    alerts.append((mtype, conf, xyxy))
        return alerts

    # ── Violence pipeline ─────────────────────────────────────────────────────

    def _run_violence(
        self,
        frame: np.ndarray,
        roi: list | None,
    ) -> tuple[str, float, list] | None:
        from app.services.model_registry import registry
        import torch

        # Sample at exactly 10fps regardless of camera FPS
        now = time.time()
        if now - self._violence_last_ts < 1.0 / VIOLENCE_TARGET_FPS:
            return None
        self._violence_last_ts = now

        fe         = registry.get_feature_extractor()
        classifier = registry.get('violence')
        if fe is None or classifier is None:
            return None

        device = registry.device

        # Crop to ROI or use full frame
        crop = _crop_roi(frame, roi) if roi else frame
        if crop is None or crop.size == 0:
            crop = frame

        # BGR → RGB → tensor
        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = registry.transform(rgb).unsqueeze(0).to(device)
        # Cast to match feature extractor dtype
        fe_dtype = next(fe.parameters()).dtype
        tensor = tensor.to(fe_dtype)

        # VGG16 feature extraction → 512-d vector
        with torch.no_grad():
            feature = fe(tensor).squeeze(0)   # (512,)

        self._violence_buffer.append(feature.cpu().float())

        if len(self._violence_buffer) < VIOLENCE_SEQ_LEN:
            return None

        # Classify sequence
        seq = torch.stack(list(self._violence_buffer), dim=0).unsqueeze(0).to(device)
        # Cast to match classifier dtype
        classifier_dtype = next(classifier.parameters()).dtype
        seq = seq.to(classifier_dtype)
        with torch.no_grad():
            pred = torch.argmax(classifier(seq), dim=1).item()  # 0=NonFight, 1=Fight

        self._violence_votes.append(pred)

        if len(self._violence_votes) < VIOLENCE_VOTE_LEN:
            return None

        # Majority vote — same as Counter.most_common(1) in original code
        from collections import Counter
        final_pred = Counter(self._violence_votes).most_common(1)[0][0]
        if final_pred == 1:  # Fight
            confidence = self._violence_votes.count(1) / len(self._violence_votes)
            color = OVERLAY_COLORS['violence']
            cv2.putText(frame, f"Fight {confidence:.0%}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.rectangle(frame, (0, 0), (TARGET_W - 1, TARGET_H - 1), color, 3)
            return ('violence', confidence, [])

        return None


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _boxes_intersect(xyxy: list, roi: list, w: int, h: int) -> bool:
    ax1, ay1, ax2, ay2 = xyxy
    bx1 = roi[0] * w;  by1 = roi[1] * h
    bx2 = roi[2] * w;  by2 = roi[3] * h
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def _crop_roi(frame: np.ndarray, roi: list) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1 = int(roi[0] * w);  y1 = int(roi[1] * h)
    x2 = int(roi[2] * w);  y2 = int(roi[3] * h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _draw_roi_zones(frame: np.ndarray, roi_map: dict, alert_types: set) -> None:
    for mtype, roi in roi_map.items():
        if not roi or len(roi) != 4:
            continue
        x1 = int(roi[0] * TARGET_W);  y1 = int(roi[1] * TARGET_H)
        x2 = int(roi[2] * TARGET_W);  y2 = int(roi[3] * TARGET_H)
        color = (0, 0, 255) if mtype in alert_types else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ROI:{mtype}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ── CameraStream ──────────────────────────────────────────────────────────────

class CameraStream:
    """One camera: capture → GPU inference → alert → frame queue."""

    def __init__(
        self,
        camera_id: int,
        rtsp_url: str,
        model_types: list[str],
        roi: dict | None = None,
        alert_callback=None,
    ):
        self.camera_id       = camera_id
        self.rtsp_url        = rtsp_url
        self._lock           = threading.Lock()
        self._model_types    = list(model_types)
        self._roi            = dict(roi) if roi else {}
        self._alert_callback = alert_callback
        self._running        = False
        self._thread: threading.Thread | None = None
        self._frame_queue    = queue.Queue(maxsize=2)
        self._engine         = InferenceEngine(camera_id)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop,
            name=f"cam-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[Cam {self.camera_id}] Stream started → {self.rtsp_url}")

    def stop(self) -> None:
        self._running = False
        logger.info(f"[Cam {self.camera_id}] Stream stopping…")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_models(self, model_types: list[str]) -> None:
        with self._lock:
            self._model_types = list(model_types)

    def update_roi(self, roi: dict) -> None:
        """Update ROI map and enable/disable inference per model accordingly."""
        with self._lock:
            self._roi = dict(roi) if roi else {}
        # Enable inference for any model that now has an ROI
        # Disable intrusion inference if its ROI was cleared
        for mtype in ('intrusion', 'fire', 'violence'):
            has_roi = bool(roi and roi.get(mtype))
            if mtype == 'intrusion':
                if has_roi:
                    self._engine.enable_inference('intrusion')
                else:
                    self._engine.disable_inference('intrusion')

    def get_frame(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        reconnect_wait = RECONNECT_INIT

        while self._running:
            cap = self._open_capture()
            if cap is None:
                logger.warning(
                    f"[Cam {self.camera_id}] Cannot open source. "
                    f"Retrying in {reconnect_wait:.0f}s…"
                )
                time.sleep(reconnect_wait)
                reconnect_wait = min(reconnect_wait * RECONNECT_BACKOFF, RECONNECT_MAX)
                continue

            reconnect_wait = RECONNECT_INIT
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 25.0
            frame_delay = 1.0 / fps
            logger.info(f"[Cam {self.camera_id}] Capture opened @ {fps:.1f} fps")

            while self._running:
                loop_start = time.time()
                ret, raw   = cap.read()

                if not ret:
                    logger.warning(f"[Cam {self.camera_id}] Frame read failed — reconnecting…")
                    break

                frame = cv2.resize(raw, (TARGET_W, TARGET_H))

                with self._lock:
                    active_models = list(self._model_types)
                    current_roi   = dict(self._roi)

                detections    = self._engine.run(frame, active_models, current_roi)
                alerted_types: set[str] = set()

                for (mtype, conf, box) in detections:
                    alerted_types.add(mtype)
                    self._trigger_alert(mtype, conf, box, frame)

                _draw_roi_zones(frame, current_roi, alerted_types)

                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                cv2.putText(frame, ts, (10, TARGET_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                self._push_frame(frame)

                elapsed   = time.time() - loop_start
                sleep_for = max(0.001, frame_delay - elapsed)
                time.sleep(sleep_for)

            cap.release()

        logger.info(f"[Cam {self.camera_id}] Capture loop exited.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture | None:
        source: str | int = self.rtsp_url
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        # Resolve YouTube URLs to direct stream URL via yt-dlp
        if isinstance(source, str) and ('youtube.com' in source or 'youtu.be' in source):
            source = self._resolve_youtube(source) or source

        cap = cv2.VideoCapture(source)
        if isinstance(source, str) and source.lower().startswith('rtsp'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def _resolve_youtube(self, url: str) -> str | None:
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info.get('url') or (
                    info['formats'][-1]['url'] if info.get('formats') else None
                )
                logger.info(f"[Cam {self.camera_id}] Resolved YouTube URL")
                return stream_url
        except Exception as e:
            logger.error(f"[Cam {self.camera_id}] yt-dlp failed: {e}")
            return None

    def _push_frame(self, frame: np.ndarray) -> None:
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

    def _trigger_alert(
        self,
        alert_type: str,
        confidence: float,
        box: list,
        frame: np.ndarray,
    ) -> None:
        snapshot_path = self._save_snapshot(alert_type, frame, time.time())
        if self._alert_callback:
            try:
                self._alert_callback(self.camera_id, alert_type, confidence, snapshot_path)
            except Exception as e:
                logger.error(f"[Cam {self.camera_id}] Alert callback error: {e}", exc_info=True)

    def _save_snapshot(self, alert_type: str, frame: np.ndarray, ts: float) -> str | None:
        try:
            os.makedirs(SNAP_DIR, exist_ok=True)
            fname = f"cam{self.camera_id}_{alert_type}_{int(ts)}.jpg"
            path  = os.path.join(SNAP_DIR, fname)
            cv2.imwrite(path, frame)
            return path
        except Exception as e:
            logger.warning(f"[Cam {self.camera_id}] Snapshot save failed: {e}")
            return None


# ── StreamManager ─────────────────────────────────────────────────────────────

class StreamManager:
    """Thread-safe registry of all active CameraStream instances."""

    def __init__(self):
        self._streams: dict[int, CameraStream] = {}
        self._lock = threading.Lock()

    def start_camera(
        self,
        camera_id: int,
        rtsp_url: str,
        model_types: list[str],
        roi: dict | None = None,
        alert_callback=None,
    ) -> None:
        with self._lock:
            if camera_id in self._streams:
                self._streams[camera_id].stop()
            stream = CameraStream(camera_id, rtsp_url, model_types, roi, alert_callback)
            stream.start()
            self._streams[camera_id] = stream

    def stop_camera(self, camera_id: int) -> None:
        with self._lock:
            stream = self._streams.pop(camera_id, None)
        if stream:
            stream.stop()

    def stop_all(self) -> None:
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
        for s in streams:
            s.stop()

    def get_frame(self, camera_id: int, timeout: float = 0.1) -> np.ndarray | None:
        with self._lock:
            stream = self._streams.get(camera_id)
        return stream.get_frame(timeout) if stream else None

    def is_running(self, camera_id: int) -> bool:
        with self._lock:
            return camera_id in self._streams

    def update_roi(self, camera_id: int, roi: dict) -> None:
        with self._lock:
            stream = self._streams.get(camera_id)
        if stream:
            stream.update_roi(roi)

    def update_models(self, camera_id: int, model_types: list[str]) -> None:
        with self._lock:
            stream = self._streams.get(camera_id)
        if stream:
            stream.update_models(model_types)

    def active_cameras(self) -> list[int]:
        with self._lock:
            return list(self._streams.keys())

    def status(self) -> dict:
        with self._lock:
            return {
                cid: {
                    'alive':   s.is_alive(),
                    'models':  s._model_types,
                    'has_roi': bool(s._roi),
                }
                for cid, s in self._streams.items()
            }


# ── Global singleton ──────────────────────────────────────────────────────────
stream_manager = StreamManager()
