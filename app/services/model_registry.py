"""
model_registry.py
─────────────────
Thread-safe singleton registry for all AI models.

Models
------
intrusion  : YOLOv8n (COCO) — detects class 0 (person) for zone intrusion
fire       : YOLOv8 custom   — detects class 0 (fire) and class 1 (smoke)
violence   : BiLSTM + VGG16 feature extractor
             Architecture:
               input  : VGG16 features → AdaptiveAvgPool2d → 512-d vector
               lstm   : bidirectional, 2 layers, hidden=128, dropout=0.5
               fc1    : 256 → 64
               fc2    : 64  → 2  (0=NonFight, 1=Fight)
"""

import os
import threading
import logging

logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as T
    import numpy as np
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch / torchvision not installed — violence model disabled.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — intrusion/fire models disabled.")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(__file__)
MODELS_FOLDER = os.path.normpath(os.path.join(_HERE, '..', '..', 'models'))

_MODEL_FILES = {
    'intrusion': 'yolov8n.pt',
    'fire':      'fire_detect_model.pt',
    'violence':  'violence_detection.pt',
}

# YOLOv8 class filters
INTRUSION_CLASSES = [0]          # person only
FIRE_CLASSES      = [0, 1]       # fire, smoke

# Violence model hyper-parameters
_V_INPUT_SIZE  = 512
_V_HIDDEN_SIZE = 128
_V_NUM_LAYERS  = 2
_V_NUM_CLASSES = 2     # 0=NonFight, 1=Fight
VIOLENCE_SEQ_LEN  = 30
VIOLENCE_VOTE_LEN = 20  # majority vote window

# ── GPU capability probe ───────────────────────────────────────────────────────
def _probe_gpu() -> dict:
    """
    Returns a dict with GPU info used at startup and by the /api/gpu/status endpoint.
    FP16 (half-precision) is enabled when the GPU compute capability >= 7.0
    (Volta+: V100, RTX 20xx/30xx/40xx). RTX 3050 is compute 8.6 → FP16 ON.
    """
    info = {
        'available': False,
        'name': 'CPU',
        'vram_gb': 0.0,
        'compute_capability': (0, 0),
        'cuda_version': None,
        'cudnn_version': None,
        'fp16_enabled': False,
        'device_str': 'cpu',
    }
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(0)
    cc    = (props.major, props.minor)
    info.update({
        'available':           True,
        'name':                props.name,
        'vram_gb':             round(props.total_memory / 1024 ** 3, 1),
        'compute_capability':  cc,
        'cuda_version':        torch.version.cuda,
        'cudnn_version':       torch.backends.cudnn.version(),
        'fp16_enabled':        cc >= (7, 0),   # Volta and newer
        'device_str':          'cuda:0',
    })
    return info

GPU_INFO: dict = _probe_gpu() if TORCH_AVAILABLE else {
    'available': False, 'name': 'CPU', 'vram_gb': 0.0,
    'compute_capability': (0,0), 'cuda_version': None,
    'cudnn_version': None, 'fp16_enabled': False, 'device_str': 'cpu'
}


# ── BiLSTM architecture ───────────────────────────────────────────────────────

class ViolenceClassifier(nn.Module):
    """
    Bidirectional 2-layer LSTM classifier.
    Input  : (batch, seq_len, 512)  — VGG16 frame features
    Output : (batch, 2)             — logits [NonFight, Fight]
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=_V_INPUT_SIZE,
            hidden_size=_V_HIDDEN_SIZE,
            num_layers=_V_NUM_LAYERS,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.fc1     = nn.Linear(256, 64)
        self.fc2     = nn.Linear(64, _V_NUM_CLASSES)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]            # last timestep → (batch, 256)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)           # (batch, 2)


# ── Registry ──────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Lazy-loading, thread-safe model registry.
    Each model is loaded exactly once on first access.
    """

    def __init__(self):
        self._models: dict = {}
        self._locks:  dict = {k: threading.Lock() for k in _MODEL_FILES}
        self._device    = torch.device(GPU_INFO['device_str']) if TORCH_AVAILABLE else None
        self._fp16      = GPU_INFO['fp16_enabled']          # use half-precision on GPU
        self._fe_lock   = threading.Lock()
        self._feature_extractor = None
        self._transform = None
        if TORCH_AVAILABLE:
            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        if GPU_INFO['available']:
            logger.info(
                f"[ModelRegistry] GPU: {GPU_INFO['name']} "
                f"({GPU_INFO['vram_gb']} GB VRAM, "
                f"compute {GPU_INFO['compute_capability'][0]}.{GPU_INFO['compute_capability'][1]}, "
                f"CUDA {GPU_INFO['cuda_version']}, "
                f"FP16={'ON' if self._fp16 else 'OFF'})"
            )
        else:
            logger.warning("[ModelRegistry] No GPU found — running on CPU")

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, model_type: str):
        """Return the loaded model for model_type, or None if unavailable."""
        if model_type not in _MODEL_FILES:
            return None
        with self._locks[model_type]:
            if model_type not in self._models:
                self._models[model_type] = self._load(model_type)
        return self._models.get(model_type)

    def get_feature_extractor(self):
        """Return the shared MobileNetV2 feature extractor (for violence model)."""
        with self._fe_lock:
            if self._feature_extractor is None:
                self._feature_extractor = self._load_feature_extractor()
        return self._feature_extractor

    @property
    def device(self):
        return self._device

    @property
    def fp16(self) -> bool:
        return self._fp16

    @property
    def transform(self):
        return self._transform

    @property
    def gpu_info(self) -> dict:
        return GPU_INFO

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load(self, model_type: str):
        path = os.path.join(MODELS_FOLDER, _MODEL_FILES[model_type])
        if not os.path.exists(path):
            logger.warning(f"[ModelRegistry] Model file not found: {path}")
            return None

        try:
            if model_type in ('intrusion', 'fire'):
                return self._load_yolo(model_type, path)
            elif model_type == 'violence':
                return self._load_violence(path)
        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to load {model_type}: {e}", exc_info=True)
            return None

    def _load_yolo(self, model_type: str, path: str):
        if not YOLO_AVAILABLE:
            return None
        model = YOLO(path)
        # Do NOT call .to('cuda') at load time — ultralytics manages the device
        # internally via the device= argument in predict(). Pre-moving causes
        # the FP16 BatchNorm bug on torch 2.4.0 Windows (issue #15049).
        logger.info(
            f"[ModelRegistry] Loaded YOLO '{model_type}' — classes: {model.names}"
        )
        return model

    def _load_violence(self, path: str):
        if not TORCH_AVAILABLE:
            return None
        state_dict = torch.load(path, map_location=self._device, weights_only=True)
        model = ViolenceClassifier()
        model.load_state_dict(state_dict)
        model.to(self._device)
        if self._fp16:
            model.half()
        model.eval()
        logger.info(
            f"[ModelRegistry] Loaded ViolenceClassifier on {self._device} "
            f"({'FP16' if self._fp16 else 'FP32'})"
        )
        return model

    def _load_feature_extractor(self):
        """VGG16 truncated at features + AdaptiveAvgPool2d — outputs 512-d vectors."""
        if not TORCH_AVAILABLE:
            return None
        try:
            vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
            extractor = nn.Sequential(
                vgg.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            extractor.to(self._device)
            extractor.eval()
            for p in extractor.parameters():
                p.requires_grad = False
            logger.info(
                f"[ModelRegistry] Loaded VGG16 FE on {self._device}"
            )
            return extractor
        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to load VGG16 feature extractor: {e}", exc_info=True)
            return None


# ── Global singleton ──────────────────────────────────────────────────────────
registry = ModelRegistry()
