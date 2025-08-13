import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from app.config.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_processor = None

def load_medgemma_model():
    global _model, _processor

    if _model is None or _processor is None:
        logger.info("[INFO] Loading MedGemma model and processor...")
        _model = AutoModelForImageTextToText.from_pretrained(
            config.MODEL_ID,
            torch_dtype=config.TORCH_DTYPE,
            device_map=config.DEVICE_MAP,
        )
        _processor = AutoProcessor.from_pretrained(config.MODEL_ID)

    return _model, _processor
