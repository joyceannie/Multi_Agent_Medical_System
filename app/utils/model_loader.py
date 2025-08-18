from mlx_vlm import load
from mlx_vlm.utils import load_config
from app.config.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_processor = None
_config = None

def load_medgemma_model():
    global _model, _processor, _config

    if _model is None or _processor is None:
        logger.info("[INFO] Loading {config.MODEL_ID}...")
        _model, _processor = load(config.MODEL_ID)
        _config = load_config(config.MODEL_ID)

    return _model, _processor, _config
