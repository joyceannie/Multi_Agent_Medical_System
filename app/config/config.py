import mlx.core as mx
from mlx_lm import load, generate 

class Config:
    MODEL_ID = "mlx-community/medgemma-4b-it-4bit"
    # MAX_NEW_TOKENS = 1024
    # TORCH_DTYPE = torch.float32 if torch.backends.mps.is_available() else torch.bfloat16
    # DEVICE_MAP = "auto"

config = Config()