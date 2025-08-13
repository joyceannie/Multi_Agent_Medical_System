import os
import torch

class Config:
    MODEL_ID = "google/medgemma-4b-it"
    MAX_NEW_TOKENS = 1024
    TORCH_DTYPE = torch.float32 if torch.backends.mps.is_available() else torch.bfloat16
    DEVICE_MAP = "auto"

config = Config()