# app/main.py

from fastapi import FastAPI
from app.api.analyze import router as analyze_router

app = FastAPI(
    title="Multi-Agent Healthcare Assistant",
    description="Accepts clinical text + image and processes via multiple agents.",
    version="1.0.0"
)

app.include_router(analyze_router, prefix="/api")
