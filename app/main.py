# app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.analyze import router as analyze_router
import os

app = FastAPI()

app.include_router(analyze_router, prefix="/api")

# Serve static HTML
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join("app/static", "index.html"))
