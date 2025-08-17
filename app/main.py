from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.analyze import router as analyze_router
import os
from langsmith import Client
from langsmith.run_helpers import traceable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Access LangSmith environment variables
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

app = FastAPI()
langsmith_client = Client()
app.include_router(analyze_router, prefix="/api")

# Serve static HTML
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join("app/static", "index.html"))
