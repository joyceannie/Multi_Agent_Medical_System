from PIL import Image
import base64
from io import BytesIO
import io
import re
from app.utils.logger import get_logger
import json

logger = get_logger(__name__)

def convert_uploadfile_to_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Invalid image uploaded.") from e
    
    
def clean_json_response(response: str) -> str:
    # Replace single quotes with double quotes
    response = re.sub(r"'", '"', response)
    # Remove markdown triple backticks and optional language specifier
    cleaned = re.sub(r"^```json\s*|```$", "", response.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        logger.info(f"Parsed JSON response: {data}")
        # If data is a list, remove entries with empty 'description'
        if isinstance(data, list):
            data = [entry for entry in data if entry.get("description")]
        # If data is a dict with a list under a key (e.g., 'results'), clean that list
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = [entry for entry in value if entry.get("description")]
        logger.info(f"Cleaned JSON response: {data}")
        return json.dumps(data)
    except Exception as e:
        logger.error(f"JSON parsing failed: {e}\nRaw response: {cleaned}")
        # Try to recover valid objects from incomplete JSON
        # This regex matches objects like {"code": "...", "description": "..."}
        matches = re.findall(r'\{[^{}]*"code"\s*:\s*"[^"]*",\s*"description"\s*:\s*"[^"]*"\s*\}', cleaned)
        recovered = []
        for m in matches:
            try:
                obj = json.loads(m)
                if obj.get("description"):
                    recovered.append(obj)
            except Exception:
                continue
        logger.info(f"Recovered partial JSON objects: {recovered}")
        return json.dumps(recovered)
