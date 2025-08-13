from PIL import Image
import io
import re

def convert_uploadfile_to_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Invalid image uploaded.") from e
    
def clean_json_response(response: str) -> str:
    # Remove markdown triple backticks and optional language specifier
    cleaned = re.sub(r"^```json\s*|```$", "", response.strip(), flags=re.MULTILINE)
    return cleaned
