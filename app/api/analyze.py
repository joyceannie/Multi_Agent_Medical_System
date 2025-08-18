from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.graph.graph_builder import build_graph
from app.graph.types import State
from app.utils.helper import convert_uploadfile_to_image  # helper function to handle image
from app.api.schemas import (
    ICD10Response, 
    SOAPResponse, 
    ImageAnalysisResponse, 
    ErrorResponse, 
    ICD10Code, 
    AnalyzeResponse, 
    SOAPNote,
    RadiologyReport
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
graph = build_graph()

@router.post("/analyze")
async def analyze(note: str = Form(None), image: UploadFile = File(None)):
    if not note and not image:
        return JSONResponse(status_code=400, content={"error": "No input provided."})

    try:
        logger.info(f"Recieved inputs - Note: {note}, Image: {image.filename if image else 'None'}")
        state = State(type=None, payload={}, result=None, error=None)
        logger.info(f"Initial state: {state}")

        if note:
            state.payload["note"] = note

        if image and image.filename:
            contents = await image.read()
            pil_image = convert_uploadfile_to_image(contents)
            state.payload["image"] = pil_image

        logger.info(f"Initial state: {state}")

        raw_output = graph.invoke(state)

        # Ensure output is always a State
        if isinstance(raw_output, dict):
            output = State(**raw_output)
        elif isinstance(raw_output, State):
            output = raw_output
        else:
            return ErrorResponse(error="Unexpected output type from graph.")
        
        # Pick the correct model based on type
        if output.error:
            return ErrorResponse(error=output.error)

        if output.type == "icd10":
            # Example output.result expected: [{"code": "...", "description": "..."}]
            # codes = [ICD10Code(**c) for c in output.result]
            codes = [ICD10Code(code=c.code, description=c.description) for c in output.result]
            return ICD10Response(agent="icd10", result=codes)

        elif output.type == "soap":
            # Example output.result expected: {"Subjective": "...", "Objective": "...", "Assessment": "...", "Plan": "..."}
            soap_note = SOAPNote(
                Subjective=output.result.get("Subjective", ""),
                Objective=output.result.get("Objective", ""),
                Assessment=output.result.get("Assessment", ""),
                Plan=output.result.get("Plan", "")
            )
            return SOAPResponse(agent="soap", result=soap_note)

        elif output.type == "image_analysis":
            # Example output.result expected: {"technique": "...", "findings": "...", "impression": "...", "recommendations": "..."}
            radiology_report = RadiologyReport(
                technique=output.result.get("technique", ""),
                findings=output.result.get("findings", ""),
                impression=output.result.get("impression", ""),
                recommendations=output.result.get("recommendations", ""),
                answer_to_user_question=output.result.get("answer_to_user_question", None)
            )
            return ImageAnalysisResponse(agent="image_analysis", result=output.result)

        else:
            return ErrorResponse(error="Unknown analysis type.")

    except Exception as e:
        return ErrorResponse(error=str(e))
