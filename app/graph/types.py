# app/graph/types.py

from typing import TypedDict, Literal, Union, Optional, List
from PIL import Image as PILImage
from pydantic import BaseModel
from app.api.schemas import ICD10Code


class ICD10Payload(TypedDict):
    clinical_note: str

class SOAPPayload(TypedDict):
    transcript: str

class ImageAnalysisPayload(TypedDict):
    image: PILImage.Image
    clinical_note: Optional[str]

class State(BaseModel):
    type: Optional[Literal["icd10", "soap", "image_analysis"]]
    payload: dict  
    result: Optional[Union[str, List[ICD10Code], dict]]  # accept str or list or dict
    error: Optional[str]