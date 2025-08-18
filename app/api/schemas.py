from pydantic import BaseModel
from typing import List, Optional, Literal, Union, TypeAlias

class ICD10Code(BaseModel):
    code: str
    description: str

class ICD10Response(BaseModel):
    agent: Literal["icd10"]
    result: List[ICD10Code]


class SOAPNote(BaseModel):
    Subjective: str
    Objective: str
    Assessment: str
    Plan: str

class SOAPResponse(BaseModel):
    agent: Literal["soap"]
    result: SOAPNote

class RadiologyReport(BaseModel):
    technique: str
    findings: str
    impression: str
    recommendations: str
    answer_to_user_question: Optional[str] = None

class ImageAnalysisResponse(BaseModel):
    agent: str = "image_analysis"
    result: "RadiologyReport"

class ErrorResponse(BaseModel):
    error: str

# Type alias for FastAPI response_model
AnalyzeResponse: TypeAlias = Union[
    ICD10Response,
    SOAPResponse,
    ImageAnalysisResponse,
    ErrorResponse
]
