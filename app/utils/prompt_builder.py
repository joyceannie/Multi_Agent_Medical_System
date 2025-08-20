from PIL import Image
import numpy as np
from typing import Optional, List, Any
from mlx_vlm.prompt_utils import apply_chat_template
from app.utils.logger import get_logger

logger = get_logger(__name__)

def build_router_prompt(note: Optional[str], image: Optional[Image]) -> List:
    """
    Builds the prompt for the RouterAgent based on the provided note and image.
    
    Args:
        note (Optional[str]): The clinical note to analyze.
        image (Optional[Image]): The image to analyze.
    
    Returns:
        List: A list of messages formatted for the model input.
    """
    logger.info(f"Building router prompt with note: {note} and image: {image}")
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) if image is None else image
    prompt = f"""
        You are a medical routing agent. Your task is to analyze the provided imputs
        and determine the appropriate next step for processing the input. 
        If there is a textual input, it can be a clinical note or a transcript.
        If there is an image input, it can be a medical image that needs to be analyzed.
        Your task is to determine the type of input and route it to the appropriate agent.
        There are three types of agents:
        1. SOAPGeneratorAgent: For transcripts or clinical conversation between 2 parties, use SOAP note generation.
        2. ICD10Agent: For clinical notes, use ICD-10 coding.
        3. ImageAnalyzerAgent: For medical images that require analysis.
        If the input is a transcript, route it to the SOAPGeneratorAgent.
        If the input is a clinical note, route it to the ICD10Agent.
        If the input is a medical image, route it to the ImageAnalyzerAgent.
        ONLY respond with one of: "icd10", "soap", "image_analysis.
        
        Here is the input you need to analyze:
        text: {note}
        image: {image if image else "No image provided"}"""

    return prompt



def build_icd10_prompt(clinical_note: str, image: Optional[Image]) -> list:
    """
    Builds the prompt for the ICD-10 coding agent based on the clinical note.
    
    Args:
        clinical_note (str): The clinical note to analyze.
    
    Returns:
        list: A list of messages formatted for the model input.
    """
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) if image is None else image
    prompt = f"""
    You are an expert clinical coder. Extract ICD-10 codes from the note below.

    Instructions:
    - Focus on disease, symptom, and condition codes (A00–R99)
    - Avoid administrative or encounter codes (Z00–Z99) unless clinically significant
    - Extract codes from "Diagnosis" and "History & Symptoms" sections
    - Include each code only once with its description
    - Return ONLY valid JSON: an array of objects with double quotes for all keys and values
    - Do not include markdown, code fences, extra text, or repeated codes
    - If unsure, omit rather than guessing

    Example:
    [
    {{"code": "K35.80", "description": "Acute appendicitis, unspecified"}},
    {{"code": "R10.9", "description": "Abdominal pain, unspecified"}},
    {{"code": "R11.0", "description": "Nausea"}}
    ]

    Clinical note:
    {clinical_note}
    Image: {image if image else "No image provided"}
    """
    return prompt

def build_image_analyzer_prompt(image: Image, question: str = None) -> list:
    """
    Builds the prompt for the image analyzer agent based on the provided image.
    
    Args:
        image (Image): The image to analyze.
    
    Returns:
        list: A list of messages formatted for the model input.
    """
    prompt = f"""
        You are an expert radiologist and you are provided with an image of a medical condition.
        Analyze the image and provide a detailed description of the findings,
        including any abnormalities or notable features. If the user provides any question about the image,
        answer it based on the image content.

        Given the findings from a medical image, generate a structured radiology report in JSON format with the following fields:

        "technique": "Describe the imaging technique used (e.g., modality, views, contrast).",
        "findings": "Provide detailed observations from the images.",
        "impression": "Summarize the key conclusions or diagnoses.",
        "recommendations": "Suggest any follow-up, further tests, or clinical advice.",
        "answer_to_user_question": "Answer any specific questions about the image, if provided. Otherwise null"
    

        Return ONLY valid JSON with double quotes, no extra text or markdown.

        Example:

        "technique": "MRI of the brain without contrast.",
        "findings": "No acute infarct or hemorrhage. Normal ventricular size.",
        "impression": "No evidence of acute intracranial pathology.",
        "recommendations": "Clinical correlation recommended.",
        "answer_to_user_question": "The image shows no signs of acute stroke."
    
    Here is the image you need to analyze:
        {image if image else "No image provided."}
        Question: {question if question else "No specific question provided."}

    """
    return prompt

def build_soap_generator_prompt(transcript: str, image: Optional[Image]) -> list:
    """
    Builds the prompt for the SOAP note generator agent based on the clinical note.
    
    Args:
        transcript (str): The transcript to analyze.
    
    Returns:
        list: A list of messages formatted for the model input.
    """
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) if image is None else image
    prompt = f"""
    You are a clinical documentation assistant. Your task is to read medical 
    transcripts (dialogues between clinicians and patients) and convert them 
    into structured clinical notes using the SOAP format.

    Follow these rules:

    *S – Subjective*:
    Include all information reported by the patient: symptoms, duration, history,
    complaints, and any relevant lifestyle or exposure context. 
    Use the patient’s own words when possible (paraphrased for clarity).

    *O – Objective*:
    Include observable findings such as vital signs, physical exam results, 
    lab tests, imaging results, and clinician observations during the encounter.

    *A – Assessment*:
    Provide a brief summary of the clinician’s diagnostic impression. 
    Include possible or confirmed diagnoses.

    *P – Plan*:
    Outline the next steps recommended by the clinician. This can include 
    prescriptions, tests to be ordered, referrals, follow-up instructions, 
    and lifestyle recommendations.

    Keep the format clear and professional. Do not include any parts of 
    the transcript that are irrelevant or non-clinical. Do not invent 
    information not found in the transcript. Allways use a bullet point 
    format for each section of the SOAP note.

    A comprehensive SOAP note has to take into account all subjective and 
    objective information, and accurately assess it to create the 
    patient-specific assessment and plan.

    You shoud return a JSON object with exactly the following fields:

    {{
    "Subjective": "...",
    "Objective": "...",
    "Assessment": "...",
    "Plan": "..."
    }}

    Each field should contain a concise summary relevant to that section.

    Return only valid JSON with double quotes and no extra text or markdown.    

    Here is the transcript from a medical record file from which you will be
    asked to extract relevant SOAP information:

    {transcript}
    Image: {image if image else "No image provided."}

    """
    return prompt