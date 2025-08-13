from PIL import Image
import numpy as np
from typing import Optional, List
from app.utils.logger import get_logger

def build_icd10_prompt(clinical_note: str, image: Optional[Image]) -> list:
    """
    Builds the prompt for the ICD-10 coding agent based on the clinical note.
    
    Args:
        clinical_note (str): The clinical note to analyze.
    
    Returns:
        list: A list of messages formatted for the model input.
    """
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) if image is None else image
    system_prompt ="""
        You are an expert clinical coder. From the following medical note, 
        identify the most relevant ICD-10 codes:

        Instructions:
        - Include all codes for all sections of the clinical note 
        (complain, history and symptomps, diagnosis, plan
        - Include a code only once

        Given a clinical note, return a JSON array where each item contains:
        - code: the ICD-10 code
        - description: the ICD-10 description

        Return ONLY valid JSON with double quotes, no extra text, no markdown.
        There should be no additional text or code fences. 
        The response must be a valid JSON array of objects, each having code and description fields.

        
        Example:
        [
        {"code": "XXX", "description": "YYYY"},
        {"code": "XXX", "description": "YYYY"}
        ]


    """

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text":  system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": clinical_note},
                {"type": "image", "image": image}
            ]
        }
    ]
    return messages

def build_image_analyzer_prompt(image: Image, question: str = None) -> list:
    """
    Builds the prompt for the image analyzer agent based on the provided image.
    
    Args:
        image (Image): The image to analyze.
    
    Returns:
        list: A list of messages formatted for the model input.
    """
    system_prompt = f"""
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
    

    """

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text":  system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question if question else "No specific question provided."}
            ]
        }
    ]
    return messages

def build_soap_generator_prompt(transcript: str, image: Optional[Image]) -> list:
    """
    Builds the prompt for the SOAP note generator agent based on the clinical note.
    
    Args:
        transcript (str): The transcript to analyze.
    
    Returns:
        list: A list of messages formatted for the model input.
    """
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) if image is None else image
    system_prompt = """
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

    {
    "Subjective": "...",
    "Objective": "...",
    "Assessment": "...",
    "Procedure": "..."
    }

    Each field should contain a concise summary relevant to that section.

    Return only valid JSON with double quotes and no extra text or markdown.    

    Here is the transcript from a medical record file from which you will be
    asked to extract relevant SOAP information:

    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text":  system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": transcript},
                {"type": "image", "image": image}
            ]
        }
    ]
    return messages