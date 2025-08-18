from app.agents.base_agent import BaseAgent
from app.utils.model_loader import load_medgemma_model
from app.utils.prompt_builder import build_icd10_prompt
from app.graph.types import State
from app.utils.logger import get_logger
from app.utils.helper import clean_json_response
import json
from PIL import Image
from typing import Optional
from langsmith.run_helpers import traceable
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import generate
import numpy as np

logger = get_logger(__name__)

class ICD10Agent(BaseAgent):
    def __init__(self):
        super().__init__(name="ICD10Agent")
        self.model, self.processor, self.config = load_medgemma_model()

    @traceable
    def respond(self, state: State) -> str:

        logger.info(f"Called respond with state: {state}")
        clinical_note = state.payload["clinical_note"] if "clinical_note" in state.payload else None
        image = [state.payload["image"] if "image" in state.payload else  Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] 

        prompt = build_icd10_prompt(clinical_note, image)
        formatted_prompt = apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )
        logger.info(f"Generating ICD-10 codes for clinical note: {clinical_note}")
        return generate(self.model, self.processor, formatted_prompt, image)
    

    def run(self, state: State) -> State:
        logger.info("Running ICD10Agent with state: %s", state)
        try:
            raw_result = self.respond(state).text
            logger.info("ICD10Agent response: %s", raw_result)
            parsed_result = clean_json_response(raw_result)
            cleaned_result = json.loads(parsed_result)
            
            logger.info("Returning from icd10 agent with : %s", cleaned_result)
            return State(
                type="icd10",
                payload=state.payload,  # preserve existing payload
                result=cleaned_result,             # add new result
                error=None 
            )
        except Exception as e:
            return {
                "payload": state.payload,
                "error": str(e),             # add error message if exception
            }
        
# if __name__ == "__main__":
#     agent = ICD10Agent()
#     clinical_note = """
#     Chief Complaint:
#     The patient presents with abdominal pain localized to the lower right quadrant, nausea, and low-grade fever for the past 24 hours.

#     History & Symptoms:
#     - Pain began near the umbilicus and migrated to the lower right abdomen
#     - Mild nausea, no vomiting
#     - Pain increases with movement or coughing
#     - Temperature: 38.1°C (100.6°F)

#     Physical Exam:
#     - Rebound tenderness in the right lower quadrant
#     - Positive Rovsing’s sign
#     - No palpable masses

#     Diagnosis:
#     - Acute uncomplicated appendicitis

#     Plan:
#     - Schedule for laparoscopic appendectomy
#     - Start IV fluids and antibiotics pre-op
#     """
#     response = agent.respond(clinical_note)
#     print(f"ICD-10 Codes: {response}")