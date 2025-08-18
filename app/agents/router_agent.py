# app/agents/router_agent.py
from PIL import Image
import base64
import io
from app.utils.logger import get_logger
from app.graph.types import State
from app.utils.prompt_builder import build_router_prompt
from langsmith.run_helpers import traceable
from app.agents.base_agent import BaseAgent
from app.utils.model_loader import load_medgemma_model
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import generate
import numpy as np


logger = get_logger(__name__)

class RouterAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="RouterAgent")
        self.model, self.processor, self.config = load_medgemma_model()

    @traceable
    def respond(self, state: dict) -> str:
        image = [state.payload["image"] if "image" in state.payload else  Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] 
        note = state.payload.get("note", None)
        logger.info(f"Identifying next agent for image: {image} with note: {note}")
        prompt = build_router_prompt(note, image)
        logger.info(f"RouterAgent prompt: {prompt}")
        # Apply chat template
        formatted_prompt = apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )
        logger.info(f"Formatted prompt for RouterAgent: {formatted_prompt}")
        return generate(self.model, self.processor, formatted_prompt, image)
    
    
    def run(self, state: State) -> State:
        """
        Run the agent with the provided image.
        """
        logger.info(f"Running {self.name} with state: {state}")
        
        response = self.respond(state).text.lower().strip()
        logger.info("RouterAgent response: %s", response)

        if response == "icd10":
            state.payload["clinical_note"] = state.payload.get("note", "")
            state.type = "icd10"
        elif response == "soap":
            state.payload["transcript"] = state.payload.get("note", "")
            state.type = "soap"
        elif response == "image_analysis":
            state.type = "image_analysis"
            state.payload["image"] = state.payload.get("image", None)
            state.payload["clinical_note"] = state.payload.get("note", "")
        else:
            logger.error(f"Unknown response from RouterAgent: {response}")
            state.error = f"Unknown response from RouterAgent: {response}"
            return state
        return State(
            type=state.type,
            payload=state.payload,  # preserve existing payload
            result=response,            # add this line (or appropriate value)
            error=None              # no error
        )

    

