from app.agents.base_agent import BaseAgent
from app.utils.model_loader import load_medgemma_model
from app.utils.prompt_builder import build_image_analyzer_prompt
from app.utils.predictor import generate_response
from PIL import Image
from app.graph.types import State
import requests
from app.utils.logger import get_logger
from app.utils.helper import clean_json_response
import json

logger = get_logger(__name__)

class ImageAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ImageAnalyzerAgent")
        self.model, self.processor = load_medgemma_model()

    def respond(self, state: dict) -> str:
        image = state.payload["image"] if "image" in state.payload else None
        note = state.payload.get("note", None)
        logger.info(f"Generating image analysis for image: {image} with note: {note}")
        messages = build_image_analyzer_prompt(image, note)
        return generate_response(self.model, self.processor, messages)
    
    def run(self, state: State) -> State:
        """
        Run the agent with the provided image.
        """
        logger.info(f"Running {self.name} with state: {state}")
        raw_result = self.respond(state)

        logger.info("image analysis agent response: %s", raw_result)
        parsed_result = clean_json_response(raw_result)
        cleaned_result = json.loads(parsed_result)
            
        logger.info("Cleaned result: %s", cleaned_result)
        return State(
            type="image_analysis",
            payload=state.payload,  # preserve existing payload
            result=cleaned_result,   # add new result
            error=None              # no error
        )
    
# if __name__ == "__main__":
#     agent = ImageAnalyzerAgent()
#     image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
#     image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)
#     response = agent.respond(image)
#     print(response)  