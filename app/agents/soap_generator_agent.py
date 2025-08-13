from app.agents.base_agent import BaseAgent
from app.utils.model_loader import load_medgemma_model
from app.utils.prompt_builder import build_soap_generator_prompt
from app.utils.predictor import generate_response
from app.graph.types import State
from PIL import Image
from typing import Optional
from app.utils.logger import get_logger
from app.utils.helper import clean_json_response
import json

logger = get_logger(__name__)

class SoapGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SoapGeneratorAgent")
        self.model, self.processor = load_medgemma_model()

    def respond(self, state: dict) -> str:
        logger.info(f"Called respond with state: {state}")
        print(f"State payload: {state.payload}")
        print(f"Transcript: {state.payload.get('transcript', 'No transcript found')}")
        transcript = state.payload["transcript"] if "transcript" in state.payload else ""
        image = state.payload.get("image", None)
        logger.info(f"Generating SOAP note for transcript: {transcript}")
        messages = build_soap_generator_prompt(transcript, image)
        return generate_response(self.model, self.processor, messages)
    
    def run(self, state: State) -> State:
        """
        Run the agent with the provided clinical note.
        """
        logger.info(f"Running {self.name} with state: {state}")
        raw_result = self.respond(state)

        logger.info("soap_generated agent response: %s", raw_result)
        parsed_result = clean_json_response(raw_result)
        cleaned_result = json.loads(parsed_result)
            
        logger.info("Cleaned result: %s", cleaned_result)
        return State(
            type="soap",
            payload=state.payload,  # preserve existing payload
            result=cleaned_result,   # add new result
            error=None               # no error
        )
    
# if __name__ == "__main__":
#     agent = SoapGeneratorAgent()
#     sample_note = "Patient presents with a headache and nausea. No significant findings on examination."
#     response = agent.respond(sample_note)
#     print(response)