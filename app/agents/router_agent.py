# app/agents/router_agent.py

from app.utils.logger import get_logger
from app.graph.types import State

logger = get_logger(__name__)

class RouterAgent:

    def run(self, state: State) -> State:
        logger.info(f"Running RouterAgent with state: {state}")
        payload = state.payload

        icd_terms = [
            "icd10", "icd-10", "icd 10", "icd10 code",
            "icd-10 code", "icd 10 code", "icd"
        ]

        note = payload.get("note") or payload.get("clinical_note")
        image = payload.get("image")

        if image:
            state.type= "image_analysis"
            state.payload = {
                "image": image,
                "clinical_note": note
            }
        elif note:
            note_lower = note.lower()
            if any(term in note_lower for term in icd_terms):
                state.type = "icd10"
                state.payload = {"clinical_note": note}
            else:
                state.type = "soap"
                state.payload = {"transcript": note}
        else:
            state.error = "Unable to determine type from input"

        logger.info(f"RouterAgent determined next agent with state: {state}")
        return state
