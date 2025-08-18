from mlx_lm import generate
from app.utils.logger import get_logger

logger = get_logger(__name__)

def generate_response(model, processor, formatted_prompt, image):
    """
    Generate response using the MedGemma model.
    
    Args:
        model: The loaded MedGemma model.
        processor: The processor for the model.
        messages: The input messages for the model.
    
    Returns:
        str: The generated response.
    """
    logger.info("Generating response with prompt: %s", formatted_prompt)
    response = generate(model, processor, formatted_prompt, image)
    logger.info("Model outputs: %s", response)
    return response



