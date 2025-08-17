import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from app.utils.logger import get_logger

logger = get_logger(__name__)

def generate_response(model, processor, messages):
    """
    Generate ICD codes using the MedGemma model.
    
    Args:
        model: The loaded MedGemma model.
        processor: The processor for the model.
        messages: The input messages for the model.
    
    Returns:
        str: The generated response.
    """
    logger.info("Generating response with messages: %s", messages)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded


