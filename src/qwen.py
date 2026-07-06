from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
import torch


class QwenChatbot:
    """
    LLM class for Qwen
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Load the tokenizer and model from HuggingFace.

        Args:
            model_name: HuggingFace model identifier.
                Defaults to Qwen/Qwen3-0.6B.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
            )
        self.model.eval()
        self.history: list[dict[str, str]] = []

    def generate_response(self, user_input: str) -> Any:
        """
        Generate a response for a given user input.

        Args:
            user_input: The prompt string to send to the model.

        Returns:
            The generated response as a string.
        """
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            response_ids = self.model.generate( # type: ignore[misc]
                **inputs,
                max_new_tokens=200)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(
            response_ids, skip_special_tokens=True)

        # Update history
        # self.history.append({"role": "user", "content": user_input})
        # self.history.append({"role": "assistant", "content": response})

        return response
