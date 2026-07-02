from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any


class QwenChatbot:
    """
    LLM class for Qwen
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initialise the class with Qwen3-0.6B as the model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history: list[dict[str, str]] = []

    def generate_response(self, user_input: str) -> Any:
        """
        Take an user input and return an answer.
        """
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=50)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(
            response_ids, skip_special_tokens=True)

        # Update history
        # self.history.append({"role": "user", "content": user_input})
        # self.history.append({"role": "assistant", "content": response})

        return response
