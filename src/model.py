import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()


CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)


class ChatModel:
    def __init__(self, model_id: str = "google/gemma-2b-it", device="cuda"):

        ACCESS_TOKEN = os.getenv(
            "ACCESS_TOKEN"
        )  # reads .env file with ACCESS_TOKEN=<your hugging face access token>

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
        )
        self.model.eval()
        self.chat = []
        self.device = device

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):

        if context == None or context == "":
            prompt = f"""Give a detailed answer to the following question. Question: {question}"""
        else:
            prompt = f"""Using the information contained in the context, give a detailed answer to the question.
Context: {context}.
Question: {question}"""

        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(formatted_prompt)
        inputs = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response[len(formatted_prompt) :]  # remove input prompt from reponse
        response = response.replace("<eos>", "")  # remove eos token

        return response
