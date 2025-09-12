from pathlib import Path
from huggingface_hub import HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.names import *


class LlamaWrapper:
    """
    LLaMA-2 Chat wrapper with:
      - correct chat templating
      - left padding for efficient batched generation
      - forward-hook context (attach to model.model.layers[i])
      - greedy, token-by-token generation that respects hooks
    """

    def __init__(self,
                 device_map="auto",
                 torch_dtype=torch.float16):
        """
        Initialize Llama-7b-chat-hf model with tokenizer.
        """
        self.model_path = LLAMA_2_7B

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=HfFolder.get_token(), use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                          device_map=device_map,
                                                          torch_dtype=torch_dtype).eval()

        self.device = self.model.device

    def reformat_prompt(self, user_text: str) -> str:
        """
        Reformats a prompt. Adds chat template and assign role.
        Format: "[INST] {prompt} [/INST] "
        :param user_text: raw instruction from user.
        :return: reformated prompt
        """
        massage = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(massage, tokenize=False, add_generation_prompt=True)

    def to_tokens(self, user_text: str):
        """
        Reformats and tokenize a user prompt with the Llama-2 tokenizer.
        :param user_text: raw instruction from user.
        :return: 1. input_ids: Tokenized prompt IDs.
                 2. attn_mask: Attention mask aligned with input_ids.
                 3. prompt_len: Number of non-pad tokens.
                    * steering vectors affect only new tokens.
        """
        text = self.reformat_prompt(user_text)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)

        prompt_len = int(attn_mask[0].sum().item())

        return input_ids, attn_mask, prompt_len

    def generate_once(self, input_ids, attn_mask, max_new_tokens=128):
        """
        Generate one completion from Llama-2-7b-chat-hf using greedy decoding.
        :param input_ids: Tokenized prompt IDs.
        :param attn_mask: Attention mask aligned with input_ids.
        :param max_new_tokens:  Maximum number of new tokens to generate.
        :return: 1. decoded_text: The decoded response from Llama-2-7b-chat-hf.
                 2. prompt_len: The tokenized prompt length (start position of generation)

        Notes: prompt_len is used downstream to apply steering only to generated tokens
               (positions >= prompt_len) while leaving the prompt unmodified.
        """
        with torch.inference_mode():
            out_ids = self.model.generate(input_ids=input_ids,
                                          attention_mask=attn_mask,
                                          do_sample=False,
                                          temperature=None,
                                          top_p=None,
                                          max_new_tokens=max_new_tokens,
                                          pad_token_id=self.tokenizer.eos_token_id)

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    @staticmethod
    def load_sv(path: Path,
                dtype=torch.float32,
                map_location="cpu") -> torch.Tensor:
        """
        loads a steering vector from a path, normalize it and allocate contiguous space in memory.
        :param path: path to steering vector.
        :param dtype: data type convertion for steering vector.
        :param map_location: cpu or cuda
        :return: tensor of size [4096]
        """
        r = torch.load(str(path), weights_only=True, map_location=map_location)
        r = r.to(dtype)
        r /= r.norm()

        return r.contiguous()
