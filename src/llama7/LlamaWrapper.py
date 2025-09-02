from typing import Callable, List, Tuple, Optional, Iterable, Dict
import contextlib
import torch

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

    def __init__(self):
        """
        Initialize Llama-7b-chat-hf model with tokenizer. Sets to eval mode.
        """
        self.device = DEVICE
        self.model_path = LLAMA_2_7B

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                          device_map="auto",
                                                          torch_dtype=torch.float16)
        self.model.eval()

    def build_prompt(self, instruction: str) -> str:
        """
        Wrap the instruction with chat template tokens.
        If tokenizer has "apply_chat_template" function, utilizing it. Else, wrap manually.
        :param instruction: Raw prompts from user.
        :return: Formated prompt.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": instruction}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return f"<s>[INST] {instruction} [/INST]"

    def tokenize_instructions(self, instructions: List[str]) -> torch.Tensor:
        """
        Build Llama-2 chat prompts for a batch of user instructions and tokenize them.
        :param instructions: Raw prompts from user.
        :return: A tensor of shape (batch_size, seq_len_max) containing token IDs.
        """
        prompts = []
        for instruction in instructions:
            prompts.append(self.build_prompt(instruction))

        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).input_ids
        tokens = tokens.to(self.device)

        return tokens

    @contextlib.contextmanager
    def hooks(self, fwd_hooks: Iterable[Tuple[int, Callable]] = ()):
        """
        Simple forward-hook manager (attach to transformer layers).
        fwd_hooks: iterable of (layer_index, hook_fn)
            where hook_fn has signature: hook_fn(module, inputs, output) -> output
        We attach to self.model.model.layers[layer_index]
        """
        handles = []
        try:
            for layer_idx, hook_fn in fwd_hooks:
                layer = self.model.model.layers[layer_idx]
                h = layer.register_forward_hook(hook_fn)
                handles.append(h)
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.inference_mode()
    def generate_with_hooks(self,
                            tokens: torch.Tensor,
                            fwd_hooks: Iterable[Tuple[int, Callable]] = (),
                            ) -> List[str]:
        """
        Token-by-token greedy generation that supports hooks
        tokens: Tensor [batch, seq_len], already on self.model.device
        Returns: list[str] of decoded completions (ONLY the newly generated tokens).
        """
        batch_size, seq_length = tokens.shape
        total_len = seq_length + MAX_NEW_TOKENS

        all_tokens = torch.full((batch_size, total_len),
                                self.tokenizer.pad_token_id,
                                dtype=torch.long,
                                device=self.device)
        all_tokens[:, :seq_length] = tokens

        for i in range(MAX_NEW_TOKENS):
            cur_len = seq_length + i
            with self.hooks(fwd_hooks=fwd_hooks):
                logits = self.model(input_ids=all_tokens[:, :cur_len]).logits
            next_ids = logits[:, -1, :].argmax(dim=-1)  # greedy
            all_tokens[:, cur_len] = next_ids

        # decode ONLY the generated tail for each batch item
        gens = self.tokenizer.batch_decode(all_tokens[:, seq_length:], skip_special_tokens=True)

        return gens


    def get_generations(self,
                        instructions: List[str],
                        fwd_hooks: Iterable[Tuple[int, Callable]] = (),
                        batch_size: int = 4,
                        ) -> List[str]:
        """
        Batched generation over instructions for convenience
        :param instructions: list of raw prompts from user.
        :param fwd_hooks:
        :param batch_size:
        :return:
        """
        out = []
        for i in range(0, len(instructions), batch_size):
            tokens = self.tokenize_instructions(instructions[i:i + batch_size])
            out.extend(self.generate_with_hooks(tokens, fwd_hooks=fwd_hooks))
        return out

