"""
GPTWriter: Unified abstraction over OpenAI GPT APIs for prompt-to-text generation.

Supports:
- GPT-3.5, GPT-4 (via openai.ChatCompletion)
- Extendable to Claude, Mistral, local models

Usage:
    writer = GPTWriter(model="gpt-4")
    output = writer.write("Tell me a joke.")
"""

import os
import openai
import logging
from typing import Optional, List

logger = logging.getLogger("gpt_writer")


class GPTWriter:
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY environment variable.")

        openai.api_key = self.api_key

    def write(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Sends a prompt to the GPT model and returns the response.

        Args:
            prompt (str): User prompt
            system_prompt (Optional[str]): Optional system message (e.g. "You are a helpful assistant.")

        Returns:
            str: Model-generated text
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            logger.info(f"[GPTWriter] Sending prompt to {self.model}")

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            result = response["choices"][0]["message"]["content"].strip()
            return result

        except openai.error.OpenAIError as e:
            logger.exception(f"[GPTWriter] OpenAI API error: {e}")
            return "⚠️ GPT response unavailable due to API error."
        except Exception as e:
            logger.exception(f"[GPTWriter] Unexpected error: {e}")
            return "⚠️ GPT encountered an unexpected error."

    def write_batch(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """
        Process a list of prompts sequentially using the same model config.

        Returns:
            List of responses
        """
        results = []
        for prompt in prompts:
            results.append(self.write(prompt, system_prompt))
        return results
