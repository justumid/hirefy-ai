import logging
from typing import Dict, Literal, Optional, Tuple, List
import os

from app.base.models import CopilotRequest, CopilotResponse
from app.models.gpt_wrapper import GPTWriter
from app.base.utils.prompt_loader import load_prompt_template, render_prompt

logger = logging.getLogger("copilot_service")

# Supported prompt types and their file paths
PROMPT_PATHS: Dict[str, str] = {
    "job_description": "app/prompts/jd_prompt.txt",
    "email": "app/prompts/email_offer_prompt.txt",
    "offer_letter": "app/prompts/email_offer_prompt.txt",  # Could be split if needed
    "rejection_letter": "app/prompts/rejection_letter.txt",  # Optional, add file if needed
}


class CopilotService:
    def __init__(self):
        self.llm = GPTWriter()

    def generate(self, request: CopilotRequest) -> CopilotResponse:
        """
        Generate content for copilot tasks such as JD, offer letters, emails, etc.
        """
        try:
            logger.info(
                f"[Copilot] Generating: type={request.type} | tone={request.tone} | lang={request.language}"
            )

            # === Step 1: Build Prompt ===
            prompt, tags = self._build_prompt(request)

            # === Step 2: LLM Call ===
            model = request.model or "gpt-4-turbo"
            content = self.llm.write(prompt, model=model)

            logger.info(f"[Copilot] LLM generation complete. Model used: {model}")
            return CopilotResponse(
                type=request.type,
                content=content.strip(),
                language=request.language or "en",
                tags=tags,
                model_used=model
            )

        except Exception as e:
            logger.exception(f"[Copilot Error] Failed to generate {request.type}")
            raise RuntimeError(f"Copilot generation failed: {e}")

    def _build_prompt(self, req: CopilotRequest) -> Tuple[str, List[str]]:
        """
        Load the appropriate prompt template and render it with user-provided inputs.
        Returns the final prompt string and list of tags (keys from inputs).
        """
        prompt_path = PROMPT_PATHS.get(req.type)
        if not prompt_path or not os.path.exists(prompt_path):
            raise ValueError(f"Unsupported or missing prompt file for type: {req.type}")

        raw_template = load_prompt_template(prompt_path)

        # Safely render using user context
        context = {
            **(req.inputs or {}),
            "tone": req.tone or "formal",
            "language": req.language or "en"
        }

        prompt = render_prompt(raw_template, context)
        return prompt, list(context.keys())
