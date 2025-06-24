import os
import json
import logging
from typing import List, Optional
from openai import OpenAIError
import openai

from app.base.models import InterviewQuestion

# === Logging Setup ===
logger = logging.getLogger("prompt_templates")

# === OpenAI Config ===
openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")
DEFAULT_LANG = "uz"

# === Supported Languages ===
SUPPORTED_LANGUAGES = {
    "uz": "Uzbek",
    "ru": "Russian",
    "en": "English"
}

# === Prompt Templates ===

DYNAMIC_QUESTION_PROMPT = """
You are a multilingual AI interviewer assistant.

Generate 6–10 **interview questions** in {language_name} based on:
- Candidate's resume
- Job description

Cover both general and domain-specific aspects:
- Soft skills
- Communication & leadership
- Experience
- Motivation & culture fit
- Problem-solving
- Role relevance

Respond ONLY in valid JSON format:
[
  {{
    "text": "question text",
    "type": "general|behavioral|situational|technical",
    "tags": ["tag1", "tag2"]
  }},
  ...
]
Do not include explanations or extra text. Output JSON only.
Resume:
{resume}

Job Description:
{job_description}
"""

SESSION_SUMMARY_TEMPLATE = """
Generate a concise summary of the candidate's interview session based on the following answers.

Summarize in {language_name}. Include:
- Strengths and weaknesses
- Communication style
- Key skills mentioned
- Personality traits
- Overall performance

Format in clear paragraphs.

Answers:
{answers}
Domain: {domain}
Level: {level}
"""

SKILL_EXTRACTION_TEMPLATE = """
Extract a structured list of skills and personality traits demonstrated in the following interview transcript.

Job Context: {job_context}

Transcript:
{full_text}

Respond in JSON as:
{{
  "skills": ["skill1", "skill2", ...],
  "traits": ["trait1", "trait2", ...],
  "intent": "string (inferred candidate intent)"
}}
Only output JSON.
"""

# === Uzbek Fallback ===
UZBEK_STATIC_QUESTIONS = [
    InterviewQuestion(text="Iltimos, o'zingiz haqingizda qisqacha tanishtiring.", type="general", tags=["intro"]),
    InterviewQuestion(text="Sizning asosiy kuchli va zaif tomonlaringiz qanday?", type="behavioral", tags=["strengths", "weaknesses"]),
    InterviewQuestion(text="Qiyin muammoni qanday hal qilganingizni misol keltiring.", type="situational", tags=["problem-solving"]),
    InterviewQuestion(text="Bu lavozim sizni nega qiziqtiryapti?", type="motivation", tags=["intent"]),
    InterviewQuestion(text="Jamoada qanday ishlaysiz?", type="behavioral", tags=["teamwork"]),
    InterviewQuestion(text="Oxirgi loyihangizda qanday mas'uliyatlar bo'lgan?", type="technical", tags=["experience", "project"]),
]

# === Dynamic Question Generator ===
def generate_dynamic_questions(
    resume: str,
    job_description: Optional[str] = None,
    model: str = LLM_MODEL,
    language: str = DEFAULT_LANG,
    temperature: float = 0.5,
    fallback: bool = True
) -> List[InterviewQuestion]:
    lang_code = language.lower()
    language_name = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES[DEFAULT_LANG])
    prompt = DYNAMIC_QUESTION_PROMPT.format(
        language_name=language_name,
        resume=resume.strip() or "N/A",
        job_description=job_description.strip() or "N/A"
    )

    try:
        logger.info(f"[LLM] Generating questions in {language_name}")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are a multilingual structured interview generator in {language_name}."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )
        raw_json = response.choices[0].message["content"].strip()
        parsed = json.loads(raw_json)

        questions = []
        for q in parsed:
            if not q.get("text"):
                continue
            questions.append(
                InterviewQuestion(
                    text=q["text"].strip(),
                    type=q.get("type", "general"),
                    tags=q.get("tags", [])
                )
            )

        if not questions:
            raise ValueError("No questions returned from LLM")

        return questions

    except (OpenAIError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[Fallback] LLM failed or returned invalid: {e}")
        if fallback:
            logger.info("[Fallback] Returning static Uzbek questions")
            return UZBEK_STATIC_QUESTIONS
        return []
