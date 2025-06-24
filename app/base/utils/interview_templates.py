import os
import random
from typing import List, Literal, Optional
from app.base.models import InterviewQuestion

# === Templates for Extraction ===
PROFILE_EXTRACTION_TEMPLATE = """
Given the following answers, extract the candidate's profile attributes as a JSON object:
- expected_salary
- total_experience
- notable_projects
- preferred_stack
- relocation_interest
- current_role

Answers:
{full_text}
"""

SKILL_EXTRACTION_TEMPLATE = """
Analyze the following candidate answers and extract:
- skills (list)
- intent (job search intent)
- traits (personality traits)

Answers:
{full_text}

Context: The candidate applied for a {job_context} position.
"""

SESSION_SUMMARY_TEMPLATE = """
Summarize the candidate's performance during the interview.

Domain: {domain}
Level: {level}

Answers:
{answers}
"""

# === Base Path ===
QUESTION_BANK_DIR = "app/prompts/interview_questions"

# === Type Aliases ===
QuestionType = Literal["hard", "soft", "case", "management", "mixed"]

# === Loader ===

def load_questions(
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
    type: QuestionType = "mixed",
    language: str = "en",
    count: int = 5
) -> List[InterviewQuestion]:
    """
    Load questions from local prompt files based on type.
    """
    type_to_file = {
        "soft": "soft_skills.txt",
        "hard": "hard_skills.txt",
        "case": "case_study.txt",
        "management": "management_skills.txt"
    }

    files_to_load = (
        list(type_to_file.values()) if type == "mixed"
        else [type_to_file.get(type)]
    )

    questions = []

    for filename in files_to_load:
        full_path = os.path.join(QUESTION_BANK_DIR, filename)
        if not os.path.exists(full_path):
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in lines:
                questions.append(InterviewQuestion(
                    text=line,
                    type=filename.replace("_skills.txt", ""),
                    domain=domain or "general",
                    difficulty=difficulty or "mixed",
                    language=language
                ))

    # Shuffle and return the desired count
    random.shuffle(questions)
    return questions[:count]
