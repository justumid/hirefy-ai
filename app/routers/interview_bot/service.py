import asyncio
import logging
from typing import List, Optional, Dict

from base.utils.interview_templates import (
    load_questions,
    SESSION_SUMMARY_TEMPLATE,
    SKILL_EXTRACTION_TEMPLATE,
)
from app.routers.interview_bot.transcription import transcribe_audio_stream
from app.routers.interview_bot.scoring import score_answer, summarize_session, extract_skills
from app.base.models import (
    InterviewSessionInit,
    InterviewQuestion,
    InterviewAnswer,
    InterviewResult,
    SkillExtraction,
)

logger = logging.getLogger("interview_bot_service")


class InterviewBotSession:
    def __init__(self, init: InterviewSessionInit):
        self.candidate_id = init.candidate_id
        self.resume = init.resume
        self.job_description = init.job_description
        self.language = init.language or "uz"
        self.domain = init.domain or "general"
        self.difficulty = init.difficulty or "intermediate"

        # Load static questions from prompt bank (predefined for hard/soft/case/management/mixed)
        self.questions: List[InterviewQuestion] = load_questions(
            domain=self.domain,
            difficulty=self.difficulty,
            type="mixed",
            language=self.language,
            count=5
        )

        self.answers: List[InterviewAnswer] = []
        self._current_idx = 0

        logger.info(f"[InterviewBot] Session initialized for {self.candidate_id} with {len(self.questions)} questions.")

    def get_next_question(self) -> Optional[InterviewQuestion]:
        if self._current_idx < len(self.questions):
            return self.questions[self._current_idx]
        return None

    async def handle_audio_chunk(self, chunk: bytes) -> Optional[InterviewAnswer]:
        transcript, confidence = transcribe_audio_stream(chunk, language=self.language)

        if not transcript.strip():
            logger.info("[Transcription] Empty response from audio chunk.")
            return None

        current_q = self.get_next_question()
        if not current_q:
            return None

        try:
            score_data = score_answer(question=current_q.text, answer=transcript, language=self.language)
        except Exception as e:
            logger.warning(f"[LLM Scoring Fallback] Failed to score Q{self._current_idx + 1}: {e}")
            score_data = {
                "score": 0.0,
                "reasoning": "Scoring unavailable",
                "tags": [],
                "model_used": "fallback"
            }

        answer = InterviewAnswer(
            question=current_q.text,
            answer=transcript,
            score=score_data["score"],
            reasoning=score_data.get("reasoning", ""),
            tags=score_data.get("tags", []),
            language=self.language,
            transcription_confidence=confidence,
            metadata={"model": score_data.get("model_used", "fallback")}
        )

        self.answers.append(answer)
        self._current_idx += 1
        return answer

    async def finalize(self) -> InterviewResult:
        full_text = "\n".join([a.answer for a in self.answers])

        # === Skill Extraction ===
        try:
            skill_prompt = SKILL_EXTRACTION_TEMPLATE.format(
                full_text=full_text,
                job_context=self.domain
            )
            skill_data = extract_skills(skill_prompt)
        except Exception as e:
            logger.warning(f"[Skill Extraction Fallback] Failed: {e}")
            skill_data = {}

        skill_extraction = SkillExtraction(
            extracted_skills=skill_data.get("skills", []),
            personality_traits=skill_data.get("traits", []),
            inferred_intent=skill_data.get("intent", ""),
            language_used=self.language
        )

        # === Session Summary ===
        try:
            answer_block = "\n".join([f"Q: {a.question}\nA: {a.answer}" for a in self.answers])
            summary_prompt = SESSION_SUMMARY_TEMPLATE.format(
                answers=answer_block,
                domain=self.domain,
                level=self.difficulty
            )
            summary = summarize_session(summary_prompt)
        except Exception as e:
            logger.warning(f"[Summary Generation Fallback] Failed: {e}")
            summary = "Session summary could not be generated."

        return InterviewResult(
            candidate_id=self.candidate_id,
            domain=self.domain,
            difficulty=self.difficulty,
            answers=self.answers,
            session_summary=summary,
            skill_extraction=skill_extraction
        )


# === 🎯 Session Manager ===

class InterviewBotService:
    def __init__(self):
        self.sessions: Dict[str, InterviewBotSession] = {}

    def start_session(self, init: InterviewSessionInit) -> InterviewBotSession:
        session = InterviewBotSession(init)
        self.sessions[init.candidate_id] = session
        logger.info(f"[InterviewBotService] Started session for {init.candidate_id}")
        return session

    def get_session(self, candidate_id: str) -> Optional[InterviewBotSession]:
        return self.sessions.get(candidate_id)

    def end_session(self, candidate_id: str) -> Optional[InterviewResult]:
        session = self.sessions.pop(candidate_id, None)
        if session:
            return asyncio.run(session.finalize())
        return None
