import asyncio
import hashlib
import logging
from typing import Dict, List, Optional

from app.models.whisper_wrapper import WhisperTranscriber
from app.models.gpt_wrapper import GPTScorer
from app.base.utils.interview_templates import (
    load_questions,
    SESSION_SUMMARY_TEMPLATE,
    SKILL_EXTRACTION_TEMPLATE,
)
from app.base.models import (
    InterviewResult,
    InterviewQuestion,
    InterviewAnswer,
    CandidateStreamInit,
    SkillExtraction,
)

logger = logging.getLogger("interview_stream_service")


class InterviewStreamSession:
    def __init__(self, candidate_id: str, domain: str, difficulty: str, language: str = "auto"):
        self.candidate_id = candidate_id
        self.domain = domain
        self.difficulty = difficulty
        self.language = language or "uz"
        self.transcriber = WhisperTranscriber()
        self.llm_scorer = GPTScorer()

        # Load static or predefined question set
        self.questions: List[InterviewQuestion] = load_questions(
            domain=self.domain,
            difficulty=self.difficulty,
            type="mixed",
            language=self.language,
            count=5
        )

        self.answers: List[InterviewAnswer] = []
        self.skill_extraction: Optional[SkillExtraction] = None
        self.session_summary: Optional[str] = None
        self._current_q_idx = 0
        self._buffer = b""
        self._max_questions = len(self.questions)
        self._recent_checksums = set()

        logger.info(f"[StreamInit] Started session for {self.candidate_id} | domain={self.domain} | level={self.difficulty} | Qs={self._max_questions}")

    def _compute_checksum(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get_next_question(self) -> Optional[str]:
        if self._current_q_idx < self._max_questions:
            return self.questions[self._current_q_idx].text
        return None

    async def handle_stream_chunk(self, audio_chunk: bytes) -> Optional[InterviewAnswer]:
        self._buffer += audio_chunk

        if len(self._buffer) < 16000 * 5:
            return None  # Wait until at least 5 seconds of audio is collected

        checksum = self._compute_checksum(self._buffer)
        if checksum in self._recent_checksums:
            logger.debug("[Stream] Duplicate buffer skipped")
            self._buffer = b""
            return None
        self._recent_checksums.add(checksum)

        try:
            transcript, confidence = self.transcriber.transcribe_from_bytes(
                self._buffer, language=self.language
            )
        except Exception as e:
            logger.warning(f"[Transcription Error] {e}")
            self._buffer = b""
            return None

        self._buffer = b""

        if not transcript.strip():
            logger.info("[Stream] Empty transcript segment.")
            return None

        if self._current_q_idx >= self._max_questions:
            logger.warning("[Stream] Overflow - received more answers than expected")
            return None

        current_q = self.questions[self._current_q_idx]

        try:
            score_result = self.llm_scorer.score_answer(current_q.text, transcript)
        except Exception as e:
            logger.warning(f"[LLM Fallback] Failed scoring: {e}")
            score_result = {
                "score": 0.0,
                "reasoning": f"LLM error: {e}",
                "tags": [],
                "model_used": "fallback"
            }

        answer = InterviewAnswer(
            question=current_q.text,
            answer=transcript,
            score=score_result["score"],
            reasoning=score_result["reasoning"],
            language=self.language,
            transcription_confidence=confidence,
            tags=score_result.get("tags", []),
            metadata={
                "question_type": current_q.type,
                "source": "llm_score",
                "model": score_result.get("model_used", "fallback")
            }
        )

        self.answers.append(answer)
        self._current_q_idx += 1

        logger.info(f"[Scored] Q{self._current_q_idx}/{self._max_questions} → score={score_result['score']:.2f}")
        return answer

    async def finalize(self) -> InterviewResult:
        logger.info(f"[Finalize] {self.candidate_id=} | Total Answers={len(self.answers)}")

        # === Extract Skills ===
        try:
            full_text = "\n".join([a.answer for a in self.answers])
            skill_prompt = SKILL_EXTRACTION_TEMPLATE.format(
                full_text=full_text,
                job_context=self.domain or "general"
            )
            skills_result = self.llm_scorer.extract_skills(skill_prompt)

            self.skill_extraction = SkillExtraction(
                extracted_skills=skills_result.get("skills", []),
                inferred_intent=skills_result.get("intent"),
                personality_traits=skills_result.get("traits", []),
                language_used=self.language
            )
        except Exception as e:
            logger.warning(f"[Skill Extraction Failed] {e}")
            self.skill_extraction = SkillExtraction(language_used=self.language)

        # === Generate Summary ===
        try:
            summary_prompt = SESSION_SUMMARY_TEMPLATE.format(
                answers="\n".join([f"Q: {a.question}\nA: {a.answer}" for a in self.answers]),
                domain=self.domain,
                level=self.difficulty,
            )
            self.session_summary = self.llm_scorer.get_summary(summary_prompt)
        except Exception as e:
            logger.warning(f"[Summary Generation Failed] {e}")
            self.session_summary = "Summary could not be generated."

        return InterviewResult(
            candidate_id=self.candidate_id,
            domain=self.domain,
            difficulty=self.difficulty,
            answers=self.answers,
            skill_extraction=self.skill_extraction,
            session_summary=self.session_summary,
        )


class InterviewStreamService:
    def __init__(self):
        self.sessions: Dict[str, InterviewStreamSession] = {}

    def start_session(self, init: CandidateStreamInit) -> InterviewStreamSession:
        session = InterviewStreamSession(
            candidate_id=init.candidate_id,
            domain=init.domain,
            difficulty=init.difficulty,
            language=init.language or "uz"
        )
        self.sessions[init.candidate_id] = session
        logger.info(f"[Start] Stream session started for {init.candidate_id}")
        return session

    def get_session(self, candidate_id: str) -> Optional[InterviewStreamSession]:
        return self.sessions.get(candidate_id)

    def end_session(self, candidate_id: str) -> Optional[InterviewResult]:
        session = self.sessions.pop(candidate_id, None)
        if session:
            try:
                result = asyncio.run(session.finalize())
                logger.info(f"[End] Session complete → {candidate_id}")
                return result
            except Exception as e:
                logger.exception(f"[Finalization Error] Failed for {candidate_id}: {e}")
        return None
