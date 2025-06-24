import os
import logging
import hashlib
from typing import List, Tuple, Dict

from app.models.whisper_wrapper import WhisperTranscriber
from app.models.gpt_wrapper import GPTScorer
from app.base.utils.interview_templates import (
    load_questions,
    SESSION_SUMMARY_TEMPLATE,
    SKILL_EXTRACTION_TEMPLATE,
    PROFILE_EXTRACTION_TEMPLATE,
)
from app.base.models import (
    CandidateAudioInput,
    InterviewQuestion,
    InterviewAnswer,
    InterviewResult,
    SkillExtraction,
    CandidateProfile,
)

logger = logging.getLogger("interview_service")


class InterviewService:
    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.llm_scorer = GPTScorer()
        self.transcript_cache: Dict[str, Tuple[str, float]] = {}

    def _hash_audio(self, path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _transcribe_audio(self, path: str, lang: str = "auto") -> Tuple[str, float]:
        checksum = self._hash_audio(path)
        if checksum in self.transcript_cache:
            logger.debug(f"[Cache] Using cached transcript for: {path}")
            return self.transcript_cache[checksum]

        transcript, confidence = self.transcriber.transcribe(path, language=lang)
        self.transcript_cache[checksum] = (transcript, confidence)
        return transcript, confidence

    def process_interview(self, input_data: CandidateAudioInput) -> InterviewResult:
        try:
            logger.info(f"[InterviewService] Starting interview for {input_data.candidate_id}")
            language = input_data.language or "uz"

            # === Step 1: Load Questions ===
            questions: List[InterviewQuestion] = load_questions(
                domain=input_data.domain,
                difficulty=input_data.difficulty,
                type=input_data.interview_type or "mixed"
            )
            if not questions:
                raise ValueError("No questions loaded for given parameters.")

            logger.info(f"[InterviewService] Loaded {len(questions)} questions")

            # === Step 2: Transcribe Answers ===
            transcripts = []
            for idx, audio_path in enumerate(input_data.audio_paths):
                transcript, confidence = self._transcribe_audio(audio_path, lang=language)
                logger.info(f"[Transcription] Q{idx + 1} confidence={confidence:.2f}")
                if not transcript.strip():
                    logger.warning(f"[EmptyTranscript] Q{idx + 1} returned empty transcript.")
                transcripts.append((transcript.strip(), confidence))

            if len(transcripts) != len(questions):
                raise ValueError("Mismatch between number of questions and answers")

            # === Step 3: Score Answers ===
            answer_objs: List[InterviewAnswer] = []
            for i, (answer_text, confidence) in enumerate(transcripts):
                try:
                    score_result = self.llm_scorer.score_answer(questions[i].text, answer_text)
                except Exception as e:
                    logger.warning(f"[ScoreFallback] Q{i + 1}: {e}")
                    score_result = {
                        "score": 0,
                        "reasoning": "Scoring failed",
                        "tags": [],
                        "model_used": "fallback"
                    }

                answer_objs.append(InterviewAnswer(
                    question=questions[i].text,
                    answer=answer_text,
                    score=score_result["score"],
                    reasoning=score_result["reasoning"],
                    language=language,
                    transcription_confidence=confidence,
                    tags=score_result.get("tags", []),
                    metadata={
                        "question_type": questions[i].type,
                        "source": "llm_score",
                        "model": score_result.get("model_used")
                    }
                ))

            full_text = "\n".join([f"A: {a.answer}" for a in answer_objs])

            # === Step 4: Extract Profile ===
            try:
                profile_prompt = PROFILE_EXTRACTION_TEMPLATE.format(full_text=full_text)
                profile_data = self.llm_scorer.extract_profile(profile_prompt)
                extracted_profile = CandidateProfile(
                    expected_salary=profile_data.get("expected_salary"),
                    total_experience=profile_data.get("total_experience"),
                    notable_projects=profile_data.get("notable_projects", []),
                    preferred_stack=profile_data.get("preferred_stack", []),
                    relocation_interest=profile_data.get("relocation_interest"),
                    current_role=profile_data.get("current_role")
                )
            except Exception as e:
                logger.warning(f"[ProfileFallback] Profile extraction failed: {e}")
                extracted_profile = CandidateProfile()

            # === Step 5: Extract Skills & Intent ===
            try:
                skill_prompt = SKILL_EXTRACTION_TEMPLATE.format(
                    full_text=full_text,
                    job_context=input_data.domain or "general"
                )
                skills_result = self.llm_scorer.extract_skills(skill_prompt)
                extracted_skills = SkillExtraction(
                    extracted_skills=skills_result.get("skills", []),
                    inferred_intent=skills_result.get("intent"),
                    personality_traits=skills_result.get("traits", []),
                    language_used=language
                )
            except Exception as e:
                logger.warning(f"[SkillFallback] Skill extraction failed: {e}")
                extracted_skills = SkillExtraction(language_used=language)

            # === Step 6: Generate Summary ===
            try:
                summary_prompt = SESSION_SUMMARY_TEMPLATE.format(
                    answers="\n".join([f"Q: {a.question}\nA: {a.answer}" for a in answer_objs]),
                    domain=input_data.domain or "general",
                    level=input_data.difficulty or "mixed"
                )
                session_summary = self.llm_scorer.get_summary(summary_prompt)
            except Exception as e:
                logger.warning(f"[SummaryFallback] Summary generation failed: {e}")
                session_summary = "Summary could not be generated."

            logger.info(f"[InterviewService] Completed session for {input_data.candidate_id}")

            return InterviewResult(
                candidate_id=input_data.candidate_id,
                domain=input_data.domain,
                difficulty=input_data.difficulty,
                answers=answer_objs,
                skill_extraction=extracted_skills,
                session_summary=session_summary,
                profile=extracted_profile
            )

        except Exception as e:
            logger.exception(f"[InterviewService Error] {e}")
            raise RuntimeError(f"Interview processing failed: {e}")
