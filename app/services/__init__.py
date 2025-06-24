"""
HirefyAI Services Module

This module provides centralized access to all core microservices used in the platform.
Each service is responsible for a specific domain task such as parsing resumes,
matching candidates to jobs, conducting interviews, scoring candidates, and more.
"""

# === Resume Processing Services ===
from .resume_parser_service import ResumeParserService
from .resume_generator_service import ResumeGeneratorService

# === Matching Engines ===
from .job_matcher_service import JobMatcherService
from .candidate_matcher_service import CandidateMatcherService

# === Interview Automation ===
from .interview_service import InterviewService
from .interview_stream_service import InterviewStreamService
from .interview_scheduler_service import InterviewSchedulerService

# === Psychometric & Scoring Engines ===
from .psychometrics_service import PsychometricsService
from .scoring_service import ScoringService

# === Language & LLM Copilot Services ===
from .copilot_service import CopilotService

# === Fairness & Explainability ===
from .audit_explainer_service import AuditExplainerService

# === Embedding & Vector Store ===
from .embedding_store_service import EmbeddingStoreService

# === Exported Interface ===
__all__ = [
    "ResumeParserService",
    "ResumeGeneratorService",
    "JobMatcherService",
    "CandidateMatcherService",
    "InterviewService",
    "InterviewStreamService",
    "InterviewSchedulerService",
    "PsychometricsService",
    "ScoringService",
    "CopilotService",
    "AuditExplainerService",
    "EmbeddingStoreService"
]
