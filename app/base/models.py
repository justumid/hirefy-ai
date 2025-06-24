from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime



class AuditExplanationResult(BaseModel):
    final_score: float
    explanation: Dict[str, float]
    top_contributors: List[str]
    fairness_flags: Optional[Dict[str, str]] = None
    summary: Optional[str]

class EmbeddingRecord(BaseModel):
    text: str = Field(..., description="The raw text that was embedded")
    type: str = Field(..., description="Type/category of the embedding (e.g. resume, skill, psychometrics)")
    metadata: Optional[Dict[str, Union[str, int, float]]] = Field(default_factory=dict, description="Optional metadata")


class EmbeddingSearchResult(BaseModel):
    id: int = Field(..., description="FAISS internal ID or record index")
    score: float = Field(..., description="Cosine similarity score")
    payload: Dict[str, Union[str, int, float]] = Field(..., description="Metadata or content returned with the match")


class JobPosting(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Full job description text")
    required_skills: List[str] = Field(..., description="List of required or preferred skills")
    location: Optional[str] = Field(None, description="Job location, e.g., city or remote")
    created_at: Optional[str] = Field(None, description="ISO datetime string when the job was posted")


class ResumeProfile(BaseModel):
    candidate_id: str = Field(..., description="Unique candidate identifier")
    resume_text: str = Field(..., description="Full parsed resume text")
    skills: List[str] = Field(..., description="Extracted or user-provided skill list")
    language: Optional[str] = Field("auto", description="Resume language if known (auto-detected by default)")


class ResumeGenerationRequest(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    full_name: str = Field(..., description="Candidate's full name")
    contact_email: Optional[str] = Field(None, description="Email address")
    phone_number: Optional[str] = Field(None, description="Phone number")
    summary: Optional[str] = Field(None, description="Short personal summary or objective")
    education: List[Dict] = Field(..., description="List of education entries, each with degree, institution, years")
    experience: List[Dict] = Field(..., description="List of work experience entries with role, company, duration, responsibilities")
    skills: List[str] = Field(..., description="List of skills relevant to the job")
    languages: Optional[List[str]] = Field(None, description="Languages spoken or written")
    target_job_title: Optional[str] = Field(None, description="The position the resume is optimized for")
    preferred_format: Optional[str] = Field("markdown", description="Output format (markdown, plain, html)")


class ResumeGenerationResult(BaseModel):
    candidate_id: str = Field(..., description="Candidate ID for reference")
    generated_resume: str = Field(..., description="Final resume text in the requested format")
    summary_embedding: Optional[List[float]] = Field(None, description="Optional embedding vector for similarity search or clustering")
    generation_metadata: Optional[Dict] = Field(None, description="Metadata such as model used, prompt version, or generation time")

# === 📊 Psychometric Input & Output ===

class PsychometricsInput(BaseModel):
    candidate_id: str
    responses: Dict[str, str]  # e.g., {"q1": "Strongly Agree", ...}
    language: Optional[str] = "en"


class TraitScore(BaseModel):
    trait: str  # e.g., "Openness", "Conscientiousness"
    score: float  # 0.0 - 1.0
    description: Optional[str] = None


class PsychometricsResult(BaseModel):
    candidate_id: str
    personality_type: Optional[str]
    trait_scores: List[TraitScore]
    summary: Optional[str]
    recommendations: Optional[List[str]] = None


class EmbeddingMetadata(BaseModel):
    id: str
    source: str  # e.g., "psychometrics", "resume", etc.
    tags: Optional[List[str]]
    extra: Optional[Dict[str, str]] = None

# === 🔍 Resume Parsing ===
class ResumeParseRequest(BaseModel):
    job_id: Optional[str] = None
    resume_text: Optional[str] = None
    resume_file_url: Optional[str] = None


class ResumeData(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    skills: List[str]
    experience: List[Dict[str, Union[str, int, float]]]
    education: List[Dict[str, Union[str, int]]]
    summary: Optional[str]
    language: Optional[str] = "auto"
    extracted_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


# === ✨ Resume Generation ===
class ResumeGenRequest(BaseModel):
    candidate_name: str
    profile_summary: Optional[str]
    experience: List[Dict[str, Union[str, int]]]
    education: List[Dict[str, Union[str, int]]]
    skills: List[str]
    job_title: Optional[str]
    language: Optional[str] = "uz"


# === 🤝 Job Matching ===
class MatchRequest(BaseModel):
    resume_text: str
    job_description: str
    language: Optional[str] = "auto"


class MatchResult(BaseModel):
    similarity_score: float
    top_keywords: List[str]
    matched_sections: Optional[List[str]]
    explanation: Optional[str]


# === 🧠 Copilot (Prompt-Based LLM Content) ===
class CopilotRequest(BaseModel):
    type: str  # e.g. "job_description", "offer_letter", "email"
    inputs: Optional[Dict[str, Union[str, int, float, list, dict]]] = {}
    tone: Optional[str] = "formal"
    language: Optional[str] = "en"
    model: Optional[str] = "gpt-4-turbo"


class CopilotResponse(BaseModel):
    type: str
    content: str
    language: str
    tags: Optional[List[str]] = []
    model_used: Optional[str]
    reasoning: Optional[str] = None


# === 📊 Psychometric Scoring ===
class PsychometricInput(BaseModel):
    candidate_id: str
    responses: Dict[str, Union[int, str, float]]


class PsychometricScore(BaseModel):
    candidate_id: str
    personality_type: str
    trait_scores: Dict[str, float]
    summary: Optional[str]


# === 📉 Explainability / SHAP ===
class ExplanationRequest(BaseModel):
    input_features: Dict[str, Union[str, float, int]]
    model_name: Optional[str] = "default_model"


class ExplanationResult(BaseModel):
    shap_values: Dict[str, float]
    top_features: List[str]
    explanation_plot_url: Optional[str]


# === 🧠 Embedding Store ===
class EmbeddingInput(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, str]] = None


class EmbeddingQuery(BaseModel):
    query: str
    top_k: int = 5


class EmbeddingResult(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, str]]


# === 🧪 Interview Bot ===
class InterviewQuestion(BaseModel):
    text: str
    type: Optional[str] = "general"


class InterviewAnswer(BaseModel):
    question: str
    answer: str
    score: float
    reasoning: Optional[str]
    language: Optional[str]
    transcription_confidence: Optional[float]
    tags: Optional[List[str]]
    metadata: Optional[Dict[str, Union[str, float]]]


class CandidateAudioInput(BaseModel):
    candidate_id: str
    domain: str
    difficulty: Optional[str] = "mixed"
    interview_type: Optional[str] = "mixed"
    language: Optional[str] = "auto"
    audio_paths: List[str]


class SkillExtraction(BaseModel):
    extracted_skills: List[str]
    inferred_intent: Optional[str]
    personality_traits: List[str]
    language_used: str


class CandidateProfile(BaseModel):
    expected_salary: Optional[str]
    total_experience: Optional[str]
    notable_projects: List[str]
    preferred_stack: List[str]
    relocation_interest: Optional[str]
    current_role: Optional[str]


class InterviewResult(BaseModel):
    candidate_id: str
    domain: str
    difficulty: str
    answers: List[InterviewAnswer]
    skill_extraction: Optional[SkillExtraction]
    session_summary: Optional[str]
    profile: Optional[CandidateProfile]


# === 🟢 Streamed Interview Init ===
class CandidateStreamInit(BaseModel):
    candidate_id: str
    domain: str
    difficulty: Optional[str] = "mixed"
    language: Optional[str] = "auto"


class StreamInterviewConfig(BaseModel):
    candidate_id: str
    domain: str
    difficulty: Optional[str] = "mixed"
    interview_type: Optional[str] = "mixed"
    language: Optional[str] = "auto"


# === 📈 Final Scoring ===
class FinalScoreOutput(BaseModel):
    candidate_id: str
    fraud_score: Optional[float]
    pd_score: Optional[float]
    limit_estimate: Optional[float]
    recommended_action: Optional[str]
    explanation: Optional[str]


# === 📁 File Upload ===
class FileUploadResponse(BaseModel):
    file_url: str
    checksum: Optional[str] = None


# === 🔐 Auth Response ===
class AuthResponse(BaseModel):
    api_key_valid: bool
    usage_remaining: Optional[int] = None


# === InterviewScheduler Related (Bonus) ===
class CreateSlotRequest(BaseModel):
    interviewer_id: str
    start_datetime_utc: str
    duration_minutes: int = Field(default=30, ge=15, le=120)
    interview_type: Optional[str] = "general"
    timezone: Optional[str] = "UTC"


class SlotBookingRequest(BaseModel):
    candidate_id: str
    candidate_email: EmailStr
    slot_id: str


class SlotCancelRequest(BaseModel):
    slot_id: str


class RescheduleRequest(BaseModel):
    slot_id: str
    new_start_datetime_utc: str
    new_duration_minutes: Optional[int]


class SlotResponse(BaseModel):
    slot_id: str
    interviewer_id: str
    start_datetime_utc: str
    duration_minutes: int
    booked: bool
    candidate_id: Optional[str]
    calendar_event_id: Optional[str]
    interview_type: Optional[str]
    timezone: Optional[str]


# === Forward Reference Fix ===
InterviewResult.update_forward_refs()
