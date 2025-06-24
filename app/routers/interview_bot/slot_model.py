# app/routers/interview_bot/slot_model.py

from sqlalchemy import Column, String, Integer, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class InterviewSlotModel(Base):
    __tablename__ = "interview_slots"

    slot_id = Column(String, primary_key=True, index=True)
    interviewer_id = Column(String, nullable=False)
    candidate_id = Column(String, nullable=True)
    start_datetime_utc = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    booked = Column(Boolean, default=False)
    interview_type = Column(String, default="voice")  # voice, video, etc.
    calendar_event_id = Column(String, nullable=True)
    timezone = Column(String, default="UTC")
