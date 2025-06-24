from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid

from app.services.interview_scheduler_service import (
    get_db, calendar,
    InterviewSlotModel, SessionLocal
)

router = APIRouter(prefix="/interview_scheduler", tags=["Interview Scheduler"])

# === Pydantic Schemas ===
class CreateSlotRequest(BaseModel):
    interviewer_id: str
    start_datetime_utc: str
    duration_minutes: int = Field(default=30, ge=15, le=120)

class SlotBookingRequest(BaseModel):
    candidate_id: str
    candidate_email: str
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
    created_at: str

# === API Endpoints ===

@router.post("/create_slot", response_model=SlotResponse)
def create_slot(req: CreateSlotRequest, db: Session = Depends(get_db)):
    dt = datetime.fromisoformat(req.start_datetime_utc)
    slot_id = str(uuid.uuid4())
    calendar_event_id = calendar.create_event(req.interviewer_id, dt.isoformat(), req.duration_minutes, "TBD")

    slot = InterviewSlotModel(
        slot_id=slot_id,
        interviewer_id=req.interviewer_id,
        start_datetime_utc=dt,
        duration_minutes=req.duration_minutes,
        booked=False,
        calendar_event_id=calendar_event_id
    )
    db.add(slot)
    db.commit()
    db.refresh(slot)
    return SlotResponse(**slot.__dict__)


@router.get("/list_slots/{interviewer_id}", response_model=List[SlotResponse])
def list_slots(interviewer_id: str, db: Session = Depends(get_db)):
    records = db.query(InterviewSlotModel).filter_by(interviewer_id=interviewer_id, booked=False).all()
    return [SlotResponse(**r.__dict__) for r in records]


@router.post("/book")
def book_slot(req: SlotBookingRequest, db: Session = Depends(get_db)):
    slot = db.query(InterviewSlotModel).filter_by(slot_id=req.slot_id).first()
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    if slot.booked:
        raise HTTPException(status_code=409, detail="Slot already booked")

    slot.booked = True
    slot.candidate_id = req.candidate_id
    db.commit()

    from app.services.interview_scheduler_service import send_email
    send_email(req.candidate_email, slot.interviewer_id, slot.start_datetime_utc.isoformat())

    return {"status": "booked", "slot_id": slot.slot_id}


@router.delete("/cancel")
def cancel_slot(req: SlotCancelRequest, db: Session = Depends(get_db)):
    slot = db.query(InterviewSlotModel).filter_by(slot_id=req.slot_id).first()
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    if slot.calendar_event_id:
        calendar.delete_event(slot.calendar_event_id)
    db.delete(slot)
    db.commit()
    return {"status": "cancelled", "slot_id": req.slot_id}


@router.patch("/reschedule")
def reschedule_slot(req: RescheduleRequest, db: Session = Depends(get_db)):
    slot = db.query(InterviewSlotModel).filter_by(slot_id=req.slot_id).first()
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")

    new_dt = datetime.fromisoformat(req.new_start_datetime_utc)
    if slot.calendar_event_id:
        calendar.update_event(slot.calendar_event_id, new_dt.isoformat(), req.new_duration_minutes or slot.duration_minutes)

    slot.start_datetime_utc = new_dt
    if req.new_duration_minutes:
        slot.duration_minutes = req.new_duration_minutes
    db.commit()
    return {"status": "rescheduled", "slot_id": req.slot_id}
