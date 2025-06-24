# app/services/interview_scheduler_service.py

import logging
import uuid
from typing import List, Optional
from datetime import datetime

from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.base.models import (
    CreateSlotRequest,
    SlotBookingRequest,
    SlotCancelRequest,
    RescheduleRequest,
    SlotResponse,
)
from app.routers.interview_bot.email_utils import send_confirmation_email
from app.routers.interview_bot.calendar_utils import GoogleCalendarClient
from app.routers.interview_bot.slot_model import InterviewSlotModel
logger = logging.getLogger("interview_scheduler_service")


class InterviewSchedulerService:
    """
    Handles the creation, booking, cancellation, and rescheduling of interview slots,
    with Google Calendar sync and optional email notifications.
    """

    def __init__(self, calendar_client: Optional[GoogleCalendarClient] = None):
        self.calendar = calendar_client or GoogleCalendarClient()

    def create_slot(self, req: CreateSlotRequest, db: Session) -> SlotResponse:
        start_dt = datetime.fromisoformat(req.start_datetime_utc)
        slot_id = str(uuid.uuid4())

        try:
            calendar_event_id = self.calendar.create_event(
                interviewer_id=req.interviewer_id,
                start_time=start_dt.isoformat(),
                duration_min=req.duration_minutes,
                candidate_id="TBD"
            )
        except Exception as e:
            logger.error(f"[Calendar] Failed to create event: {e}")
            raise HTTPException(status_code=500, detail="Calendar event creation failed")

        slot = InterviewSlotModel(
            slot_id=slot_id,
            interviewer_id=req.interviewer_id,
            start_datetime_utc=start_dt,
            duration_minutes=req.duration_minutes,
            booked=False,
            calendar_event_id=calendar_event_id,
            interview_type=req.interview_type,
            timezone=req.timezone or "UTC",
        )

        db.add(slot)
        db.commit()
        db.refresh(slot)

        logger.info(f"[Create] Slot created: {slot_id} at {start_dt}")
        return SlotResponse(**slot.__dict__)

    def list_slots(self, interviewer_id: str, db: Session) -> List[SlotResponse]:
        slots = db.query(InterviewSlotModel).filter_by(
            interviewer_id=interviewer_id,
            booked=False
        ).order_by(InterviewSlotModel.start_datetime_utc.asc()).all()

        logger.info(f"[List] {len(slots)} open slots for interviewer {interviewer_id}")
        return [SlotResponse(**s.__dict__) for s in slots]

    def book_slot(self, req: SlotBookingRequest, db: Session) -> dict:
        slot = db.query(InterviewSlotModel).filter_by(slot_id=req.slot_id).first()

        if not slot:
            raise HTTPException(status_code=404, detail="Slot not found")
        if slot.booked:
            raise HTTPException(status_code=409, detail="Slot already booked")

        slot.booked = True
        slot.candidate_id = req.candidate_id
        db.commit()

        try:
            send_confirmation_email(
                to_email=req.candidate_email,
                interviewer_id=slot.interviewer_id,
                start_time=slot.start_datetime_utc.isoformat(),
                duration=slot.duration_minutes
            )
            logger.info(f"[Book] Slot {slot.slot_id} booked and email sent")
        except Exception as e:
            logger.warning(f"[Email] Failed to send confirmation: {e}")

        return {"status": "booked", "slot_id": slot.slot_id}

    def cancel_slot(self, req: SlotCancelRequest, db: Session) -> dict:
        slot = db.query(InterviewSlotModel).filter_by(slot_id=req.slot_id).first()
        if not slot:
            raise HTTPException(status_code=404, detail="Slot not found")

        try:
            if slot.calendar_event_id:
                self.calendar.delete_event(slot.calendar_event_id)
                logger.info(f"[Cancel] Calendar event {slot.calendar_event_id} deleted")
        except Exception as e:
            logger.warning(f"[Calendar] Failed to delete event: {e}")

        db.delete(slot)
        db.commit()

        logger.info(f"[Cancel] Slot cancelled: {req.slot_id}")
        return {"status": "cancelled", "slot_id": req.slot_id}

    def reschedule_slot(self, req: RescheduleRequest, db: Session) -> dict:
        slot = db.query(InterviewSlotModel).filter_by(slot_id=req.slot_id).first()
        if not slot:
            raise HTTPException(status_code=404, detail="Slot not found")

        new_dt = datetime.fromisoformat(req.new_start_datetime_utc)
        new_duration = req.new_duration_minutes or slot.duration_minutes

        try:
            if slot.calendar_event_id:
                self.calendar.update_event(slot.calendar_event_id, new_dt.isoformat(), new_duration)
                logger.info(f"[Reschedule] Updated event {slot.calendar_event_id}")
        except Exception as e:
            logger.warning(f"[Calendar] Failed to update calendar event: {e}")

        slot.start_datetime_utc = new_dt
        slot.duration_minutes = new_duration
        db.commit()

        logger.info(f"[Reschedule] Slot {slot.slot_id} moved to {new_dt}")
        return {"status": "rescheduled", "slot_id": slot.slot_id}
