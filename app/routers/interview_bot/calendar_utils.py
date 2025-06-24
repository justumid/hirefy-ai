import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger("calendar_utils")

SCOPES = ["https://www.googleapis.com/auth/calendar"]
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "secrets/gcal_service_account.json")


class GoogleCalendarClient:
    def __init__(self, calendar_id: str, timezone: str = "Asia/Tashkent"):
        self.calendar_id = calendar_id
        self.timezone = timezone
        self.service = self._init_service()

    def _init_service(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            service = build("calendar", "v3", credentials=credentials)
            logger.info("[GoogleCalendar] Authenticated with service account.")
            return service
        except Exception as e:
            logger.exception(f"[GoogleCalendar] Failed to authenticate: {e}")
            raise

    def list_events(self, time_min: datetime, time_max: datetime) -> List[Dict]:
        try:
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=time_min.isoformat() + "Z",
                timeMax=time_max.isoformat() + "Z",
                timeZone=self.timezone,
                singleEvents=True,
                orderBy="startTime"
            ).execute()
            return events_result.get("items", [])
        except HttpError as e:
            logger.error(f"[GoogleCalendar] API Error (list_events): {e}")
            return []

    def is_slot_available(self, start: datetime, end: datetime) -> bool:
        events = self.list_events(start, end)
        if events:
            logger.info(f"[SlotCheck] Conflict: {len(events)} event(s) in time range.")
            return False
        return True

    def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: str = "",
        attendees: Optional[List[Dict[str, str]]] = None,
        location: Optional[str] = None,
        send_notifications: bool = True
    ) -> Optional[str]:
        event = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start.isoformat(),
                "timeZone": self.timezone,
            },
            "end": {
                "dateTime": end.isoformat(),
                "timeZone": self.timezone,
            },
            "attendees": attendees or [],
            "location": location,
            "reminders": {
                "useDefault": True,
            }
        }

        try:
            created_event = self.service.events().insert(
                calendarId=self.calendar_id,
                body=event,
                sendUpdates="all" if send_notifications else "none"
            ).execute()
            logger.info(f"[EventCreate] Created: {created_event.get('htmlLink')}")
            return created_event.get("id")
        except HttpError as e:
            logger.error(f"[EventCreate] Failed: {e}")
            return None

    def delete_event(self, event_id: str) -> bool:
        try:
            self.service.events().delete(calendarId=self.calendar_id, eventId=event_id).execute()
            logger.info(f"[EventDelete] Deleted event ID: {event_id}")
            return True
        except HttpError as e:
            logger.warning(f"[EventDelete] Failed: {e}")
            return False

    def generate_available_slots(
        self,
        start_day: datetime,
        slot_duration_minutes: int = 30,
        working_hours: Tuple[int, int] = (9, 18)
    ) -> List[Dict[str, str]]:
        """
        Generates available time slots within working hours, avoiding existing events.

        Returns:
            List of dicts with 'start' and 'end' ISO 8601 timestamps.
        """
        slots = []
        now = datetime.now()
        date_cursor = start_day.replace(hour=working_hours[0], minute=0, second=0, microsecond=0)

        while date_cursor.hour < working_hours[1]:
            slot_end = date_cursor + timedelta(minutes=slot_duration_minutes)
            if slot_end > now and self.is_slot_available(date_cursor, slot_end):
                slots.append({
                    "start": date_cursor.isoformat(),
                    "end": slot_end.isoformat()
                })
            date_cursor += timedelta(minutes=slot_duration_minutes)

        logger.info(f"[SlotGen] {len(slots)} slots found for {start_day.date()}")
        return slots
