import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger("email_utils")

# You can set these via ENV or config file in production
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "noreply@hirefy.ai"  # replace with actual
SENDER_PASSWORD = "your-password"   # use secure env storage in production


def send_confirmation_email(to_email: str, interviewer_id: str, start_time: str, duration: int) -> None:
    """
    Sends a confirmation email to the candidate.
    """
    try:
        subject = "✅ Interview Slot Confirmed"
        body = f"""
Dear Candidate,

Your interview with interviewer **{interviewer_id}** has been successfully booked.

🗓 Date & Time: {start_time} UTC  
⏱ Duration: {duration} minutes  

Best regards,  
HirelyAI Interview Bot  
        """

        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = to_email
        message["Subject"] = subject

        message.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)

        logger.info(f"[Email] Confirmation sent to {to_email}")

    except Exception as e:
        logger.error(f"[Email] Failed to send confirmation: {e}")
        raise
