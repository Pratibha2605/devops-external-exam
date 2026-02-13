import smtplib
import os
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

def send_alert_email(message: str):
    sender_email = os.getenv("SMTP_USER")
    sender_pass = os.getenv("SMTP_PASS")
    receiver_email = os.getenv("ALERT_EMAIL")

    msg = MIMEText(message)
    msg["Subject"] = "KT Exam Alert"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_pass)
        server.send_message(msg)
        server.quit()
        print("Alert Email Sent Successfully!")
    except Exception as e:
        print("Email sending failed:", e)
