#!/usr/bin/env python3

import smtplib
import subprocess
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
sender_email = "your_email@umass.edu"  # Replace with your UMass email
receiver_email = "avinashnandyala921@gmail.com"
smtp_server = "smtp.umass.edu"  # UMass SMTP server
smtp_port = 587

def get_squeue_status():
    """Get the squeue status for the current user."""
    try:
        result = subprocess.run(['squeue', '--me'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error getting squeue status: {str(e)}"

def send_email(subject, body):
    """Send email using SMTP."""
    try:
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject

        # Add body to email
        message.attach(MIMEText(body, "plain"))

        # Create SMTP session
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            # You'll need to enter your UMass email password when prompted
            server.login(sender_email, input("Enter your UMass email password: "))
            server.send_message(message)
            
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

def main():
    while True:
        # Get squeue status
        status = get_squeue_status()
        
        # Send email
        send_email("SLURM Queue Status Update", status)
        
        # Wait for 2 minutes
        time.sleep(120)

if __name__ == "__main__":
    main()