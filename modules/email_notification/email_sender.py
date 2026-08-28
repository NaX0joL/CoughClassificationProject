import json
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import TypeAlias, TypedDict



SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
CONFIG_FIELDS = ("email_user", "email_pass", "recipient")

AttachmentPath:TypeAlias=str|Path
AttachmentPaths:TypeAlias=AttachmentPath|list[AttachmentPath]|None



class EmailConfig(TypedDict):
    email_user:str
    email_pass:str
    recipient:str



class EmailSender:

    def __init__(self, config_name:str="config.json") -> None:
        self._initialize_config(config_name)
        return

    def send_email_notification(
        self,
        subject:str="No Subject",
        body:str="lorem ipsum",
        attachment_paths:AttachmentPaths=None,
    ) -> None:
        message = self._create_message_container(subject)
        self._add_body_text(message, body)

        if attachment_paths:
            regularized_paths = self._regularize_paths(attachment_paths)
            self._process_attachments(message, regularized_paths)

        self._send_email(message)
        return

    def _initialize_config(self, config_name:str) -> None:
        self.config = self._load_config(config_name)
        self.from_email = self.config["email_user"]
        self.to_email = self.config["recipient"]
        self.email_user = self.config["email_user"]
        self.email_pass = self.config["email_pass"]
        return

    def _load_config(self, filename:str) -> EmailConfig:
        config_path = Path(__file__).parent / filename

        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                raw_config = json.load(config_file)
                
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Required config file missing: {config_path}"
            ) from error
            
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in email config: {config_path}"
            ) from error
            
        except OSError as error:
            raise OSError(
                f"Could not read email config: {config_path}"
            ) from error

        config = self._validate_config(raw_config, config_path)
        return config

    @staticmethod
    def _validate_config(raw_config:object, config_path:Path) -> EmailConfig:
        if not isinstance(raw_config, dict):
            raise ValueError(f"Email config must be an object: {config_path}")

        for field in CONFIG_FIELDS:
            value = raw_config.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Email config field {field!r} must be a non-blank string: "
                    f"{config_path}"
                )

        config = EmailConfig(
            email_user=raw_config["email_user"],
            email_pass=raw_config["email_pass"],
            recipient=raw_config["recipient"],
        )
        return config

    def _create_message_container(self, subject:str) -> MIMEMultipart:
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = subject
        return message

    def _add_body_text(self, message:MIMEMultipart, body:str) -> None:
        message.attach(MIMEText(body, "plain"))
        return

    @staticmethod
    def _regularize_paths(
        paths:AttachmentPath|list[AttachmentPath],
    ) -> list[Path]:
        if isinstance(paths, list):
            regularized_paths = [Path(path) for path in paths]
            return regularized_paths

        return [Path(paths)]

    def _process_attachments(
        self,
        message:MIMEMultipart,
        attachment_paths:list[Path],
    ) -> None:
        for path in attachment_paths:
            if path.is_file():
                self._attach_file(message, path)
            else:
                print(f"Warning: Attachment file not found at {path}")
        return

    @staticmethod
    def _attach_file(
        message:MIMEMultipart,
        attachment_path:Path,
    ) -> None:
        with attachment_path.open("rb") as attachment_file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_file.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={attachment_path.name}",
        )
        message.attach(part)
        return

    def _send_email(
        self,
        message:MIMEMultipart,
        verbose:bool=False,
    ) -> None:
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)
                server.sendmail(
                    self.from_email,
                    self.to_email,
                    message.as_string(),
                )
        except (OSError, smtplib.SMTPException) as error:
            print(f"Failed to send email: {error}")
            return

        if verbose:
            print("Notification sent successfully.")
        return



def send_email(
    subject:str="No Subject",
    body:str="lorem ipsum",
    attachment_paths:AttachmentPaths=None,
) -> None:
    sender = EmailSender()
    sender.send_email_notification(subject, body, attachment_paths)
    return
