import json
import smtplib
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from pathlib import Path



class EmailSender():
    
    def __init__(self):
        self._init_config()
        return
    
    ### public functions
    
    def send_email_notification(self, subject:str="No Subject", body:str="lorem ipsum", attachment_paths:Path|list[Path]=None) -> None:
        message = self._create_message_container(subject)
        self._add_body_text(message, body)
        
        if attachment_paths:
            cleaned_paths = self._regularize_path(attachment_paths)
            self._process_attachments(message, cleaned_paths)
        
        self._send_email(message)
        return
    
    ### private functions
    
    def _init_config(self, config_name:str="config.json") -> None:
        self.config = self._load_config(config_name)
        self.from_email = self.config["email_user"]
        self.to_email = self.config["recipient"]
        self.email_user = self.config["email_user"]
        self.email_pass = self.config["email_pass"]
        return
    
    def _load_config(self, filename:str) -> dict:
        config_path = Path(__file__).parent / filename
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required config file missing: {config_path}") from e
        return
    
    def _create_message_container(self, subject: str) -> MIMEMultipart:
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = subject
        return message
    
    def _add_body_text(self, message: MIMEMultipart, body: str) -> None:
        message.attach(MIMEText(body, "plain"))
        return
    
    def _regularize_path(self, paths:str|Path) -> Path:
        if isinstance(paths, list):
            paths = [Path(path) for path in paths]
        else:
            paths = [Path(paths)]
        return paths
    
    def _process_attachments(self, message:MIMEMultipart, attachment_paths:list[Path]) -> MIMEMultipart:
        for path in attachment_paths:
            if path.is_file():
                self._attach_file(message, path)
            else:
                print(f"Warning: Attachment file not found at {path}")
        return
    
    def _attach_file(self, message: MIMEMultipart, attachment_path: Path) -> None:
        with open(attachment_path, "rb") as file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file.read())
            
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={attachment_path.name}"
        )
        message.attach(part)
        return
    
    def _send_email(self, message: MIMEMultipart, verbose: bool = False) -> None:
        server = None
        
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.email_user, self.email_pass)
            server.sendmail(self.from_email, self.to_email, message.as_string())
            
            if verbose:
                print("Notification sent successfully.")
                
        except Exception as e:
            print(f"Failed to send email: {e}")
            
        finally:
            if server is not None:
                server.quit()
        
        return



def send_email(subject:str="No Subject", body:str="lorem ipsum", attachment_paths:Path|list[Path]=None) -> None:
    sender = EmailSender()
    sender.send_email_notification(subject, body, attachment_paths)
    return