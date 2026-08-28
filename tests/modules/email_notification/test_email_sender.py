import smtplib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modules.email_notification.email_sender import (
    SMTP_HOST,
    SMTP_PORT,
    EmailSender,
)



def test_sender_loads_valid_config(tmp_path:Path) -> None:
    config_path = _write_config(
        tmp_path,
        """{
            "email_user": "sender@example.com",
            "email_pass": "password",
            "recipient": "recipient@example.com"
        }""",
    )

    sender = EmailSender(config_name=str(config_path))

    assert sender.from_email == "sender@example.com"
    assert sender.to_email == "recipient@example.com"


@pytest.mark.parametrize(
    ("config_content", "message"),
    [
        ("[]", "Email config must be an object"),
        (
            '{"email_user": "sender@example.com", "email_pass": "password"}',
            "Email config field 'recipient' must be a non-blank string",
        ),
        ("{", "Invalid JSON in email config"),
    ],
)
def test_sender_reports_invalid_config(
    tmp_path:Path,
    config_content:str,
    message:str,
) -> None:
    config_path = _write_config(tmp_path, config_content)

    with pytest.raises(ValueError, match=message):
        EmailSender(config_name=str(config_path))


def test_sender_regularizes_single_and_multiple_attachment_paths() -> None:
    assert EmailSender._regularize_paths("one.txt") == [Path("one.txt")]
    assert EmailSender._regularize_paths([
        "one.txt",
        Path("two.txt"),
    ]) == [
        Path("one.txt"),
        Path("two.txt"),
    ]


def test_sender_sends_body_and_attachment(
    tmp_path:Path,
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(
        tmp_path,
        """{
            "email_user": "sender@example.com",
            "email_pass": "password",
            "recipient": "recipient@example.com"
        }""",
    )
    attachment_path = tmp_path / "summary.txt"
    attachment_path.write_text("experiment summary", encoding="utf-8")
    smtp_constructor = MagicMock()
    smtp_server = smtp_constructor.return_value.__enter__.return_value
    monkeypatch.setattr(
        "modules.email_notification.email_sender.smtplib.SMTP",
        smtp_constructor,
    )

    EmailSender(config_name=str(config_path)).send_email_notification(
        subject="Experiments complete",
        body="All experiments succeeded.",
        attachment_paths=attachment_path,
    )

    smtp_constructor.assert_called_once_with(SMTP_HOST, SMTP_PORT)
    smtp_server.starttls.assert_called_once_with()
    smtp_server.login.assert_called_once_with(
        "sender@example.com",
        "password",
    )
    smtp_server.sendmail.assert_called_once()
    message_text = smtp_server.sendmail.call_args.args[2]
    assert "All experiments succeeded." in message_text
    assert "summary.txt" in message_text


def test_sender_reports_smtp_errors_without_raising(
    tmp_path:Path,
    monkeypatch:pytest.MonkeyPatch,
    capsys:pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(
        tmp_path,
        """{
            "email_user": "sender@example.com",
            "email_pass": "password",
            "recipient": "recipient@example.com"
        }""",
    )
    monkeypatch.setattr(
        "modules.email_notification.email_sender.smtplib.SMTP",
        MagicMock(side_effect=smtplib.SMTPConnectError(421, "Unavailable")),
    )

    EmailSender(config_name=str(config_path)).send_email_notification()

    assert "Failed to send email" in capsys.readouterr().out


def _write_config(tmp_path:Path, content:str) -> Path:
    config_path = tmp_path / "email_config.json"
    config_path.write_text(content, encoding="utf-8")
    return config_path
