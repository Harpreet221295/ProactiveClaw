import base64
from email.mime.text import MIMEText

from googleapiclient.discovery import build

from ._google_auth import _get_google_creds


def _get_gmail_service():
    return build("gmail", "v1", credentials=_get_google_creds())


def _parse_email_headers(headers: list[dict], *names: str) -> dict[str, str]:
    result = {}
    lower_names = {n.lower(): n for n in names}
    for h in headers:
        key = h["name"].lower()
        if key in lower_names:
            result[lower_names[key]] = h["value"]
    return result


def list_emails(query: str = "", max_results: int = 10) -> str:
    service = _get_gmail_service()
    resp = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    messages = resp.get("messages", [])
    if not messages:
        return "No emails found."
    output = []
    for msg_stub in messages:
        msg = service.users().messages().get(
            userId="me", id=msg_stub["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        hdrs = _parse_email_headers(msg.get("payload", {}).get("headers", []),
                                     "From", "Subject", "Date")
        output.append(
            f"From: {hdrs.get('From', '?')}\n"
            f"Subject: {hdrs.get('Subject', '(no subject)')}\n"
            f"Date: {hdrs.get('Date', '?')}\n"
            f"Snippet: {msg.get('snippet', '')}\n"
            f"ID: {msg['id']}"
        )
    return "\n---\n".join(output)


def read_email(message_id: str) -> str:
    service = _get_gmail_service()
    msg = service.users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()
    hdrs = _parse_email_headers(msg.get("payload", {}).get("headers", []),
                                 "From", "To", "Subject", "Date")
    # Extract plain-text body
    body = ""
    payload = msg.get("payload", {})
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    else:
        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                break
    if not body:
        body = msg.get("snippet", "(could not extract body)")
    return (
        f"From: {hdrs.get('From', '?')}\n"
        f"To: {hdrs.get('To', '?')}\n"
        f"Subject: {hdrs.get('Subject', '(no subject)')}\n"
        f"Date: {hdrs.get('Date', '?')}\n\n"
        f"{body}"
    )


def send_email(to: str, subject: str, body: str) -> str:
    service = _get_gmail_service()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    return f"Email sent successfully (ID: {sent['id']})"


def reply_to_email(message_id: str, body: str) -> str:
    service = _get_gmail_service()
    original = service.users().messages().get(
        userId="me", id=message_id, format="metadata",
        metadataHeaders=["From", "Subject", "Message-ID"],
    ).execute()
    hdrs = _parse_email_headers(original.get("payload", {}).get("headers", []),
                                 "From", "Subject", "Message-ID")
    subject = hdrs.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    message = MIMEText(body)
    message["to"] = hdrs.get("From", "")
    message["subject"] = subject
    message["In-Reply-To"] = hdrs.get("Message-ID", "")
    message["References"] = hdrs.get("Message-ID", "")
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = service.users().messages().send(
        userId="me",
        body={"raw": raw, "threadId": original.get("threadId")},
    ).execute()
    return f"Reply sent successfully (ID: {sent['id']})"


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_emails",
            "description": "List emails from the user's Gmail inbox. Supports Gmail search queries (e.g. 'is:unread', 'from:someone@example.com', 'subject:meeting').",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gmail search query to filter emails (e.g. 'is:unread', 'from:boss@company.com'). Empty string returns recent emails.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to return (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_email",
            "description": "Read the full content of a specific email by its message ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "The ID of the email to read.",
                    },
                },
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send a new email from the user's Gmail account. Only use this when the user explicitly asks to send an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reply_to_email",
            "description": "Reply to an existing email thread. Only use this when the user explicitly asks to reply to an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "The ID of the email to reply to.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The reply body text.",
                    },
                },
                "required": ["message_id", "body"],
            },
        },
    },
]
