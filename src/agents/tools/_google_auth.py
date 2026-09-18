"""Google OAuth (shared by the Gmail and Calendar tools).

Files live in the project root by default:
  credentials.json — OAuth client downloaded from Google Cloud Console
  token.json       — created after the first sign-in, refreshed automatically
Override with GOOGLE_CREDENTIALS_FILE / GOOGLE_TOKEN_FILE in .env.
"""
from __future__ import annotations

import os

from core.paths import GOOGLE_CREDENTIALS_FILE, GOOGLE_TOKEN_FILE

_TOKEN_FILE = str(GOOGLE_TOKEN_FILE)
_CREDENTIALS_FILE = str(GOOGLE_CREDENTIALS_FILE)
_GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]


class GoogleNotConfigured(RuntimeError):
    pass


def google_is_configured() -> bool:
    return os.path.exists(_TOKEN_FILE) or os.path.exists(_CREDENTIALS_FILE)


def _get_google_creds(interactive: bool = True):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = None
    if os.path.exists(_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(_TOKEN_FILE, _GOOGLE_SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not os.path.exists(_CREDENTIALS_FILE):
            raise GoogleNotConfigured(
                f"Google is not set up: {_CREDENTIALS_FILE} not found. "
                "Download an OAuth client (Desktop app) from Google Cloud Console and save it there, "
                "then connect Google from the Settings panel or run: python src/setup_wizard.py --google"
            )
        if not interactive:
            raise GoogleNotConfigured("Google sign-in required — connect Google from the Settings panel.")
        from google_auth_oauthlib.flow import InstalledAppFlow
        flow = InstalledAppFlow.from_client_secrets_file(_CREDENTIALS_FILE, _GOOGLE_SCOPES)
        creds = flow.run_local_server(port=0)
    with open(_TOKEN_FILE, "w") as token:
        token.write(creds.to_json())
    return creds


def run_oauth_flow() -> str:
    """Force a fresh interactive sign-in. Returns the token path."""
    if os.path.exists(_TOKEN_FILE):
        os.remove(_TOKEN_FILE)
    _get_google_creds(interactive=True)
    return _TOKEN_FILE


if __name__ == "__main__":
    print(f"Signing in to Google… token will be saved to {run_oauth_flow()}")
