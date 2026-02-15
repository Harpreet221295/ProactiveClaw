import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

_TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "token.json")
_CREDENTIALS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials.json")
_GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]


def _get_google_creds():
    creds = None
    if os.path.exists(_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(_TOKEN_FILE, _GOOGLE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(_CREDENTIALS_FILE, _GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(_TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return creds
