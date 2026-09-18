from googleapiclient.discovery import build

from ._google_auth import _get_google_creds


def _get_calendar_service():
    return build("calendar", "v3", credentials=_get_google_creds())


def list_calendar_events(time_min: str, time_max: str, max_results: int = 10) -> str:
    service = _get_calendar_service()
    events_result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    events = events_result.get("items", [])
    if not events:
        return "No events found in the given time range."
    output = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        end = event["end"].get("dateTime", event["end"].get("date"))
        desc = event.get("description", "")
        output.append(
            f"Title: {event.get('summary', '(No title)')}\n"
            f"Start: {start}\nEnd: {end}\n"
            f"Description: {desc}\nID: {event['id']}"
        )
    return "\n---\n".join(output)


def create_calendar_event(title: str, start_time: str, end_time: str, description: str = "", location: str = "") -> str:
    service = _get_calendar_service()
    body = {
        "summary": title,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    event = service.events().insert(calendarId="primary", body=body).execute()
    return f"Event created: {event.get('summary')} (ID: {event['id']})\nLink: {event.get('htmlLink')}"


def update_calendar_event(event_id: str, title: str = None, start_time: str = None, end_time: str = None, description: str = None, location: str = None) -> str:
    service = _get_calendar_service()
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    if title is not None:
        event["summary"] = title
    if start_time is not None:
        event["start"] = {"dateTime": start_time}
    if end_time is not None:
        event["end"] = {"dateTime": end_time}
    if description is not None:
        event["description"] = description
    if location is not None:
        event["location"] = location
    updated = service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    start = updated["start"].get("dateTime", updated["start"].get("date"))
    return f"Updated: {updated.get('summary')} | Start: {start} | ID: {updated['id']}"


def delete_calendar_event(event_id: str) -> str:
    service = _get_calendar_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return f"Event {event_id} deleted successfully."


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": "List events from the user's Google Calendar within a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {
                        "type": "string",
                        "description": "Start of time range in ISO 8601 format (e.g. '2025-06-15T00:00:00-08:00').",
                    },
                    "time_max": {
                        "type": "string",
                        "description": "End of time range in ISO 8601 format.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default 10).",
                    },
                },
                "required": ["time_min", "time_max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new event on the user's Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title/summary of the event.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Event start time in ISO 8601 format.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Event end time in ISO 8601 format.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional event description.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional event location.",
                    },
                },
                "required": ["title", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": "Update an existing Google Calendar event. Only provided fields will be changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The ID of the event to update.",
                    },
                    "title": {
                        "type": "string",
                        "description": "New title for the event.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "New start time in ISO 8601 format.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "New end time in ISO 8601 format.",
                    },
                    "description": {
                        "type": "string",
                        "description": "New description for the event.",
                    },
                    "location": {
                        "type": "string",
                        "description": "New location for the event.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete an event from the user's Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The ID of the event to delete.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
]
