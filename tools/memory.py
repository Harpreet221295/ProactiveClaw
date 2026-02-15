"""Agent-facing long-term memory tools backed by SimpleMem."""

from memory import query_memory, retrieve_memory

SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_long_term_memory",
            "description": "Search long-term memory for past conversations and facts. Use when you need context from previous sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_long_term_memory",
            "description": "Retrieve specific memories by topic or keyword from long-term storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or keyword to retrieve",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def query_long_term_memory(query: str) -> str:
    return query_memory(query)


def retrieve_long_term_memory(query: str) -> str:
    return retrieve_memory(query)
