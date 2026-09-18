import os
from tavily import TavilyClient


def tavily_search(query: str) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query, max_results=5)
    output = []
    for r in results.get("results", []):
        output.append(f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n")
    return "\n---\n".join(output) if output else "No results found."


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Search the web for current information on a given query. Use this when you need up-to-date facts, news, or information not in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]
