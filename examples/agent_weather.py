import asyncio
import json
import os
import sys

from owly_ai import LLM
from owly_ai.agent import Agent
from owly_ai.tools import Tool


def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    weather_data = {
        "london": "Raining, 12°C",
        "tokyo": "Clear, 22°C",
        "new york": "Cloudy, 16°C",
    }
    return weather_data.get(location.lower(), "Sunny, 20°C")

async def main() -> None:
    provider = os.environ.get("PROVIDER", "openai").lower()
    
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            print("Please set OPENAI_API_KEY.")
            return
        model = "gpt-4o-mini"
    elif provider == "gemini":
        if "GEMINI_API_KEY" not in os.environ:
            print("Please set GEMINI_API_KEY.")
            return
        model = "gemini-2.5-flash-lite"
    elif provider == "vertex":
        if "GOOGLE_CLOUD_PROJECT" not in os.environ:
            print("Please set GOOGLE_CLOUD_PROJECT.")
            return
        model = os.environ.get("MODEL", "gemini-2.5-flash")
    else:
        print(f"Unknown provider: {provider}")
        return

    # Optional: Load credentials from a JSON file path if provided
    credentials = None
    creds_path = os.environ.get("CREDENTIALS_PATH")
    if creds_path and os.path.exists(creds_path):
        with open(creds_path) as f:
            credentials = json.load(f)

    llm = LLM(
        provider=provider,
        model=model,
        project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        region=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        credentials=credentials,
    )
    weather_tool = Tool.from_function(get_weather)
    
    agent = Agent(
        llm=llm,
        tools=[weather_tool],
        system_prompt="You are a helpful weather assistant. Use the get_weather tool to look up requests.",
    )

    print(f"--- Weather Agent ({provider}) initialized ---\\n")
    
    prompt1 = "What's the weather like in Tokyo right now?"
    print(f"User: {prompt1}")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.stream(prompt1):
        if hasattr(chunk, "text") and chunk.text:
            print(chunk.text, end="", flush=True)
        elif hasattr(chunk, "arguments") and chunk.arguments:
            pass
    print("\n")
    
    prompt2 = "And how about London?"
    print(f"User: {prompt2}")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.stream(prompt2):
        if hasattr(chunk, "text") and chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
