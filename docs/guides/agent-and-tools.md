# Agent and Tools

`Agent` wraps `LLM` with memory + tool loop orchestration.

## Define a tool

```python
from owly_ai import Tool


def get_weather(location: str) -> str:
    return f"Weather for {location}: Sunny"

weather_tool = Tool.from_function(get_weather)
```

## Run an agent

```python
from owly_ai import Agent, LLM

llm = LLM(provider="openai", model="gpt-4o-mini")
agent = Agent(llm=llm, tools=[weather_tool])
```

`agent.stream()` yields incremental responses and tool call chunks.
