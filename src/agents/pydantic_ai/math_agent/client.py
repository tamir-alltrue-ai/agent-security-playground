import os
from datetime import datetime

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import random

from pydantic_ai.settings import ModelSettings

from src.config import ROOT_DIR
from pydantic_ai.mcp import MCPServerStreamableHTTP, MCPServerStdio

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
BASE_URL = os.getenv("BASE_URL", None)
TRACING_API = os.getenv("TRACING_API", "http://localhost:7000")
if BASE_URL:
    os.environ[
        'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'] = f"{TRACING_API}/otel/traces/pydantic-ai-runner-{datetime.now().isoformat()}"

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


def get_model(provider: str, base_url: str | None = None):
    if provider == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        return OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=api_key, base_url=base_url),
                           settings=ModelSettings(
                               extra_headers={"alltrue-endpoint-identifier": "pydantic-ai-runner"} if base_url else {}
                           )
                           )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_server(server_type: str):
    if server_type == "http":
        return MCPServerStreamableHTTP(MCP_SERVER_URL)
    elif server_type == "stdio":
        return MCPServerStdio(
            command='python', args=['src/agents/pydantic_ai/math_agent/stdio_server.py'],
            cwd=ROOT_DIR)
    else:
        raise ValueError(f"Unsupported server type: {server_type}")


agent = Agent(toolsets=[get_server(server_type=MCP_TRANSPORT)], instrument=True,
              system_prompt="""
You are an agent that is designed to demonstrate simple statistical properties of random variables.
You will be given access to two tools: 
1. random_number(start: int, end: int) -> int: this can help you generate random numbers from a uniform distribution between 'start' and 'end'.
2. add(a: int, b: int) -> int: this can help you add two integers together.

Your task is to demonstrate that the sum of two uniform random variables has an expected value that is equal to the sum of their expected values.
You should do this by repeatedly generating random numbers from two uniform distributions, summing them, and then calculating the average of these sums over many trials.

You may not add numbers yourself as you're bad at math. Instead, use the 'add' tool available to you.
""",
              model=get_model(
                  provider=MODEL_PROVIDER, base_url=BASE_URL
              )
              )


@agent.tool_plain()
def random_number(start: int, end: int) -> int:
    """
    Returns a random number between 'start' and 'end'.
    """
    return random.randint(start, end)


async def main():
    async with agent:
        result = await agent.run('Demonstrate the expected value of the sum of two uniform random variables. '
                                 'Do so by drawing no more than 20 pairs of random numbers from two uniform distributions.')
    print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
