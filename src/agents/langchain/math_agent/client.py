# Create server parameters for stdio connection
import os

import httpx
from langchain_core.messages import SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

load_dotenv()

EXTRA_HEADERS = {"alltrue-endpoint-identifier": "langchain-runner"}
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
BASE_URL = os.getenv("BASE_URL", None)


def get_server_params():
    if MCP_TRANSPORT == "stdio":

        return StdioServerParameters(
            command="python",
            # Make sure to update to the full absolute path to your math_server.py file
            args=["-m", "src.mcp_servers.math_stdio"],  # TODO: don't harcode
        )
    elif MCP_TRANSPORT == "streamable-http":
        return {
            "url": MCP_SERVER_URL,
            "transport": "streamable-http",
        }


def get_model():
    match MODEL_PROVIDER:
        case "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.environ["OPENAI_API_KEY"]
            return ChatOpenAI(
                # TODO: parameterize this
                model_name="gpt-4.1",
                api_key=api_key,
                base_url=BASE_URL,
                default_headers=EXTRA_HEADERS,
            )
        case "azure_openai":
            from langchain_openai import AzureChatOpenAI
            api_key = os.environ["AZURE_OPENAI_API_KEY"]
            return AzureChatOpenAI(
                deployment_name="test",
                api_key=api_key,
                azure_endpoint=BASE_URL,
                default_headers=EXTRA_HEADERS,
                api_version='2024-07-01-preview',

            )
        case "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = os.environ["ANTHROPIC_API_KEY"]
            return ChatAnthropic(
                model="claude-3-7-sonnet-20250219",
                api_key=api_key,
                base_url=BASE_URL,
                default_headers=EXTRA_HEADERS,
            )
        case "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.environ["GOOGLE_API_KEY"]
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                api_key=api_key,
                client_options=dict(api_endpoint=BASE_URL),
                additional_headers=EXTRA_HEADERS,
            )
        case _:
            raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


async def run():
    server_params = get_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model=get_model(),
                                       tools=tools,
                                       prompt=SystemMessage(content="You will be given a math problem to solve. "
                                                                    "Do not do the math yourself. "
                                                                    "Instead, you must use the tools available to you."))  # TODO: make model parameterized and add custom headers
            agent_response = await agent.ainvoke({"messages": "what's 3 + 5 + 5 + 3?"})

            print(agent_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
