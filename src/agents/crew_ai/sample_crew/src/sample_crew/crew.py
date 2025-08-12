import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# TODO: make configurable if have base url
EXTRA_HEADERS = {"alltrue-endpoint-identifier": "crew-ai-runner"}
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
BASE_URL = os.getenv("BASE_URL", None)


def get_llm():
    match MODEL_PROVIDER:
        case "openai":

            return LLM(
                model='gpt-4o',
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=BASE_URL,
                extra_headers=EXTRA_HEADERS
            )
        case "azure_openai":
            return LLM(
                model='azure/test',
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                base_url=BASE_URL,
                default_headers=EXTRA_HEADERS,
                api_version="2023-05-15"

            )
        case "anthropic":
            model = LLM(
                model='anthropic/claude-3-5-sonnet-latest',
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=BASE_URL,
                default_headers=EXTRA_HEADERS
            )
            #import pdb; pdb.set_trace()
            return model
        case "gemini":
            return LLM(
                model='gemini-1.5-flash',
                api_key=os.environ["GEMINI_API_KEY"],
                base_url=BASE_URL,
                extra_headers=EXTRA_HEADERS
            )
        case _:
            raise ValueError(
                f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}. Supported providers are: openai, azure_openai, anthropic, gemini.")

def get_server_params():
    if MCP_TRANSPORT == "http":
        return [
            # Streamable HTTP Server
            {
                "url": MCP_SERVER_URL,
                "transport": "streamable-http"
            }
        ]
    elif MCP_TRANSPORT == "stdio":
        from mcp import StdioServerParameters
        return [
            # StdIO Server
            StdioServerParameters(
                command="python3",
                args=["src/mcp_servers/math_stdio.py"],  # TODO: configure
                env={"UV_PYTHON": "3.12", **os.environ},
            )
        ]


@CrewBase
class SampleCrew():
    """SampleCrew crew"""

    # TODO: make configurable
    mcp_server_params = get_server_params()

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],  # type: ignore[index]
            verbose=True,
            llm=get_llm()
        )

    @agent
    def mathematician(self) -> Agent:
        return Agent(
            config=self.agents_config['mathematician'],  # type: ignore[index]
            verbose=True,
            llm=get_llm(),
            tools=self.get_mcp_tools()
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],  # type: ignore[index]
            verbose=True,
            llm=get_llm(),
            tools=self.get_mcp_tools()
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  # type: ignore[index]
        )

    @task
    def math_task(self) -> Task:
        return Task(
            config=self.tasks_config['math_task'],  # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],  # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SampleCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
