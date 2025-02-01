import asyncio
import logging
import os

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import CodeExecutorAgent, CodingAssistantAgent
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.teams import RoundRobinGroupChat, StopMessageTermination
# from autogen_ext.code_executor.docker_executor import DockerCommandLineCodeExecutor
from autogen_ext.code_executors import DockerCommandLineCodeExecutor
from autogen_ext.models import OpenAIChatCompletionClient

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.INFO)

async def main() -> None:
    async with DockerCommandLineCodeExecutor(work_dir="coding") as code_executor:
        code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)
        coding_assistant_agent = CodingAssistantAgent(
            "coding_assistant", model_client=OpenAIChatCompletionClient(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
        )
        group_chat = RoundRobinGroupChat([coding_assistant_agent, code_executor_agent])
        result = await group_chat.run(
            task=r"Create a plot of NVDIA and TSLA stock returns YTD from 2024-01-01 and save it to 'C:\Users\damil\PycharmProjects\ai_course\autogen_projects\nvidia_tesla_2024_ytd.png'.",
            termination_condition=StopMessageTermination(),
        )

asyncio.run(main())