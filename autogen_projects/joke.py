"""
import os
from autogen import AssistantAgent, UserProxyAgent
print("***")
print(os.environ.get("OPENAI_API_KEY"))
print(os.environ["OPENAI_API_KEY"])
print("***")
"""

import os
from autogen import ConversableAgent

agent = ConversableAgent(
    "chatbot",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]},
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)
print(agent.name)
print(agent.chat_messages)
print(agent.client)
print(agent.description)

reply = agent.generate_reply(messages=[{"content": "Tell me a joke.", "role": "user"}])
print(reply)