import autogen
import os
config_list = [
    {
        # 'model': 'gpt-3.5-turbo-16k',
        'model': 'gpt-4',
        'api_key': os.environ.get("OPENAI_API_KEY")
    }
]

llm_config = {
    # "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.AssistantAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web"},
    llm_config=llm_config,
    system_message="""
    Reply TERMINATE if the task has been solved at full satisfaction Otherwise, reply CONTINUE, or the reason why the task is not solved yet.
    """
)

# task2 = "Give me a summary of this article: https://microsoft.github.io/autogen/0.2/docs/Getting-Started/"

task = r"""
write a python code to output numbers 1 to 100, and then store the code in a file named 'print_numbers.py'
"""

user_proxy.initiate_chat(
    assistant,
    message=task
)

task2 = """
Change the code in the file you just created above to instead output numbers 1 to 200. update/adjust the file with the necessary change to the python code
"""