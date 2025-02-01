import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parents[2]))
print(str(Path(__file__).resolve().parents[0]))

import openmacro
import openmacro.llm
import openmacro.core
import openmacro.memory
import openmacro.profile
import openmacro.computer
import openmacro.cli
import openmacro.extensions
import openmacro.omi
import openmacro.profiles
import openmacro.speech
import openmacro.tests
import openmacro.utils

from openmacro.profile import Profile

# Create a new profile
Profile(
    user={"name": "moyo", "version": "2.0"},
    assistant={"name": "SmartBot", "personality": "Professional"}
).set_default()

# Set the new profile as the default
# new_profile.set_default()
#print(f"New default profile is now: {new_profile.user['name']} with assistant {new_profile.assistant['name']}")

"""
# Create a profile with user and assistant information
my_profile = openmacro.Profile(
    user={"name": "adegunwa", "version": "1.0.1"},
    assistant={"name": "Marco", "personality": "Friendly"}
)

print(f"Profile created for user: {my_profile.user['name']}")
print(f"Assistant name: {my_profile.assistant['name']} with personality: {my_profile.assistant['personality']}")

# Now use the profile in a chat interaction
response = openmacro.llm.chat("Tell me a fun fact!", profile=my_profile)
print(f"Response from {my_profile.assistant['name']}: {response}")
"""



# openmacro.llm.models
# openmacro.core.Computer.
# from openmacro import Session

# Create a new session with the 'damilola' profile
# session = Session(profile='damilola')

# Send a chat message
# response = session.chat("what are the colors of the rainbow")

# Print the response
# print(response)

#rough page 1
"""
# imports
from openmacro.profile import Profile
from openmacro.extensions import BrowserKwargs, EmailKwargs
import os
os.environ['API_KEY'] = "0993ad9a-ccb5-4c58-bf05-4da4debab385"
# profile setup
profile: Profile = Profile(
env={"path": r"C:\Users\damil\.cache\openmacro"},  # Add this line
    user = {
        "name": "a2",
        "version": "1.0.0"
    },
    assistant = {
        "name": "Macro",
        "personality": "You respond in a professional attitude and respond in a formal, yet casual manner.",
        "messages": [],
        "breakers": ["the task is done.",
                     "the conversation is done."]
    },
    safeguards = {
        "timeout": 180,
        "auto_run": True,
        "auto_install": True
    },
    extensions = {
    # type safe kwargs
        "Browser": BrowserKwargs(headless=False, engine="google"),
        "Email": EmailKwargs(email="damee1993@gmail.com", password="Bett..."),
        "LLM": {"api_key":"0993ad9a-ccb5-4c58-bf05-4da4debab385"},
        "api_key":"0993ad9a-ccb5-4c58-bf05-4da4debab385"
    },
    config = {
        "verbose": True,
        "conversational": True,
        "dev": False,
        "LLM": {"api_key": "0993ad9a-ccb5-4c58-bf05-4da4debab385"},
        "api_key": "0993ad9a-ccb5-4c58-bf05-4da4debab385"
    },
    languages = {
    # specify custom paths to languages or add custom languages for openmacro
      "python": [r"C:\Windows\py.EXE", "-c"],
      "rust": ["cargo", "script", "-e"]  # not supported by default, but can be added!
    },
    tts = {
    # powered by KoljaB/RealtimeSTT
    # options ["SystemEngine", "GTTSEngine", "OpenAIEngine"]
      "enabled": True,
      "engine": "OpenAIEngine"
    },
    paths = {
        "logs": r"C:\Users\damil\.cache\openmacro\logs",
        "cache": r"C:\Users\damil\.cache\openmacro\cache"
    }
)
print("created the profile!!")
async def main():
    from openmacro.core import Openmacro

    # Initialize Openmacro with the new profile
    macro = Openmacro(profile)
    macro.llm.messages = []

    # Perform chat and stream the output
    async for chunk in macro.chat("Plot an exponential graph for me!", stream=True):
        print(chunk, end="")

# Run the async main function
import asyncio
asyncio.run(main())

"""