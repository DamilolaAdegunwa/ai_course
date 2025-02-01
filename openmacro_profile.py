# imports
from openmacro.profile import Profile
from openmacro.extensions import BrowserKwargs, EmailKwargs
import os
from apikey import apikey
os.environ['OPENAI_API_KEY'] = apikey
# Set the API key in environment variable (you might also use this later)
os.environ['API_KEY'] = "0993ad9a-ccb5-4c58-bf05-4da4debab385"
openai_apikey = apikey
# profile setup
profile: Profile = Profile(
    env={
        "path": r"C:\Users\damil\.cache\openmacro"
    },
    user={
        "name": "josh_openmacro",
        "version": "1.0.0"
    },
    assistant={
        "name": "Macro",
        "personality": "You respond in a professional attitude and respond in a formal, yet casual manner.",
        "messages": [],
        "breakers": ["the task is done.",
                     "the conversation is done."]
    },
    safeguards={
        "timeout": 180,
        "auto_run": True,
        "auto_install": True
    },
    extensions={
        # Valid extensions
        "Browser": BrowserKwargs(headless=False, engine="google"),
        "Email": EmailKwargs(email="damee1993@gmail.com", password="Bett...")
    },
    config={
        "verbose": True,
        "conversational": True,
        "dev": False
    },
    languages={
        # specify custom paths to languages or add custom languages for openmacro
        "python": [r"C:\Windows\py.EXE", "-c"],
        "rust": ["cargo", "script", "-e"]  # not supported by default, but can be added!
    },
    tts={
        # powered by KoljaB/RealtimeSTT
        # options ["SystemEngine", "GTTSEngine", "OpenAIEngine"]
        "enabled": True,
        "engine": "OpenAIEngine",
        "api_key": openai_apikey
    },
    paths={
        "logs": r"C:\Users\damil\.cache\openmacro\logs",
        "cache": r"C:\Users\damil\.cache\openmacro\cache"
    }
)

"""
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

