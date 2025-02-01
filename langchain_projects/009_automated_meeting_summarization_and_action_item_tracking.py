import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.callbacks import LoggingCallbackHandler
from google.cloud import speech
import wave


class MeetingSummarizationSystem:
    def __init__(self, gcp_credentials_json):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json
        self.memory = ConversationBufferMemory(memory_key="meeting_context")
        self.llm = OpenAI(model_name="gpt-4")
        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Audio Transcription",
                func=self.transcribe_audio,
                description="Transcribes meeting audio into text."
            ),
            Tool(
                name="Summarization",
                func=self.summarize_meeting,
                description="Summarizes the meeting discussion."
            ),
            Tool(
                name="Action Item Extraction",
                func=self.extract_action_items,
                description="Extracts actionable items from the meeting summary."
            ),
            Tool(
                name="Action Tracker",
                func=self.track_actions,
                description="Tracks action items and their assignees."
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            callback_manager=LoggingCallbackHandler()
        )

    def transcribe_audio(self, audio_file_path):
        client = speech.SpeechClient()
        with wave.open(audio_file_path, "rb") as audio_file:
            content = audio_file.readframes(audio_file.getnframes())

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        response = client.recognize(config=config, audio=audio)
        transcription = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcription

    def summarize_meeting(self, meeting_transcript):
        prompt = PromptTemplate(
            input_variables=["transcript"],
            template=("Summarize the following meeting discussion: {transcript}")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        summary = chain.run(transcript=meeting_transcript)
        return summary

    def extract_action_items(self, summary):
        prompt = PromptTemplate(
            input_variables=["summary"],
            template=(
                "From the following summary, extract actionable items and their responsible team members: {summary}")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        action_items = chain.run(summary=summary)
        return action_items

    def track_actions(self, action_items):
        # In a real implementation, this could update a database or a task management system.
        return f"Action items tracked: {action_items}"

    def process_meeting(self, audio_file_path):
        transcript = self.transcribe_audio(audio_file_path)
        summary = self.summarize_meeting(transcript)
        action_items = self.extract_action_items(summary)
        action_tracking = self.track_actions(action_items)

        return {
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "action_tracking": action_tracking
        }


# Example Usage
if __name__ == "__main__":
    # Initialize the system with Google Cloud credentials
    meeting_system = MeetingSummarizationSystem(gcp_credentials_json='path/to/your/gcp_credentials.json')

    # Example 1: Process a meeting audio file
    print("Example 1 - Meeting Audio Processing:")
    results = meeting_system.process_meeting("meeting_audio.wav")
    print("Transcript:", results['transcript'])
    print("Summary:", results['summary'])
    print("Action Items:", results['action_items'])
    print("Action Tracking:", results['action_tracking'], "\n")

    # Example 2: Another meeting audio file
    print("Example 2 - Another Meeting Audio Processing:")
    results = meeting_system.process_meeting("another_meeting_audio.wav")
    print("Transcript:", results['transcript'])
    print("Summary:", results['summary'])
    print("Action Items:", results['action_items'])
    print("Action Tracking:", results['action_tracking'], "\n")

    # Example 3: Yet another meeting audio file
    print("Example 3 - Yet Another Meeting Audio Processing:")
    results = meeting_system.process_meeting("yet_another_meeting_audio.wav")
    print("Transcript:", results['transcript'])
    print("Summary:", results['summary'])
    print("Action Items:", results['action_items'])
    print("Action Tracking:", results['action_tracking'])

#https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96