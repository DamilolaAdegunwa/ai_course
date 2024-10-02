"""
Project Title: Real-Time Multimodal Audio and Text Synchronization for Video Narratives
File Name: real_time_multimodal_audio_text_synchronization.py

Project Description:
In this project, we are building a real-time system that synchronizes audio with generated text in video narratives. The system will:

Transcribe real-time audio using OpenAI's Whisper API.
Generate corresponding captions with time stamps.
Synthesize contextual text-based narratives that provide additional details based on the transcribed speech, making it suitable for video content where commentary or captions adapt to the conversation.
Sync both the transcribed text and generated narrative with the original audio to produce a multimodal output for use in video editing software.
This project can be used in interactive videos, educational content creation, or real-time video commentary generation systems.

Python Code:
"""
import os
import librosa
import numpy as np
from openai import OpenAI
from apikey import apikey
from datetime import datetime
import time
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Step 1: Load audio and video files
def load_audio_video(file_path):
    print(f"Loading audio and video from: {file_path}")
    video_clip = VideoFileClip(file_path)
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate, video_clip


# Step 2: Transcribe the audio in real-time using Whisper
def real_time_transcribe_audio(audio_data, sample_rate):
    print("Transcribing audio in real-time...")

    start_time = time.time()
    audio_stream = BytesIO(audio_data.tobytes())
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="text"
    )

    transcribed_text = response['text']
    time_taken = time.time() - start_time
    print(f"Transcription completed in {time_taken:.2f} seconds.")

    return transcribed_text


# Step 3: Generate a contextual narrative for captions (commentary)
def generate_text_narrative(transcribed_text):
    print("Generating text narrative for additional commentary...")

    # Generate extra context or insights based on the transcribed audio using GPT
    narrative_prompt = f"Generate a detailed narrative based on this transcription:\n{transcribed_text}"

    response = client.completions.create(
        model="text-davinci-003",
        prompt=narrative_prompt,
        max_tokens=150
    )

    narrative_text = response['choices'][0]['text'].strip()
    print(f"Narrative generated: {narrative_text}")

    return narrative_text


# Step 4: Synchronize the transcription and narrative with the video
def sync_text_with_video(video_clip, transcribed_text, narrative_text, output_file):
    print("Synchronizing transcribed text and narrative with video...")

    # Create text captions and position them at the bottom of the video
    transcribed_caption = TextClip(transcribed_text, fontsize=24, color='white', size=video_clip.size)
    narrative_caption = TextClip(narrative_text, fontsize=18, color='yellow', size=video_clip.size)

    # Set the timing for captions (this is a basic sync; you can improve timing with exact timestamps)
    transcribed_caption = transcribed_caption.set_position('bottom').set_duration(video_clip.duration)
    narrative_caption = narrative_caption.set_position(('center', 'bottom')).set_duration(video_clip.duration)

    # Combine video and text captions into a composite video
    final_video = CompositeVideoClip([video_clip, transcribed_caption, narrative_caption])

    # Write the final video with synchronized captions
    final_video.write_videofile(output_file, codec='libx264')

    print(f"Final synchronized video saved as {output_file}")


# Main function to perform real-time multimodal audio-text synchronization
def multimodal_audio_text_sync(file_path, output_file):
    # Step 1: Load audio and video
    audio_data, sample_rate, video_clip = load_audio_video(file_path)

    # Step 2: Real-time transcription of audio
    transcribed_text = real_time_transcribe_audio(audio_data, sample_rate)

    # Step 3: Generate narrative for the video based on the transcription
    narrative_text = generate_text_narrative(transcribed_text)

    # Step 4: Synchronize transcribed text and narrative with the video
    sync_text_with_video(video_clip, transcribed_text, narrative_text, output_file)


# Run the project
if __name__ == "__main__":
    video_file_path = "path_to_video_file.mp4"  # Provide the path to the video file
    output_video_file = "synchronized_video_with_audio_and_text.mp4"
    multimodal_audio_text_sync(video_file_path, output_video_file)
"""
Example Inputs and Expected Outputs:
Example 1:

Input Video: A 15-second clip of a nature documentary where the narrator talks about wildlife.

Expected Output:

Transcribed Text: Captions of the narrator's speech, placed at the bottom of the video.
Narrative Text: Additional commentary generated by GPT, offering deeper insights into wildlife behavior and conservation efforts.
Final Output: A synchronized video where the transcribed captions appear at the bottom, and additional narrative commentary appears above them.
Example 2:

Input Video: A 10-second clip from a news broadcast where the reporter discusses the latest financial trends.

Expected Output:

Transcribed Text: Captions of the reporter’s words with precise timestamps.
Narrative Text: A GPT-generated commentary providing further explanation of the financial trends.
Final Output: A synchronized video with both the reporter's transcription and generated financial analysis commentary displayed.
Key Features:
Real-Time Transcription: This system captures and transcribes spoken words from video in real-time, allowing for fast integration in live broadcasts or content creation pipelines.
Contextual Narrative Generation: Beyond mere transcription, the system uses GPT to generate a richer narrative based on the spoken content, which is ideal for creating video annotations or educational supplements.
Audio-Text Synchronization: The text (both transcriptions and GPT-generated narratives) is synchronized with the video, offering seamless integration for video editing and content generation.
Multimodal Output: The system outputs a video file with synchronized text and audio, which can be readily used in video editing tools or published directly.
Video Commentary Enhancement: By generating extra commentary, this system enriches the video content, making it more informative and interactive.
Use Cases:
Content Creation: Generate real-time captions and additional insights for video creators, particularly in educational, documentary, or instructional video content.
Live Broadcasting: Implement this system for real-time captions and commentary in news broadcasting, live streams, or events.
Video Editing for Accessibility: Automatically create synchronized captions and supplementary narrative text for videos to make them more accessible to a wider audience.
Interactive Learning Tools: This system can be used in e-learning platforms to provide live explanations and annotations for educational videos.
Media Production: Use this tool in post-production workflows to speed up the process of adding subtitles and annotations to media projects.
This project introduces a more complex challenge by incorporating multimodal synchronization of both transcribed and contextual text with video, allowing for real-time caption generation and narrative enhancement. The key advancement is the ability to generate contextual commentary in addition to basic transcription, offering deeper insights or additional information based on the audio content.
"""