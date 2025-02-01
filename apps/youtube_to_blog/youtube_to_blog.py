from youtube_transcript_api import YouTubeTranscriptApi
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from online_module import *
from apikey import apikey


def get_youtube_video_id(url):
    # Regular expressions for various YouTube URL formats
    video_id_regex_list = [
        r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]+)'
    ]
    for regex in video_id_regex_list:
        match = re.search(regex, url)
        if match:
            return match.group(1)

    return None


def get_transcript(url):
    video_id = get_youtube_video_id(url)
    img_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = ' '.join(entry['text'] for entry in transcript_list)
    return [transcript_list, transcript_text, img_url]


output_format_blog = """
                <h1> Blog Title </h1>
                <h1> Table of Contents</h1> <li> links of content </li>
                <h1> Introduction </h1><p> introduction</p>
                <h1> Heading of section </h1><p>content</p>
                .
                .
                .
                <h1> Heading of section </h1><p>content</p><h4>code if required</h4>
               <h1> Conclusion </h1><p>conclusion</p>
                """
output_format_ending = """ 
                <h1> FAQ </h1><p>question answers</p>
                 <h1> Links </h1><p>useful links</p>"""

st.title("Youtube To Blog Post")

client = setup_openai(apikey)
url = st.text_input("Enter your prompt", value="https://www.youtube.com/watch?v=cxs6iXeyfEY")

if st.button("Generate Blog"):
    with st.spinner('Getting Transcript...'):
        transcript_list, transcript_text, img_url = get_transcript(url)
        st.write("Transcript Received - Word count: " + str(len(transcript_text.split())))
        # st.write(transcript_text)
        st.image(img_url, caption="Video Thumbnail")

    with st.spinner('Generating Summary...'):
        ### Generate a Summary of the Transcript
        summary_prompt = ("You are a youtuber and you want to write a blog post about your video. "
                          "Using the below transcript of the video create a summary that will be "
                          "later used to generate the blog ") + transcript_text
        text_area_placeholder = st.empty()
        summary = generate_text_openai_streamlit(client, summary_prompt,
                                                 text_area_placeholder=text_area_placeholder)
        # model = "gpt-4-0125-preview",
        text_area_placeholder.empty()

    with st.spinner('Generating Blog...'):
        blog_prompt = (f"Create a blog post from the following summary: {summary} "
                       f"using the following format: {output_format_blog}")

        text_area_placeholder = st.markdown("", unsafe_allow_html=True)

        blog = generate_text_openai_streamlit(client, blog_prompt,
                                              text_area_placeholder=text_area_placeholder,
                                              html=True)

        ending_prompt = (f"write the ending of the blog from the following summary: {summary} "
                         f"using the following format: {output_format_ending}")
        text_area_placeholder2 = st.markdown("", unsafe_allow_html=True)
        blog_end = generate_text_openai_streamlit(client, ending_prompt,
                                                  text_area_placeholder=text_area_placeholder2,
                                                  html=True)

        st.video(url)

        # st.text("Blog Generated - Word count: " + str(len(blog.split())))