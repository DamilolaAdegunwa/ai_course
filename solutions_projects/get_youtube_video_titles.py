import re
from googleapiclient.discovery import build


# Initialize YouTube API
def get_youtube_service(api_key):
    return build('youtube', 'v3', developerKey=api_key)


# Fetch the channel ID based on the @handle name
def get_channel_id(service, channel_name):
    request = service.search().list(
        part="snippet",
        q=channel_name,
        type="channel",
        maxResults=1
    )
    response = request.execute()
    if response['items']:
        return response['items'][0]['snippet']['channelId']
    else:
        raise ValueError("Channel not found. Please check the channel name.")


# Get the latest videos from the channel
def get_video_titles(service, channel_id):
    video_titles = []
    request = service.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        maxResults=50
    )
    response = request.execute()
    for item in response['items']:
        if item['id']['kind'] == "youtube#video":
            video_titles.append(item['snippet']['title'])
    return video_titles


# Main function
def main():
    api_key = input("Enter your YouTube API key: ")
    channel_name = input("Enter the YouTube channel handle (e.g., @intheworldofai): ")

    # Remove "@" from the handle if included
    channel_name = re.sub(r"^@", "", channel_name)

    # Create YouTube service
    youtube_service = get_youtube_service(api_key)

    try:
        # Fetch the channel ID
        channel_id = get_channel_id(youtube_service, channel_name)

        # Get video titles
        video_titles = get_video_titles(youtube_service, channel_id)

        print("\nVideos from the latest to the oldest:\n")
        for title in video_titles:
            print(title)
    except Exception as e:
        print(f"An error occurred: {e}")


# Run the program
if __name__ == "__main__":
    main()
