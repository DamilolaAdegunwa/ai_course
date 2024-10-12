import certifi
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
# import json

# Base URLs for microservices
ITINERARY_SERVICE_URL = "http://127.0.0.1:8000"
IMAGE_SERVICE_URL = "http://127.0.0.1:8001"

# Streamlit UI setup
st.title("AI Travel Itinerary Visualizer & Planner")
st.write("Enter travel details, create an itinerary, and visualize the locations.")

# Step 1: Create an itinerary
st.header("Create Travel Itinerary")
user_id = st.text_input("Enter your user ID:")
destination_count = st.number_input("How many destinations do you want?", min_value=1, max_value=10)

destinations = []
for i in range(destination_count):
    location = st.text_input(f"Enter destination {i + 1} (e.g., Paris, France):")
    date = st.date_input(f"Travel date for {location}:")
    time_of_day = st.selectbox(f"Time of day for {location}:", ["Morning", "Afternoon", "Evening", "Night"])
    destinations.append({"location": location, "date": str(date), "time_of_day": time_of_day})

if st.button("Create Itinerary"):
    payload = {
        "user_id": user_id,
        "destinations": destinations
    }
    response = requests.post(f"{ITINERARY_SERVICE_URL}/create_itinerary/", json=payload, verify=certifi.where())
    st.write(response.json())

# Step 2: Visualize destination images
st.header("Visualize Your Itinerary")
selected_destination = st.text_input("Enter a destination from your itinerary:")
if st.button("Generate Image for Destination"):
    prompt = f"Scenic view of {selected_destination}"
    image_response = requests.post(f"{IMAGE_SERVICE_URL}/generate_image/", json={"prompt": prompt},
                                   verify=certifi.where())
    print("image_response: ")
    print(image_response)
    if image_response.status_code == 200:
        image = Image.open(BytesIO(image_response.content))
        st.image(image, caption=f"Generated Image for {selected_destination}", use_column_width=True)
    else:
        st.write("Error generating image")
