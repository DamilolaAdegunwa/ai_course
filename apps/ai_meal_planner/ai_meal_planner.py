import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from online_module import *
from apikey import apikey
import json

st.title("AI Meal Planner")

client = setup_openai(apikey)

# Create columns for inputs
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
    weight = st.number_input('Weight (kg):', min_value=30, value=80)
with col2:
    age = st.number_input('Age', min_value=18, max_value=120, step=1, value=30)
    height = st.number_input('Height (cm)', min_value=1, max_value=250, step=1, value=170)

aim = st.selectbox('Aim', ('Lose', 'Gain', 'Maintain'))

user_data = f""" - I am a {gender}"
                - My weight is {weight} kg"
                - I am {age} years old"
                - My height is {height} cm"
                - My aim is to {aim} weight
             """
output_format = """ "range":"Range of ideal weight",
                    "target":"Target weight",
                    "difference":"Weight i need to loose or gain",
                    "bmi":"my BMI",
                    "meal_plan":"Meal plan for 7 days",
                    "total_days":"Total days to reach target weight",
                    "weight_per_week":"Weight to loose or gain per week",
                                    """

prompt = user_data + (" given the information ,follow the output format as follows."
                      " Give only json format nothing else ") + output_format

if st.button("Generate Meal Plan"):
    with st.spinner('Creating Meal plan'):
        text_area_placeholder = st.empty()
        meal_plan = generate_text_openai_streamlit(client, prompt, model="gpt-4-0125-preview",
                                                   text_area_placeholder=text_area_placeholder)
        # Check if the string starts with ```json and remove it
        if meal_plan.startswith("```json"):
            meal_plan = meal_plan.replace("```json\n", "", 1)  # Remove the first occurrence
        if meal_plan.endswith("```"):
            meal_plan = meal_plan.rsplit("```", 1)[0]  # Remove the trailing part

        meal_plan_json = json.loads(meal_plan)

        st.title("Meal Plan")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Range")
            st.write(meal_plan_json["range"])
            st.subheader("Target")
            st.write(meal_plan_json["target"])
        with col2:
            st.subheader("BMI")
            st.write(meal_plan_json["bmi"])
            st.subheader("Days")
            st.write(meal_plan_json["total_days"])

        with col3:
            st.subheader(f"{aim}")
            st.write(meal_plan_json["difference"])
            st.subheader("Per week")
            st.write(meal_plan_json["weight_per_week"])

        st.subheader("Meal plan for 7 days")
        st.write(meal_plan_json["meal_plan"])

        # my_7day_meal_plan= meal_plan_json["meal_plan"]
        # my_7day_meal_plan["Day 1"]["Breakfast"]