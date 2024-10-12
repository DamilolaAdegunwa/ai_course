# 1. how the st lib to the project
import streamlit as st
import time

# 2. add title
st.title("hi world")

# 3. run the app
# open the local terminal
# cd into the directory where this file is located then run the following command
# streamlit run streamlit_index.py

# 4. Header
st.header('This is a header')
st.subheader('This is a sub-header')
st.text('This is a text')
st.markdown('This is a markdown')

st.button('this is a button')
st.checkbox('my checkbox')
st.radio('Radio', ['1', '2', '3'])
st.selectbox('Select', ['1', '2', '3'])
image = st.file_uploader('File Uploader', type=['png', 'jpg', 'jpeg', 'ico', 'tiff'])

st.color_picker('color picker')

st.balloons()
st.date_input("date input")
st.time_input("date input")
st.text_input("date input")
st.number_input("date input")
st.snow()
st.text_area('text area')

st.slider('Slider', min_value=0, max_value=100, value=50)


my_bar = st.progress(0)
for percentage_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percentage_complete + 1)

with st.spinner('waiting...'):
    time.sleep(5)

col1, col2 = st.columns(2)

with col1:
    st.header('Column 1')
    st.text('some text in Column 1')

with col2:
    st.header('Column 2')
    st.text('some text in Column 2')

if image:
    st.image(image, caption='Upload Image', use_column_width=True)

# 👏👏👏👏👏👏👏👏 Congratulations! 👏👏👏👏👏👏👏👏