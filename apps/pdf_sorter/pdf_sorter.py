import sys
from pathlib import Path

import streamlit
from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parents[2]))

from online_module import *
from apikey import apikey

st.title("PDF Sorter")

client = setup_openai(apikey)
output_folder = "Organized"
## Load the Files ##
files = st.file_uploader("Choose a file", type="pdf", accept_multiple_files=True)

if st.button("Organize PDFs"):
    with st.spinner("Working on PDFs"):
        for i, file in enumerate(files):
            ## Reading the first page of the document ##
            reader = PdfReader(file)
            number_of_pages = len(reader.pages)
            page = reader.pages[0]
            raw_text = page.extract_text()

            # st.write(raw_text)
            ## Get the title and keywords ##
            output_format = 'title - keyword - keyword-....'
            # User interface
            prompt = ("Below is the text of a research paper. I want you to generate a name for the papers that has "
                      "the full name of the paper as well as 3 keywords that will allow me to find it later."
                      "If there are any special characters in the text like : / \ or other, remove them from title "
                      "Give raw text as the output") + \
                     "use the following format: " + output_format + \
                     f'"""{raw_text}"""'

            generated_text = generate_text_openai_streamlit(client, prompt, max_tokens=40)
            # st.write(generated_text)
            cleaned_text = ''.join(c for c in generated_text if c.isalnum() or c in [' ', '-', '_'])

            st.subheader(f"PDF {i + 1}")
            st.write("Title:" + cleaned_text)

            ## Save the Files ##
            os.makedirs(output_folder, exist_ok=True)  # This creates the directory if it does not exist
            new_file_path = f"{output_folder}/{cleaned_text}.pdf"

            # Write the uploaded file to the new location
            with open(new_file_path, "wb") as f:
                # If the file object is from Streamlit, you should have access to getbuffer()
                f.write(file.getbuffer())