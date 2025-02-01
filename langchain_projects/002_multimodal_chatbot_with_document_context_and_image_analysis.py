from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from PIL import Image
import pytesseract


class MultimodalChatbot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm = OpenAI(model_name="text-davinci-003")

        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Summarize Document",
                func=self.summarize_document,
                description="Summarizes the provided document.",
            ),
            Tool(
                name="Answer Question Based on Document",
                func=self.answer_document_question,
                description="Answers questions based on the uploaded documents."
            ),
            Tool(
                name="Analyze Image",
                func=self.analyze_image,
                description="Performs basic image analysis to identify objects and text."
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory
        )

    def load_documents(self, file_paths):
        for file_path in file_paths:
            loader = TextLoader(file_path)
            documents = loader.load()
            self.vectorstore.add_documents(documents)

    def summarize_document(self, document_text):
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Please summarize the following document: {text}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        summary = chain.run(text=document_text)
        return summary

    def answer_document_question(self, question_text):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer the following question based on the uploaded documents: {question}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(question=question_text)
        return answer

    def analyze_image(self, image_path):
        image = Image.open(image_path)
        # Using OCR for text extraction
        text = pytesseract.image_to_string(image)

        # Placeholder for object detection (simulated as an example)
        # Assuming object recognition model is available, but here we'll simulate a simple output
        detected_objects = "Heart Diagram, Labels for Aorta and Ventricles"

        # Combining OCR and object detection results
        analysis = f"Detected text: {text}\nDetected objects: {detected_objects}"
        return analysis

    def user_query(self, query, image_path=None):
        if image_path:
            response = self.analyze_image(image_path)
        else:
            response = self.agent.run(query)
        return response


# Example Usage
if __name__ == "__main__":
    system = MultimodalChatbot()
    system.load_documents(["smartphone_manual.txt", "medical_textbook.txt", "medication_info.txt"])

    # Text query example
    print(system.user_query("What are the main features of the smartphone model in this manual?"))

    # Image query example (assuming an image of a heart diagram is available)
    print(system.user_query("Please explain this heart diagram.", image_path="heart_diagram.png"))

    # Text query example for medical document
    print(system.user_query("List the side effects mentioned for the medication in the uploaded document."))

#https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96