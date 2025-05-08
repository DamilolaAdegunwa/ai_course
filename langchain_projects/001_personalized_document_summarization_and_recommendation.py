from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool


class DocumentSummarizationSystem:
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
                name="Recommend Articles",
                func=self.recommend_articles,
                description="Provides article recommendations based on user interests.",
            ),
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

    def recommend_articles(self, user_interest):
        # Placeholder logic for recommendation
        recommendations = {
            "sustainable energy": [
                "Innovations in Solar Technology",
                "Wind Energy: The Future",
                "Recycling for Sustainability"
            ],
            "AI": [
                "AI in Healthcare",
                "The Future of AI",
                "Ethics in AI"
            ]
        }
        return recommendations.get(user_interest, ["No recommendations found."])

    def user_query(self, query):
        response = self.agent.run(query)
        return response


# Example Usage
if __name__ == "__main__":
    system = DocumentSummarizationSystem()
    system.load_documents(["research_paper.txt", "news_article.txt", "course_material.txt"])

    print(system.user_query("Summarize the document about AI advancements in healthcare."))
    print(system.user_query("Recommend articles related to sustainable energy."))
    print(system.user_query("Provide a summary for the section on Machine Learning Algorithms."))


# https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96
