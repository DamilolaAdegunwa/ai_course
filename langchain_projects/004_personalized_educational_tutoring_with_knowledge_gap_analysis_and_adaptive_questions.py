from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool


class AdaptiveTutoringSystem:
    def __init__(self, subject="Biology"):
        self.subject = subject
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        self.memory = ConversationBufferMemory(memory_key="learning_history")
        self.llm = OpenAI(model_name="text-davinci-003")

        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Knowledge Gap Analysis",
                func=self.identify_knowledge_gap,
                description="Analyzes user responses for knowledge gaps."
            ),
            Tool(
                name="Adaptive Question Generator",
                func=self.generate_adaptive_question,
                description="Generates follow-up questions based on user knowledge level."
            ),
            Tool(
                name="Detailed Explanation Provider",
                func=self.provide_detailed_explanation,
                description="Provides detailed explanations to help the user understand concepts better."
            ),
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory
        )

    def identify_knowledge_gap(self, user_answer):
        prompt = PromptTemplate(
            input_variables=["answer"],
            template=(
                "Analyze the following student answer for knowledge gaps in {subject}: '{answer}'. "
                "Identify any missing information or areas where understanding may be incomplete."
            )
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        gap_analysis = chain.run(answer=user_answer, subject=self.subject)
        return gap_analysis

    def generate_adaptive_question(self, gap_analysis):
        prompt = PromptTemplate(
            input_variables=["gap"],
            template=(
                "Based on this knowledge gap analysis, '{gap}', generate a follow-up question to target this knowledge gap."
            )
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        adaptive_question = chain.run(gap=gap_analysis)
        return adaptive_question

    def provide_detailed_explanation(self, question):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Provide a detailed explanation for the following question: '{question}'"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        explanation = chain.run(question=question)
        return explanation

    def tutor_response(self, question, user_answer):
        # Step 1: Knowledge gap analysis
        knowledge_gap = self.identify_knowledge_gap(user_answer)

        # Step 2: Adaptive question generation based on the knowledge gap
        follow_up_question = self.generate_adaptive_question(knowledge_gap)

        # Step 3: Provide an explanation for clarification if needed
        explanation = self.provide_detailed_explanation(question)

        # Compile the response
        response = {
            "original_question": question,
            "user_answer": user_answer,
            "knowledge_gap_analysis": knowledge_gap,
            "follow_up_question": follow_up_question,
            "detailed_explanation": explanation
        }
        return response


# Example Usage
if __name__ == "__main__":
    tutor = AdaptiveTutoringSystem(subject="Biology")

    # Example 1: Knowledge Gap and Follow-up Question for "What is a mitochondrion?"
    print("Example 1 - Mitochondrion Question:")
    response = tutor.tutor_response(
        question="What is a mitochondrion?",
        user_answer="It’s the powerhouse of the cell."
    )
    print(response)

    # Example 2: Knowledge Gap and Follow-up for "What is mitosis?"
    print("\nExample 2 - Mitosis Question:")
    response = tutor.tutor_response(
        question="What is mitosis?",
        user_answer="I’m not sure."
    )
    print(response)

    # Example 3: Knowledge Gap and Follow-up for "What is DNA replication?"
    print("\nExample 3 - DNA Replication Question:")
    response = tutor.tutor_response(
        question="What is DNA replication?",
        user_answer="It's the copying of DNA."
    )
    print(response)

#https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96