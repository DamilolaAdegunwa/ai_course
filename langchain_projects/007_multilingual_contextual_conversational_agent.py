from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import GoogleTranslatorAPI  # Assume an API for language detection and translation
from langchain.callbacks import LoggingCallbackHandler


class MultilingualConversationalAgent:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="conversation_history")
        self.llm = OpenAI(model_name="gpt-4")
        self.translator = GoogleTranslatorAPI()  # API for language detection and translation
        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Language Detection",
                func=self.detect_language,
                description="Detects the language of the user's message and sets the response language accordingly."
            ),
            Tool(
                name="Domain-Specific Knowledge Retrieval",
                func=self.retrieve_knowledge,
                description="Retrieves relevant information from domain-specific knowledge based on user query."
            ),
            Tool(
                name="Contextual Shifting",
                func=self.contextual_shift,
                description="Shifts the conversational context when the user changes topics."
            ),
            Tool(
                name="Escalation Suggestion",
                func=self.suggest_escalation,
                description="Suggests escalation to a human agent if unable to resolve the query."
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            callback_manager=LoggingCallbackHandler()
        )

    def detect_language(self, message):
        # Detects language and translates if necessary
        language = self.translator.detect_language(message)
        if language != "en":
            translated_message = self.translator.translate(message, target_language="en")
            return translated_message, language
        return message, "en"

    def retrieve_knowledge(self, message, context):
        prompt = PromptTemplate(
            input_variables=["message", "context"],
            template=("Using the context '{context}', retrieve information to help answer: '{message}'."
                      " Use specialized domain knowledge if applicable.")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        knowledge = chain.run(message=message, context=context)
        return knowledge

    def contextual_shift(self, message, current_context):
        prompt = PromptTemplate(
            input_variables=["message", "current_context"],
            template=("Detect any contextual shifts in the conversation based on the message: '{message}' "
                      "and update from the current context: '{current_context}' to the appropriate new one.")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        new_context = chain.run(message=message, current_context=current_context)
        return new_context

    def suggest_escalation(self, message):
        prompt = PromptTemplate(
            input_variables=["message"],
            template=("For the message: '{message}', if unable to provide a confident response, suggest "
                      "escalation to a human agent with a polite message.")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        escalation_message = chain.run(message=message)
        return escalation_message

    def respond(self, message, context):
        # Step 1: Detect Language
        message, language = self.detect_language(message)

        # Step 2: Contextual Shift Detection
        new_context = self.contextual_shift(message, context)

        # Step 3: Retrieve Knowledge Based on Context
        response = self.retrieve_knowledge(message, new_context)

        # Step 4: Suggest Escalation if Needed
        if "escalate" in response.lower():
            escalation_suggestion = self.suggest_escalation(message)
            return escalation_suggestion, new_context

        # Step 5: Translate Response if Necessary
        if language != "en":
            response = self.translator.translate(response, target_language=language)

        return response, new_context


# Example Usage
if __name__ == "__main__":
    agent = MultilingualConversationalAgent()
    current_context = "Account Inquiry"

    # Example 1: English Message about Account
    print("Example 1 - English Message about Account:")
    response, new_context = agent.respond("Hello, I need help with my account", current_context)
    print(f"Response: {response}\nNew Context: {new_context}\n")

    # Example 2: Spanish Message about Application Issue
    print("Example 2 - Spanish Message about Application Issue:")
    response, new_context = agent.respond("Mi aplicación no funciona", "Technical Support")
    print(f"Response: {response}\nNew Context: {new_context}\n")

    # Example 3: Context Shift to Product Inquiry
    print("Example 3 - Context Shift to Product Inquiry:")
    response, new_context = agent.respond("Actually, I want to know about the product specifications.",
                                          "Technical Support")
    print(f"Response: {response}\nNew Context: {new_context}\n")

#https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96