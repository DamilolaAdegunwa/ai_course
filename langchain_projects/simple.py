from langchain import Chain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Initialize the language model
llm = OpenAI(api_key='your-api-key')

# Create a prompt template
prompt_template = PromptTemplate(template="You are a helpful assistant. Answer the following question: {question}")

# Build a chain
chain = Chain(llm=llm, prompt=prompt_template)

# User query
user_question = "What is the capital of France?"

# Get response from the chain
response = chain.run(question=user_question)
print("Chatbot Response:", response)