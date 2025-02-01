import os
from transformers import pipeline, logging
import certifi

# Set logging level
logging.set_verbosity(logging.WARNING)

# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['ENV'] = 'dev'
os.environ['ENVIRONMENT'] = 'dev'

# Define pipelines for summarization and question-answering
summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn", device=0)
qa_pipeline = pipeline(task="question-answering", model="distilbert-base-cased-distilled-squad", device=0)

# Example long text for summarization
long_text = """
The Amazon rainforest, alternatively, the Amazon jungle or Amazonia, is a moist broadleaf tropical rainforest in 
the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 
(2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes 
territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, 
followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname, 
and French Guiana. States or departments in four nations contain “Amazonas” in their names. The Amazon represents over 
half of the planet's remaining rainforests and comprises the largest and most biodiverse tract of tropical rainforest 
in the world, with an estimated 390 billion individual trees divided into 16,000 species.
"""

# Example context and questions for question answering
context = """
Hugging Face is a technology company that develops tools for building applications using machine learning. 
They are known for creating the Transformers library, which has become the go-to library for Natural Language 
Processing (NLP) tasks. Hugging Face was founded in 2016 and is headquartered in New York City. Their open-source 
library is widely used for tasks such as text classification, question answering, text generation, and more.
"""
questions = [
    "What is Hugging Face known for?",
    "When was Hugging Face founded?",
    "Where is Hugging Face headquartered?"
]

# Perform Summarization
print("Summarization Results:")
summary_result = summarizer(long_text, max_length=60, min_length=30, do_sample=False)
print(f"Original Text Length: {len(long_text)}")
print(f"Summary: {summary_result[0]['summary_text']}\n")

# Perform Question Answering
print("Question Answering Results:")
for question in questions:
    answer = qa_pipeline(question=question, context=context)
    print(f"Question: {question}\nAnswer: {answer['answer']}, Confidence: {answer['score']:.2f}\n")
