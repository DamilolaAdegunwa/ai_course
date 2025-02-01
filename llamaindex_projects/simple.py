import llama_index  #import DocumentIndex

class Document:
    def __init__(self, text):
        self.text = text
class DocumentIndex:
    def __init__(self, documents):
        self.documents = documents
    def query(self, query_text):
        return [doc.text for doc in self.documents if query_text in doc.text]

# Initialize the index
index = DocumentIndex()

# Add documents to the index
documents = [
    "The sky is blue.",
    "The sun is bright.",
    "The grass is green."
]
for doc in documents:
    index.add_document(doc)

# Query the index
query = "What color is the sky?"
results = index.query(query)
print("Search Results:", results)