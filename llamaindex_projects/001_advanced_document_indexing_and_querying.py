import os
import json
import llama_index  #import SimpleDocumentIndex, Document

class Document:
    def __init__(self, text):
        self.text = text
class SimpleDocumentIndex:
    def __init__(self, documents):
        self.documents = documents
    def query(self, query_text):
        return [doc.text for doc in self.documents if query_text in doc.text]

class AdvancedDocumentIndexer:
    def __init__(self):
        self.index = SimpleDocumentIndex()

    def load_documents(self, directory):
        """Load documents from a specified directory."""
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                with open(os.path.join(directory, filename), 'r') as file:
                    document = json.load(file)
                    # Create a Document instance
                    doc = Document(content=document['content'], metadata=document['metadata'])
                    self.index.add(doc)

    def search(self, query, filters=None, sort_by=None):
        """Search the indexed documents based on a query, optional filters, and sorting criteria."""
        results = self.index.query(query)

        if filters:
            results = self.apply_filters(results, filters)

        if sort_by:
            results = self.sort_results(results, sort_by)

        return results

    def apply_filters(self, results, filters):
        """Apply filters to the results based on metadata."""
        filtered_results = []
        for result in results:
            match = True
            for key, value in filters.items():
                if key == 'date':
                    # Assuming date is in 'YYYY-MM-DD' format
                    if not (value.split(" to ")[0] <= result.metadata['date'] <= value.split(" to ")[1]):
                        match = False
                elif key in result.metadata and result.metadata[key] != value:
                    match = False

            if match:
                filtered_results.append(result)

        return filtered_results

    def sort_results(self, results, sort_by):
        """Sort results based on a given criterion."""
        if sort_by == "relevance":
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == "date":
            return sorted(results, key=lambda x: x.metadata['date'])
        return results

if __name__ == "__main__":
    indexer = AdvancedDocumentIndexer()
    indexer.load_documents("documents")

    # Example query
    query_input = {
        "query": "climate change",
        "filters": {"date": "2022-01-01 to 2022-12-31"},
        "sort_by": "relevance"
    }

    results = indexer.search(query_input['query'], query_input['filters'], query_input['sort_by'])
    print("Search Results:", results)
