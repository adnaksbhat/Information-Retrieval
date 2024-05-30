Enter a word. Based on provided list of documents, it returns which document it is present.
Search mechanism.

Code in python:
===============

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class InformationRetrievalSystem:
    def __init__(self, documents_dir):
        self.documents_dir = documents_dir
        self.document_paths = [os.path.join(documents_dir, filename) for filename in os.listdir(documents_dir)]
        self.document_texts = self.load_documents()

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.document_texts)

    def load_documents(self):
        document_texts = []
        for document_path in self.document_paths:
            with open(document_path, 'r', encoding='utf-8') as file:
                document_texts.append(file.read())
        return document_texts

    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]  # Get indices of top k documents
        top_documents = [(self.document_paths[idx], cosine_similarities[idx]) for idx in top_indices]
        return top_documents

if __name__ == "__main__":
    documents_dir = '/content/drive/MyDrive/Colab Notebooks/IRS'
    ir_system = InformationRetrievalSystem(documents_dir)

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        top_documents = ir_system.search(query)
        print("Top 5 documents related to the query:")
        for doc_path, similarity in top_documents:
            print(f"{doc_path} (Similarity: {similarity:.2f})")
