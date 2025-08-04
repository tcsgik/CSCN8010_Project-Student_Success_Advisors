import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissSearchEngine:
    def __init__(self, index_path='models/faiss.index', meta_path='models/texts.pkl'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = None
        self._load_index(index_path, meta_path)

    def _load_index(self, index_path, meta_path):
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.texts = pickle.load(f)

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        D, I = self.index.search(query_embedding, top_k * 2)
        seen = set()
        results = []
        for rank, i in enumerate(I[0]):
            item = self.texts[i]
            content = item['content']
            if content not in seen:
                seen.add(content)
                results.append((item, D[0][rank]))
            if len(results) == top_k:
                break
        return results
    
# vector_search = FaissSearchEngine()
# results = vector_search.search("How do I make a payment")
# for question in results:
#     print(question)
# print("###################################################################")
# results = vector_search.search("Can I have an extension/ instalment plan of my payment due")
# for question in results:
#     print(question)
# print("###################################################################")
# results = vector_search.search("When is the last day to drop a course without penalty")
# for question in results:
#     print(question)
    
