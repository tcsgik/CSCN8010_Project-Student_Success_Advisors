import faiss
import pickle
import numpy as np

class FaissSearchEngine:
    def __init__(self, model, index_path='models/faiss.index', meta_path='models/texts.pkl'):
        self.model = model
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
        D, I = self.index.search(query_embedding, top_k)
        return [(self.texts[i], D[0][rank]) for rank, i in enumerate(I[0])]
