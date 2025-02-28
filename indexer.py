import os
import numpy as np
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class DocumentIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = None
        self.index = None
        self.embeddings = None
    
    def load_documents(self, file_path='data/processed/all_documents.csv'):
        """Load documents from processed CSV file"""
        print("Loading documents...")
        self.documents = pd.read_csv(file_path)
        print(f"Loaded {len(self.documents)} documents.")
        return self.documents
    
    def create_index(self):
        """Create FAISS index from document content"""
        if self.documents is None:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        print("Creating document embeddings...")
        # Encode documents
        texts = self.documents['content'].tolist()
        
        # Process in batches to avoid memory issues
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings)
        
        # Create FAISS index
        print("Building FAISS index...")
        vector_dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Created index with {self.index.ntotal} vectors of dimension {vector_dimension}")
        return self.index
    
    def save_index(self, index_path='data/processed/faiss_index.bin', embeddings_path='data/processed/embeddings.npy'):
        """Save the FAISS index and embeddings"""
        if self.index is None:
            raise ValueError("No index created. Call create_index() first.")
        
        # Save FAISS index
        print("Saving index...")
        faiss.write_index(self.index, index_path)
        
        # Save embeddings
        print("Saving embeddings...")
        np.save(embeddings_path, self.embeddings)
        
        # Save model info
        with open('data/processed/model_info.pkl', 'wb') as f:
            pickle.dump({
                'model_name': self.model.get_sentence_embedding_dimension()
            }, f)
        
        print("Index and embeddings saved.")
    
    def load_index(self, index_path='data/processed/faiss_index.bin', embeddings_path='data/processed/embeddings.npy'):
        """Load a saved FAISS index and embeddings"""
        print("Loading index and embeddings...")
        self.index = faiss.read_index(index_path)
        self.embeddings = np.load(embeddings_path)
        print(f"Loaded index with {self.index.ntotal} vectors.")
        return self.index

def build_index():
    """Main function to build the index"""
    # Check if data directory exists
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    
    # Check if processed data file exists
    if not os.path.exists('data/processed/all_documents.csv'):
        print("Error: Processed data file not found. Run data_ingestion.py first.")
        return False
    
    indexer = DocumentIndexer()
    indexer.load_documents()
    indexer.create_index()
    indexer.save_index()
    return True

if __name__ == "__main__":
    build_index()