import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import re

class DocumentRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load the model
        self.model = SentenceTransformer(model_name)
        
        # Load index and documents
        self.index = None
        self.documents = None
        self.load_resources()
    
    def load_resources(self):
        """Load FAISS index and document data"""
        # Check if index exists
        if not os.path.exists('data/processed/faiss_index.bin') or not os.path.exists('data/processed/all_documents.csv'):
            raise FileNotFoundError("Index or document files not found. Run indexer.py first.")
        
        # Load the index
        self.index = faiss.read_index('data/processed/faiss_index.bin')
        
        # Load the documents
        self.documents = pd.read_csv('data/processed/all_documents.csv')
        
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.documents)} documents.")
    
    def retrieve(self, query, top_k=5, cdp_filter=None):
        """Retrieve the most relevant documents for a query"""
        # Encode the query
        query_vector = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k * 3)  # Get more results for filtering
        
        # Get the documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents.iloc[idx].to_dict()
                doc['score'] = 1.0 / (1.0 + distances[0][i])  # Convert distance to a similarity score
                results.append(doc)
        
        # Filter by CDP if specified
        if cdp_filter:
            results = [doc for doc in results if doc['cdp'] == cdp_filter.lower()]
        
        # Return top_k results
        return results[:top_k]
    
    def answer_question(self, query):
        """Answer a CDP-related question"""
        # Determine if the question is about a specific CDP
        cdp_filter = self.detect_cdp(query)
        
        # Check if this is a comparison question
        cdps_to_compare = self.detect_comparison(query)
        
        if cdps_to_compare:
            return self.handle_comparison(query, cdps_to_compare)
        
        # Retrieve relevant documents
        docs = self.retrieve(query, top_k=5, cdp_filter=cdp_filter)
        
        if not docs:
            return "I'm sorry, I couldn't find information to answer your question. Could you rephrase or ask about another topic?"
        
        # Build answer from retrieved documents
        answer = self.format_answer(query, docs, cdp_filter)
        return answer
    
    def detect_cdp(self, query):
        """Detect which CDP the query is referring to"""
        query_lower = query.lower()
        
        cdps = {
            "segment": ["segment"],
            "mparticle": ["mparticle", "m particle", "m-particle"],
            "lytics": ["lytics"],
            "zeotap": ["zeotap"]
        }
        
        for cdp, keywords in cdps.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return cdp
        
        return None
    
    def detect_comparison(self, query):
        """Detect if the query is asking for a comparison between CDPs"""
        query_lower = query.lower()
        
        # Check for comparison keywords
        comparison_keywords = ["compare", "comparison", "difference", "differences", "vs", "versus", "or"]
        
        has_comparison = any(keyword in query_lower for keyword in comparison_keywords)
        
        if not has_comparison:
            return None
        
        # Detect which CDPs are being compared
        cdps = []
        for cdp in ["segment", "mparticle", "lytics", "zeotap"]:
            if cdp in query_lower or self.get_cdp_variations(cdp)[0] in query_lower:
                cdps.append(cdp)
        
        return cdps if len(cdps) >= 2 else None
    
    def get_cdp_variations(self, cdp):
        """Get variations of CDP names for flexible matching"""
        variations = {
            "segment": ["segment"],
            "mparticle": ["mparticle", "m particle", "m-particle"],
            "lytics": ["lytics"],
            "zeotap": ["zeotap"]
        }
        
        return variations.get(cdp, [cdp])
    
    def handle_comparison(self, query, cdps):
        """Handle comparison questions between multiple CDPs"""
        # Get information about each CDP for the specific task
        responses = {}
        
        # Extract the task from the query
        task_patterns = [
            r"how (?:do|to|can) \w+ (.*?) in",
            r"how (?:do|to|can) \w+ (.*?) with",
            r"how (?:do|to|can) \w+ (.*?) using",
            r"how (?:do|to|can) \w+ (.*?) between",
            r"how does (.*?) (?:work|compare)"
        ]
        
        task = ""
        for pattern in task_patterns:
            match = re.search(pattern, query.lower())
            if match:
                task = match.group(1)
                break
        
        if not task:
            task = "handle this functionality"
        
        # Get information for each CDP
        for cdp in cdps:
            modified_query = f"How to {task} in {cdp}"
            docs = self.retrieve(modified_query, top_k=3, cdp_filter=cdp)
            
            if docs:
                # Extract the most relevant content
                content = "\n".join([doc['content'] for doc in docs])
                responses[cdp] = content
        
        # Build comparison answer
        answer = f"## Comparison: {task.capitalize()} in {', '.join(cdps[:-1])} and {cdps[-1]}\n\n"
        
        for cdp in cdps:
            if cdp in responses:
                answer += f"### {cdp.capitalize()}\n"
                # Extract a summary (first 500 characters)
                summary = responses[cdp][:500] + "..."
                answer += summary + "\n\n"
                
                # Add a source link if available
                if 'url' in docs[0] and docs[0]['url']:
                    answer += f"[View {cdp.capitalize()} documentation]({docs[0]['url']})\n\n"
            else:
                answer += f"### {cdp.capitalize()}\n"
                answer += "I couldn't find specific information about this task in this CDP.\n\n"
        
        answer += "\n**Note:** For complete step-by-step instructions, please refer to the official documentation for each platform."
        
        return answer
    
    def format_answer(self, query, docs, cdp_filter):
        """Format the answer from retrieved documents"""
        # Extract CDP name for the header
        cdp_name = cdp_filter.capitalize() if cdp_filter else "CDP"
        
        # Check if query is a "how-to" question
        is_how_to = re.search(r"how (?:do|to|can) (?:i|you|we)", query.lower()) is not None
        
        # Start building the answer
        if is_how_to:
            answer = f"## How to {self.extract_task(query)} in {cdp_name}\n\n"
        else:
            answer = f"## {cdp_name} Information\n\n"
        
        # Combine content from top results
        main_content = "\n".join([doc['content'] for doc in docs[:2]])
        
        # Clean up the content (remove duplicate paragraphs, etc.)
        main_content = self.clean_content(main_content)
        
        # Add the main content
        answer += main_content + "\n\n"
        
        # Add sources
        answer += "### Sources\n"
        urls_added = set()
        for doc in docs:
            if 'url' in doc and doc['url'] and doc['url'] not in urls_added:
                answer += f"- [{doc['title']}]({doc['url']})\n"
                urls_added.add(doc['url'])
        
        return answer
    
    def extract_task(self, query):
        """Extract the task from a how-to question"""
        patterns = [
            r"how (?:do|to|can) (?:i|you|we) (.*?)(?:\?|$|\s+in|\s+with|\s+using)",
            r"how (?:do|to|can) (?:i|you|we) (.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip()
        
        return "perform this task"
    
    def clean_content(self, content):
        """Clean up content for better readability"""
        # Remove duplicate blank lines
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Limit to ~1000 characters for readability
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        return content
    
    def check_irrelevant_question(self, query):
        """Check if a question is irrelevant to CDPs"""
        query_lower = query.lower()
        
        # List of CDP-related terms
        cdp_terms = [
            "segment", "mparticle", "lytics", "zeotap", "cdp", "customer data", 
            "platform", "integration", "source", "destination", "audience", 
            "track", "identify", "event", "user", "profile", "data"
        ]
        
        # Check if query contains any CDP-related terms
        has_cdp_term = any(term in query_lower for term in cdp_terms)
        
        # Check for common irrelevant topics
        irrelevant_topics = [
            "movie", "film", "show", "tv", "weather", "sport", "game", 
            "restaurant", "food", "recipe", "book", "music"
        ]
        
        has_irrelevant_topic = any(topic in query_lower for topic in irrelevant_topics)
        
        # If query has irrelevant topics but no CDP terms, it's likely irrelevant
        if has_irrelevant_topic and not has_cdp_term:
            return True
        
        return False

if __name__ == "__main__":
    # Test the retriever
    retriever = DocumentRetriever()
    
    test_queries = [
        "How do I set up a new source in Segment?",
        "How can I create a user profile in mParticle?",
        "How does Segment's audience creation process compare to Lytics'?",
        "What movie is releasing this weekend?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        if retriever.check_irrelevant_question(query):
            print("This question appears to be unrelated to CDPs.")
        else:
            result = retriever.answer_question(query)
            print(result[:300] + "..." if len(result) > 300 else result)