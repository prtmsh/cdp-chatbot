from flask import Flask, render_template, request, jsonify
import os
import threading
import time
from indexer import build_index
from retriever import DocumentRetriever
from utils import is_how_to_question, extract_cdp_names, check_cdp_relevance

app = Flask(__name__)

# Global variables
retriever = None
index_built = False

# Initialize the retriever
def initialize_retriever():
    global retriever, index_built
    
    # Check if index exists
    if os.path.exists('data/processed/faiss_index.bin') and os.path.exists('data/processed/all_documents.csv'):
        try:
            print("Loading existing retriever...")
            retriever = DocumentRetriever()
            index_built = True
            print("Retriever loaded successfully.")
        except Exception as e:
            print(f"Error loading retriever: {str(e)}")
            index_built = False
    else:
        print("No index found. Will build index on first request.")
        index_built = False

# Start initialization in a separate thread
init_thread = threading.Thread(target=initialize_retriever)
init_thread.daemon = True
init_thread.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    global retriever, index_built
    
    # Get the question from the request
    data = request.get_json()
    question = data.get('question', '')
    
    # Check if question is empty
    if not question.strip():
        return jsonify({
            'answer': 'Please ask a question about one of the supported CDPs (Segment, mParticle, Lytics, or Zeotap).'
        })
    
    # Check if the index is built
    if not index_built:
        # Try to build the index if needed
        try:
            print("Building index...")
            index_built = build_index()
            if index_built:
                retriever = DocumentRetriever()
                print("Index built and retriever loaded.")
            else:
                return jsonify({
                    'answer': 'The system is still initializing. Please try again in a moment.'
                })
        except Exception as e:
            print(f"Error building index: {str(e)}")
            return jsonify({
                'answer': 'An error occurred while preparing the system. Please try again later.'
            })
    
    # Check if the question is relevant to CDPs
    if not check_cdp_relevance(question) and not is_how_to_question(question):
        return jsonify({
            'answer': "I'm a CDP support agent focused on Segment, mParticle, Lytics, and Zeotap. Please ask a question related to these platforms."
        })
    
    try:
        # Get the answer
        answer = retriever.answer_question(question)
        
        return jsonify({
            'answer': answer
        })
    except Exception as e:
        print(f"Error retrieving answer: {str(e)}")
        return jsonify({
            'answer': 'Sorry, an error occurred while processing your question. Please try again.'
        })

@app.route('/api/status', methods=['GET'])
def status():
    global index_built
    return jsonify({
        'status': 'ready' if index_built else 'initializing',
        'message': 'System is ready to answer questions.' if index_built else 'System is still initializing. Please wait.'
    })

if __name__ == '__main__':
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)