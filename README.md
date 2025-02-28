# CDP Support Agent

A chatbot designed to answer "how-to" questions related to Customer Data Platforms (CDPs): Segment, mParticle, Lytics, and Zeotap. The chatbot extracts relevant information from official documentation to guide users on specific tasks.

## Features

- Answers "how-to" questions for Segment, mParticle, Lytics, and Zeotap
- Extracts information directly from official documentation
- Handles variations in question phrasing
- Provides cross-CDP comparisons
- Handles advanced configuration and integration questions

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/cdp-support-agent.git
cd cdp-support-agent
```

2. Create a virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
   ```
   venv\Scripts\activate
   ```
   - On macOS/Linux:
   ```
   source venv/bin/activate
   ```

4. Install dependencies:
```
pip install -r requirements.txt
```

5. Run the data ingestion script to download and process documentation:
```
python data_ingestion.py
```

6. Start the Flask application:
```
python app.py
```

7. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Type your question in the input field.
2. The chatbot will retrieve relevant information from the CDP documentation.
3. For cross-CDP comparisons, specifically mention both CDPs in your question.

## Project Structure

- `app.py`: Main Flask application
- `data_ingestion.py`: Scripts to download and process CDP documentation
- `indexer.py`: Creates searchable indices from processed documentation
- `retriever.py`: Retrieves relevant context for user questions
- `utils.py`: Utility functions used across the project
- `templates/`: HTML templates for the web interface
- `data/`: Storage for raw and processed documentation
