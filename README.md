# Scheme Research Tool

A Streamlit-based web application for analyzing government schemes and policies using natural language processing.

## Features

- URL content extraction and analysis
- Intelligent question-answering system
- Structured scheme summaries
- Data storage
- User-friendly interface

## Installation

1. Clone the repository:

git clone https://github.com/oneshikaa/scheme-research-tool.git
cd scheme-research-tool


2. Create and activate virtual environment:

python -m venv venv
venv\Scripts\activate  # On Windows


3. Install dependencies:

pip install -r requirements.txt


4. Configure API keys:
- Create `.streamlit/secrets.toml` or
- Set environment variables or
- Use the UI input field

## Usage

1. Start the application:

streamlit run main.py


2. Access the web interface at http://localhost:8501

3. Add scheme URLs:
   - Enter URLs directly
   - Upload a text file containing URLs
   - Click "Process URLs"

4. Interact with the system:
   - Ask questions about schemes
   - Generate structured summaries
   - View source attribution

## Project Structure


scheme_research_tool/
├── .streamlit/                # Streamlit configuration
├── data/                     # Data storage
├── main.py                   # Main application
├── requirements.txt          # Dependencies
└── README.md                # Documentation


## Configuration

### API Keys
Set your OpenAI API key using one of these methods:

1. Streamlit secrets:

# .streamlit/secrets.toml
OPENAI_API_KEY = "your-key-here"


2. Environment variable:

export OPENAI_API_KEY="your-key-here"


3. UI input field in the application

### Environment Variables
Customize application behavior in `.env`:

DEBUG=False
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=200


## License

This project is licensed under the MIT License - see the LICENSE file for details.