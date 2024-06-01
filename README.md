# LLM Sales Agent

## Overview

The LLM Sales Agent is a voice-interactive agent designed to assist with sales and customer support inquiries. It uses the Deepgram API for speech-to-text and text-to-speech functionalities, allowing it to communicate via voice. The agent can route queries to either a sales or a customer support module and retrieve relevant product information.

## Features

- **Voice Interaction**: Uses Deepgram API for converting speech to text and text to speech.
- **Sales and Support Handling**: Routes queries to appropriate modules (sales or customer support).
- **Product Information Retrieval**: Retrieves and provides information on products.

## Repository Structure

```
sales_agent/
│
├── app.py                              # Main application entry point
├── sales_agent.py                      # Sales agent module
├── sample_product_catalog.txt          # Sample product catalog
├── example_product_price_id_mapping.json # Example product to price ID mapping
├── sales_agent_test.ipynb              # Jupyter notebook for testing functions in the sales agent module
├── static/                             # Static files (HTML file for the web application for chat)
└── __pycache__/                        # Compiled Python files
```

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AmzArch/LLM-Sales-Agent.git
    cd LLM-Sales-Agent/sales_agent
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Set up environment variables**:
    - Create a `.env` file in the root directory and add your Deepgram API key:
      ```bash
      DEEPGRAM_API_KEY=your_deepgram_api_key
      OPENAI_API_KEY=your_openai_api_key
      ```

## License

This project is licensed under the MIT License.
