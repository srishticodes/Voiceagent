# Walmart Voice Assistant

A clean, AI-powered shopping assistant that delivers a hyper-personalized, interactive experience to Walmart customers. Features voice and text input with intelligent product search and shopping cart management.

## Features

- **Voice & Text Input**: Dual input modes for maximum accessibility
- **AI-Powered Product Search**: Semantic search through Walmart's product database
- **Delivery Status Tracking**: Real-time delivery status for all orders
- **Shopping Cart Management**: Add, view, and manage items in your cart
- **Product Recommendations**: Intelligent suggestions based on user queries
- **Modern Web Interface**: Beautiful, responsive UI with real-time updates
- **Voice Output**: Text-to-speech responses for hands-free interaction

## Project Structure

```
Voice-AI-Agent/
├── walmart_assistant.py          # Main application (voice + web)
├── templates/
│   └── index.html               # Web interface template
├── walmart_products_db/         # Vector database storage
├── walmart_orders_500plus.csv   # Walmart product data
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Ollama**:
   ```bash
   # Install Ollama (https://ollama.ai/)
   ollama serve
   ollama pull gemma3:1b
   ollama pull mxbai-embed-large
   ```

3. **Set up Google Cloud TTS** (optional, for voice output):
   ```bash
   # Set up Google Cloud credentials
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   ```

## Usage

### Voice Interface
```bash
python walmart_assistant.py
```

### Web Interface
```bash
python walmart_assistant.py web
```
Then visit `http://localhost:5000`

## Example Interactions

### Product Search
- "Find me headphones"
- "Show me electronics"
- "I need a laptop"
- "What phones do you have?"

### Shopping Cart
- "Add iPhone to my cart"
- "Show my cart"
- "Clear my cart"
- "What's in my cart?"

## Technical Architecture

- **Voice Processing**: Whisper (STT) + Google Cloud TTS
- **AI/ML**: Ollama with Gemma 3B + Vector embeddings
- **Web Framework**: Flask + Modern HTML/CSS/JS
- **Data**: Mock Walmart product database with 500+ products
- **Vector Database**: Chroma with semantic search

## Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   ollama serve
   ```

2. **Model not found**:
   ```bash
   ollama pull gemma3:1b
   ollama pull mxbai-embed-large
   ```

3. **Google TTS errors**:
   - Ensure Google Cloud credentials are set
   - Check billing is enabled for the project

## License

This project is for educational and demonstration purposes.
