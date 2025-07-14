# Walmart Voice AI Assistant

An intelligent voice-powered shopping assistant that provides a natural conversational interface for product search, cart management, order placement, and order tracking. Built with AI models for speech recognition, natural language processing, and vector search capabilities.

## Features

### 🛒 Shopping & Cart Management
- **Product Search**: AI-powered semantic search across 2000+ products
- **Add to Cart**: Voice commands like "Add Sony headphones to cart"
- **Cart Management**: View, modify, and clear shopping cart
- **Smart Transcription**: Handles common speech errors (e.g., "card" vs "cart")

### 📦 Order Management
- **Order Placement**: Complete checkout flow with address and payment selection
- **Dynamic Order Tracking**: Real-time status updates that change daily
- **Order History**: View all current and past orders
- **Delivery Estimates**: Smart delivery predictions based on order age

### 🎤 Voice Interface
- **Speech-to-Text**: Whisper AI model for accurate transcription
- **Text-to-Speech**: Google Cloud TTS with natural voice responses
- **Voice Commands**: Natural language processing for intuitive interaction
- **Conversation Flow**: Context-aware multi-step processes (checkout, etc.)

### 📊 Data Management
- **User Authentication**: Secure user login system
- **Persistent Storage**: CSV-based data storage for users, orders, and carts
- **Address & Payment**: Saved user addresses and payment methods
- **Order Analytics**: 15-day return/refund eligibility tracking

## Installation

### Prerequisites
- Python 3.8+
- Microphone for voice input
- Google Cloud TTS credentials (optional, fallback TTS available)
- Ollama with required models

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Voice-AI-Agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama models**
```bash
ollama pull gemma3:1b
ollama pull mxbai-embed-large
```

4. **Setup Google Cloud TTS (Optional)**
- Place your Google Cloud credentials JSON file in the project root
- Name it `diptak_tts[1].json` or update the path in code

5. **Initialize data files**
The following CSV files should be present:
- `users.csv` - User accounts
- `user_addresses.csv` - User delivery addresses  
- `user_payment_methods.csv` - User payment methods
- `walmart_inventory.csv` - Product inventory
- `user_orders.csv` - Order history (created automatically)
- `user_carts.csv` - Shopping carts (created automatically)

## Usage

### Voice Interface (Recommended)
```bash
python walmart_assistant.py
```

### Web Interface
```bash
python walmart_assistant.py web
```

### Streamlit Interface
```bash
streamlit run streamlit_app.py
```

## Voice Commands

### Product Search
- "Find me headphones"
- "What laptops do you have?"
- "Show me gaming chairs"

### Cart Management
- "Add Sony headphones to cart"
- "Show my cart"
- "Remove from cart"
- "Why is [item] in my cart?"

### Order Management
- "Place order" / "Checkout"
- "Where is my order?"
- "When will my order be delivered?"
- "Show my orders" / "Order history"
- "Check refund status"

### General
- "What do you recommend?"
- "Stop speaking" (interrupt current speech)

## User Authentication

Login with predefined user IDs:
- **U001**: John Doe
- **U002**: Jane Smith

Each user has:
- Saved addresses
- Payment methods
- Personal shopping cart
- Order history

## Architecture

### Core Components
- **WalmartAssistant**: Main application class
- **VoiceManager**: Handles speech-to-text and text-to-speech
- **Vector Database**: Chroma DB for semantic product search
- **CSV Storage**: Persistent data storage system

### AI Models
- **Speech Recognition**: OpenAI Whisper (tiny model)
- **Language Model**: Ollama Gemma3:1b
- **Embeddings**: mxbai-embed-large
- **Text-to-Speech**: Google Cloud TTS with fallback

### Data Flow
1. Voice input → Whisper transcription
2. Text query → Intent classification
3. Intent → Appropriate handler function
4. Response generation → TTS output
5. Data persistence → CSV files

## Configuration

### Voice Settings
- Sample rate: 16kHz (recording), 24kHz (playback)
- Model: Whisper tiny (for speed)
- TTS: Google Cloud with pyttsx3 fallback

### Database
- Vector store: Chroma (persistent)
- Product search: Top 5 results
- Embedding model: mxbai-embed-large

## Deployment

### Local Development
```bash
python walmart_assistant.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Set entry point: `streamlit_app.py`
4. Deploy automatically

### Requirements for Deployment
- All CSV data files
- Google Cloud credentials (if using)
- Ollama models accessible
- Sufficient memory for AI models

## File Structure

```
Voice-AI-Agent/
├── walmart_assistant.py      # Main application
├── streamlit_app.py          # Streamlit web interface  
├── requirements.txt          # Python dependencies
├── users.csv                 # User accounts
├── user_addresses.csv        # User addresses
├── user_payment_methods.csv  # Payment methods
├── walmart_inventory.csv     # Product catalog
├── user_orders.csv          # Order history
├── user_carts.csv           # Shopping carts
├── diptak_tts[1].json       # Google Cloud credentials
└── walmart_inventory_db/    # Vector database
```

## Troubleshooting

### Common Issues

**Audio not working**
- Check microphone permissions
- Ensure pygame audio system is initialized
- Verify Google Cloud TTS credentials

**Models not loading**
- Install Ollama and required models
- Check available memory (models require ~2GB)
- Verify Ollama service is running

**CSV file errors**
- Ensure all required CSV files exist
- Check file permissions
- Verify CSV format and headers

**Voice recognition issues**
- Speak clearly and wait for prompt
- Check microphone input levels
- Try text input for testing

### Performance Tips
- Use "tiny" Whisper model for faster transcription
- Close other applications to free memory
- Use SSD storage for faster model loading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs in console
3. Ensure all dependencies are installed
4. Verify CSV data file formats 