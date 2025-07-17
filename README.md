# Waltz Voice AI Assistant

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

4. **Setup Google Cloud TTS (Optional)**
- Place your Google Cloud credentials JSON file in the project root


5. **Initialize data files**
The following CSV files should be present:
- `users.csv` - User accounts
- `user_addresses.csv` - User delivery addresses  
- `user_payment_methods.csv` - User payment methods
- `waltz_inventory.csv` - Product inventory
- `user_orders.csv` - Order history 
- `user_carts.csv` - Shopping carts 

## Usage

### Voice Interface
```bash
python waltz_assistant.py
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
- **WaltzAssistant**: Main application class
- **VoiceManager**: Handles speech-to-text and text-to-speech
- **Vector Database**: Chroma DB for semantic product search
- **CSV Storage**: Persistent data storage system

### AI Models
- **Speech Recognition**: OpenAI Whisper (tiny model)
- **Language Model**: Google Gemini (via Google AI Studio)
- **Embeddings**: Google Generative AI Embeddings
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
- Embedding model: Google Generative AI Embeddings

## Deployment

### Local Development
```bash
python waltz_assistant.py
```

### Requirements for Deployment
- All CSV data files
- Google Cloud credentials (if using)
- Sufficient memory for AI models

## File Structure

```
Voice-AI-Agent/
├── waltz_assistant.py         # Main application (Flask, AI, all logic)
├── requirements.txt           # Python dependencies
├── waltz_inventory.csv        # Product catalog (for ChromaDB)
├── waltz_inventory_db/        # ChromaDB vector database (products)
├── templates/
│   └── waltz.html             # Web UI template
├── index.html                 # Web UI
├── app.py                     #  Flask app
├── .gitignore                 # Git ignore rules
├── venv/                      # Python virtual environment
└── voiceaiagent-*.json        # Google Cloud credentials 
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

This project is for educational and demonstration purposes.

