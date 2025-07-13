#!/usr/bin/env python3
"""
Walmart Voice Assistant - Core functionality for AI-powered shopping assistant
"""

import sounddevice as sd
import whisper
import numpy as np
import scipy.io.wavfile as wav
import os
import json
import io
import base64
import threading
import queue
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, Response, stream_template
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import texttospeech
import pandas as pd
import pygame
import tempfile
import wave

# Constants
RECORD_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

class VoiceManager:
    """Voice management with real-time capabilities"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.audio_thread = None
        self.setup_audio()
        
    def setup_audio(self):
        """Initialize audio system"""
        try:
            pygame.mixer.init(frequency=TTS_SAMPLE_RATE, size=-16, channels=1)
            print("Audio system initialized")
        except Exception as e:
            print(f"Audio system error: {e}")
            
    def text_to_speech(self, text, voice_name="en-US-Wavenet-C", language_code="en-US"):
        """Convert text to speech with enhanced capabilities"""
        print(f"Speaking: {text[:50]}...")
        
        try:
            # Check Google Cloud credentials
            if not os.path.exists(os.path.expanduser('~/.config/gcloud/application_default_credentials.json')):
                print("Google Cloud credentials not found. Using fallback TTS.")
                return self.fallback_tts(text)
            
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=TTS_SAMPLE_RATE,
                speaking_rate=0.9,
                pitch=0.0,
                volume_gain_db=0.0
            )
            
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            audio_array = np.frombuffer(response.audio_content, dtype=np.int16)
            return audio_array
            
        except Exception as e:
            print(f"TTS error: {e}")
            return self.fallback_tts(text)
    
    def fallback_tts(self, text):
        """Fallback TTS using system text-to-speech"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            engine.save_to_file(text, temp_file.name)
            engine.runAndWait()
            
            with wave.open(temp_file.name, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16)
            
            os.unlink(temp_file.name)
            return audio_array
            
        except Exception as e:
            print(f"Fallback TTS error: {e}")
            return None
    
    def speak_async(self, text):
        """Speak text asynchronously"""
        def speak_thread():
            audio_array = self.text_to_speech(text)
            if audio_array is not None:
                self.is_speaking = True
                sd.play(audio_array, samplerate=TTS_SAMPLE_RATE)
                sd.wait()
                self.is_speaking = False
                print("Speech complete")
        
        self.audio_thread = threading.Thread(target=speak_thread)
        self.audio_thread.start()
    
    def speak_sync(self, text):
        """Speak text synchronously"""
        audio_array = self.text_to_speech(text)
        if audio_array is not None:
            self.is_speaking = True
            sd.play(audio_array, samplerate=TTS_SAMPLE_RATE)
            sd.wait()
            self.is_speaking = False
            print("Speech complete")
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.is_speaking:
            sd.stop()
            self.is_speaking = False
            print("Speech stopped")

class WalmartAssistant:
    def __init__(self):
        self.setup_models()
        self.setup_vector_database()
        self.setup_shopping_cart()
        self.voice_manager = VoiceManager()
        
    def setup_models(self):
        """Initialize AI models"""
        print("Loading models...")
        
        # Speech-to-text
        self.stt_model = whisper.load_model("base")
        print("Whisper model loaded")
        
        # LLM
        try:
            self.llm = OllamaLLM(model="gemma3:1b")
            print("Ollama model loaded")
        except Exception as e:
            print(f"Error loading Ollama model: {e}")
            self.llm = None
            
        # Embeddings
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        print("Embeddings model loaded")
        
    def setup_vector_database(self):
        """Setup Walmart product vector database"""
        print("Setting up product database...")
        
        # Load products
        try:
            # Try to load the new CSV with delivery status first
            csv_file = "walmart_orders_with_delivery.csv"
            if not os.path.exists(csv_file):
                csv_file = "walmart_orders_inr_only.csv"
                print("Using original CSV file.")
            
            df = pd.read_csv(csv_file)
            unique_products = df.drop_duplicates(subset=['product_id']).copy()
            print(f"Found {len(unique_products)} unique products from {csv_file}")
        except FileNotFoundError:
            print("Product CSV file not found. Please ensure CSV file exists.")
            unique_products = pd.DataFrame()
        except Exception as e:
            print(f"Error loading CSV: {e}")
            unique_products = pd.DataFrame()
        
        # Create vector database
        db_location = "./walmart_products_db"
        add_documents = not os.path.exists(db_location)
        
        if add_documents:
            documents = []
            ids = []
            
            for i, row in unique_products.iterrows():
                # Create comprehensive searchable product content
                product_name = str(row['product_name']).lower()
                category = str(row['product_category']).lower()
                description = str(row['product_description']).lower()
                
                # Handle different price column names
                price_column = 'price_inr' if 'price_inr' in row else 'price'
                price = row[price_column]
                
                # Add delivery status if available
                delivery_info = ""
                if 'delivery_status' in row:
                    delivery_info = f"\nDelivery Status: {row['delivery_status']}"
                    if (row['delivery_status'] == 'Delivered' and 
                        'delivery_date' in row):
                        delivery_date = row['delivery_date']
                        if delivery_date and str(delivery_date) != 'nan' and str(delivery_date) != 'None':
                            delivery_info += f" (Delivered on {delivery_date})"
                
                # Create searchable content with multiple variations
                product_content = f"""
Product: {row['product_name']}
Category: {row['product_category']}
Description: {row['product_description']}
Price: ₹{price}
ID: {row['product_id']}{delivery_info}

Search Terms: {product_name} {category} {description}
                """.strip()
                
                document = Document(
                    page_content=product_content,
                    metadata={
                        "product_id": row['product_id'],
                        "product_name": row['product_name'],
                        "category": row['product_category'],
                        "price": float(price),
                        "description": row['product_description'],
                        "delivery_status": row.get('delivery_status', 'Unknown')
                    },
                    id=str(row['product_id'])
                )
                ids.append(str(row['product_id']))
                documents.append(document)
            
            # Create and populate vector store
            self.vector_store = Chroma(
                collection_name="walmart_products",
                persist_directory=db_location,
                embedding_function=self.embeddings
            )
            self.vector_store.add_documents(documents=documents, ids=ids)
            print("Product database created")
        else:
            self.vector_store = Chroma(
                collection_name="walmart_products",
                persist_directory=db_location,
                embedding_function=self.embeddings
            )
            print("Using existing product database")
            
        # Create retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
    def setup_shopping_cart(self):
        """Initialize shopping cart"""
        self.cart = []
        self.user_session = {"customer_id": "C001"}
        
    def record_audio(self, filename="input.wav", duration=5):
        """Record audio from microphone"""
        print("Recording...")
        recording = sd.rec(int(duration * RECORD_SAMPLE_RATE), 
                          samplerate=RECORD_SAMPLE_RATE, 
                          channels=1, dtype='int16')
        sd.wait()
        wav.write(filename, RECORD_SAMPLE_RATE, recording)
        print("Recording complete")
        return filename
        
    def transcribe(self, filename):
        """Convert audio to text"""
        print("Transcribing...")
        try:
            if not os.path.exists(filename):
                print(f"Audio file not found: {filename}")
                return ""
            result = self.stt_model.transcribe(filename)
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def speak(self, text, async_mode=True):
        """Enhanced speak function with async option"""
        if async_mode:
            self.voice_manager.speak_async(text)
        else:
            self.voice_manager.speak_sync(text)
    
    def stop_speaking(self):
        """Stop current speech"""
        self.voice_manager.stop_speaking()
            
    def search_products(self, query):
        """Search for products"""
        try:
            results = self.retriever.invoke(query)
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
            
    def process_query(self, query):
        """Process user query and generate response"""
        if not self.llm:
            return "Sorry, the AI model is not available."
            
        # Search for products
        products = self.search_products(query)
        
        # Format products
        products_text = ""
        if products:
            if hasattr(products, '__iter__'):
                products_text = "\n".join([doc.page_content for doc in products])
            else:
                products_text = str(products)
        else:
            products_text = "No products found"
            
        # Format cart
        cart_text = ""
        if self.cart:
            cart_text = "\n".join([f"- {item['name']} (₹{item['price']})" for item in self.cart])
        else:
            cart_text = "Empty"
            
        # Enhanced prompt for better voice responses
        template = """
You are a helpful Walmart shopping assistant with a friendly, conversational voice. 
Help customers find products and manage their cart. Speak naturally and clearly.

Shopping cart: {cart}
Available products: {products}

User query: {query}

Provide a helpful, conversational response that sounds natural when spoken aloud.
- If products are found, mention them clearly with prices
- If the user wants to add items to cart, confirm the action clearly
- Keep responses concise but informative
- Use natural speech patterns
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        result = chain.invoke({
            "cart": cart_text,
            "products": products_text,
            "query": query
        })
        
        return str(result)
        
    def add_to_cart(self, product_name, price):
        """Add item to shopping cart"""
        item = {
            "name": product_name,
            "price": price,
            "added_at": datetime.now().isoformat()
        }
        self.cart.append(item)
        return f"Added {product_name} to cart for ₹{price}"
        
    def get_cart_total(self):
        """Get cart total"""
        return sum(item['price'] for item in self.cart)
        
    def clear_cart(self):
        """Clear shopping cart"""
        self.cart.clear()
        return "Cart cleared"

# Create global assistant instance
assistant = WalmartAssistant()

# Flask app for web interface
app = Flask(__name__)
app.secret_key = 'walmart_assistant_secret'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('message', '')
        speak_response = data.get('speak', True)
        
        if not query.strip():
            return jsonify({'error': 'No message provided'}), 400
            
        response = assistant.process_query(query)
        
        # Speak response if requested
        if speak_response:
            assistant.speak(response, async_mode=True)
        
        return jsonify({
            'response': response,
            'cart_count': len(assistant.cart),
            'cart_total': assistant.get_cart_total()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak_text():
    """Speak text via API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        assistant.speak(text, async_mode=True)
        return jsonify({'message': 'Speaking started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speak/stop', methods=['POST'])
def stop_speaking():
    """Stop current speech"""
    try:
        assistant.stop_speaking()
        return jsonify({'message': 'Speech stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cart', methods=['GET'])
def get_cart():
    return jsonify({
        'cart': assistant.cart,
        'total': assistant.get_cart_total(),
        'count': len(assistant.cart)
    })

@app.route('/api/cart/clear', methods=['POST'])
def clear_cart():
    message = assistant.clear_cart()
    return jsonify({
        'message': message,
        'cart_count': 0,
        'cart_total': 0
    })

@app.route('/api/voice', methods=['POST'])
def process_voice():
    """Process voice input from web interface"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
            
        # Save uploaded audio
        audio_path = "web_input.wav"
        audio_file.save(audio_path)
        
        # Transcribe audio
        query = assistant.transcribe(audio_path)
        
        if not query.strip():
            return jsonify({'error': 'No speech detected'}), 400
            
        # Process query
        response = assistant.process_query(query)
        
        # Speak response
        assistant.speak(response, async_mode=True)
        
        return jsonify({
            'query': query,
            'response': response,
            'cart_count': len(assistant.cart),
            'cart_total': assistant.get_cart_total()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def voice_interface():
    """Command line voice interface"""
    print("\nWalmart Voice Assistant")
    print("Press Enter to speak, or type 'q' to quit")
    print("Example queries:")
    print("- 'Find me headphones'")
    print("- 'Show me electronics'")
    print("- 'Add iPhone to cart'")
    print("- 'Show my cart'")
    print("- 'Stop speaking' to interrupt current speech")
    
    while True:
        print("\n" + "="*50)
        mode = input("Press Enter to talk (or type 'q' to quit): ")
        if mode.strip().lower() == "q":
            print("Thank you for shopping with Walmart!")
            break
            
        # Record and transcribe
        audio_file = "input.wav"
        assistant.record_audio(filename=audio_file)
        query = assistant.transcribe(audio_file)
        
        if not query.strip():
            print("No speech detected. Please try again.")
            continue
            
        print(f"You said: {query}")
        
        # Check for stop command
        if "stop speaking" in query.lower():
            assistant.stop_speaking()
            print("Speech stopped")
            continue
        
        # Process query
        response = assistant.process_query(query)
        print(f"Assistant: {response}")
        
        # Speak response
        assistant.speak(response, async_mode=False)

def web_interface():
    """Start web interface"""
    print("Starting web interface...")
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        web_interface()
    else:
        voice_interface() 