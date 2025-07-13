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
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, Response, stream_template
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import texttospeech
import pandas as pd
import numpy as np
import pygame
import tempfile
import wave
import uuid

# Constants
RECORD_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Mock user data
MOCK_USERS = {
    "U001": {
        "name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "+1-555-0123",
        "addresses": [
            {
                "id": "A001",
                "type": "Home",
                "street": "123 Main Street",
                "city": "New York",
                "state": "NY",
                "zipcode": "10001",
                "country": "USA"
            },
            {
                "id": "A002", 
                "type": "Work",
                "street": "456 Business Ave",
                "city": "New York", 
                "state": "NY",
                "zipcode": "10002",
                "country": "USA"
            }
        ],
        "payment_methods": [
            {
                "id": "P001",
                "type": "Credit Card",
                "last4": "1234",
                "expiry": "12/25"
            },
            {
                "id": "P002",
                "type": "PayPal",
                "email": "john.doe@email.com"
            }
        ]
    },
    "U002": {
        "name": "Jane Smith",
        "email": "jane.smith@email.com", 
        "phone": "+1-555-0456",
        "addresses": [
            {
                "id": "A003",
                "type": "Home",
                "street": "789 Oak Drive",
                "city": "Los Angeles",
                "state": "CA", 
                "zipcode": "90210",
                "country": "USA"
            }
        ],
        "payment_methods": [
            {
                "id": "P003",
                "type": "Credit Card",
                "last4": "5678",
                "expiry": "06/26"
            }
        ]
    }
}

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
            # Check Google Cloud credentials - try multiple paths
            cred_paths = [
                "diptak_tts[1].json",
                "diptak_tts.json", 
                os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
            ]
            
            cred_found = False
            for cred_path in cred_paths:
                if os.path.exists(cred_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
                    cred_found = True
                    print(f"Using Google Cloud credentials from: {cred_path}")
                    break
            
            if not cred_found:
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
        self.current_user = None
        self.checkout_state = None
        
    def setup_models(self):
        """Initialize AI models"""
        print("Loading models...")
        
        # Speech-to-text - use smaller model to avoid memory issues
        try:
            self.stt_model = whisper.load_model("tiny")
            print("Whisper model loaded (tiny)")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.stt_model = None
        
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
        """Setup Walmart product inventory database"""
        print("Setting up product inventory database...")
        
        # Load inventory
        try:
            inventory_file = "walmart_inventory.csv"
            if not os.path.exists(inventory_file):
                print(f"Inventory file not found: {inventory_file}")
                return
                
            df = pd.read_csv(inventory_file)
            print(f"Loaded {len(df)} products from inventory")
            
            # Filter only in-stock items for search
            in_stock_df = df[df['stock_status'] == 'In Stock'].copy()
            print(f"Found {len(in_stock_df)} in-stock products")
            
        except FileNotFoundError:
            print("Inventory CSV file not found. Please ensure 'walmart_inventory.csv' exists.")
            return
        except Exception as e:
            print(f"Error loading inventory: {e}")
            return
        
        # Create vector database
        db_location = "./walmart_inventory_db"
        add_documents = not os.path.exists(db_location)
        
        if add_documents:
            documents = []
            ids = []
            
            for i, row in in_stock_df.iterrows():
                # Create comprehensive searchable product content
                product_name = str(row['product_name']).lower()
                category = str(row['product_category']).lower()
                description = str(row['product_description']).lower()
                price = row['price_inr']
                stock_qty = row['stock_quantity']
                
                # Create searchable content with inventory info
                product_content = f"""
Product: {row['product_name']}
Category: {row['product_category']}
Description: {row['product_description']}
Price: ₹{price:,}
Stock: {stock_qty} units available
ID: {row['product_id']}

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
                        "stock_status": row['stock_status'],
                        "stock_quantity": int(stock_qty)
                    },
                    id=str(row['product_id'])
                )
                ids.append(str(row['product_id']))
                documents.append(document)
            
            # Create and populate vector store
            self.vector_store = Chroma(
                collection_name="walmart_inventory",
                persist_directory=db_location,
                embedding_function=self.embeddings
            )
            self.vector_store.add_documents(documents=documents, ids=ids)
            print("Inventory database created")
        else:
            self.vector_store = Chroma(
                collection_name="walmart_inventory",
                persist_directory=db_location,
                embedding_function=self.embeddings
            )
            print("Using existing inventory database")
            
        # Create retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
    def load_user_cart(self, user_id):
        """Load user's cart from CSV"""
        try:
            if not os.path.exists("user_carts.csv"):
                return []
                
            df = pd.read_csv("user_carts.csv")
            user_cart = df[(df['user_id'] == user_id) & (df['status'] == 'active')]
            
            cart_items = []
            for _, row in user_cart.iterrows():
                item = {
                    "id": row['cart_id'],
                    "product_id": row['product_id'],
                    "name": row['product_name'],
                    "price": float(row['price_inr']),
                    "quantity": int(row['quantity']),
                    "added_at": row['added_at']
                }
                cart_items.append(item)
                
            return cart_items
            
        except Exception as e:
            print(f"Error loading user cart: {e}")
            return []
            
    def save_user_cart(self, user_id, cart_items):
        """Save user's cart to CSV"""
        try:
            # Load existing carts
            if os.path.exists("user_carts.csv"):
                df = pd.read_csv("user_carts.csv")
            else:
                df = pd.DataFrame({
                    'cart_id': [],
                    'user_id': [],
                    'product_id': [],
                    'product_name': [],
                    'price_inr': [],
                    'quantity': [],
                    'added_at': [],
                    'status': []
                })
            
            # Remove old active items for this user
            df = df[~((df['user_id'] == user_id) & (df['status'] == 'active'))]
            
            # Add new cart items
            for item in cart_items:
                new_row = {
                    'cart_id': f"CART{user_id}",
                    'user_id': user_id,
                    'product_id': item['product_id'],
                    'product_name': item['name'],
                    'price_inr': item['price'],
                    'quantity': item['quantity'],
                    'added_at': item['added_at'],
                    'status': 'active'
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            df.to_csv("user_carts.csv", index=False)
            
        except Exception as e:
            print(f"Error saving user cart: {e}")
            
    def setup_shopping_cart(self):
        """Initialize shopping cart"""
        self.cart = []
        self.user_session = {"customer_id": None}
        self.cart_history = []
        
    def authenticate_user(self, user_id):
        """Authenticate user and load their cart"""
        if user_id in MOCK_USERS:
            self.current_user = MOCK_USERS[user_id]
            self.user_session["customer_id"] = user_id
            
            # Load user's existing cart
            self.cart = self.load_user_cart(user_id)
            print(f"Loaded {len(self.cart)} items from user's cart")
            
            return True
        return False
        
    def get_user_info(self):
        """Get current user information"""
        if self.current_user:
            return {
                "name": self.current_user["name"],
                "email": self.current_user["email"],
                "addresses": self.current_user["addresses"],
                "payment_methods": self.current_user["payment_methods"]
            }
        return None
        
    def record_audio(self, duration=None):
        """Record audio from microphone until stopped manually"""
        print("Recording... Press Enter to stop recording")
        
        # Global variables for recording
        recording_data = []
        recording_finished = False
        
        def callback(indata, frames, time, status):
            """Callback function for recording"""
            if status:
                print(f"Recording status: {status}")
            recording_data.append(indata.copy())
        
        try:
            # Start recording stream
            with sd.InputStream(callback=callback, 
                              channels=1, 
                              samplerate=RECORD_SAMPLE_RATE,
                              dtype='int16'):
                
                print("Recording started. Press Enter to stop...")
                
                try:
                    input()  # Wait for Enter key
                    recording_finished = True
                    print("Recording stopped by user")
                    
                except KeyboardInterrupt:
                    print("Recording interrupted")
                    recording_finished = True
                    
        except Exception as e:
            print(f"Recording error: {e}")
            return None
            
        if recording_data and recording_finished:
            # Combine all recorded chunks
            combined_data = np.concatenate(recording_data, axis=0)
            return combined_data
        else:
            return None
        
    def transcribe(self, audio_array):
        """Convert audio array to text with in-memory processing"""
        print("Transcribing...")
        
        if not self.stt_model:
            print("Whisper model not available")
            return ""
            
        if audio_array is None:
            print("No audio recorded (interrupted)")
            return ""
        
        try:
            # Ensure audio_array is 2D (samples, channels)
            if len(audio_array.shape) == 1:
                audio_array = audio_array.reshape(-1, 1)
            
            # Convert audio array to the format Whisper expects
            # Whisper expects float32 audio in the range [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # If stereo, convert to mono by averaging channels
            if audio_float.shape[1] > 1:
                audio_float = np.mean(audio_float, axis=1, keepdims=True)
            
            # Reshape to 1D for Whisper
            audio_float = audio_float.flatten()
            
            print(f"Audio shape: {audio_float.shape}, sample rate: {RECORD_SAMPLE_RATE}")
            
            # Transcribe with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Transcription attempt {attempt + 1}...")
                    
                    # Use Whisper's transcribe method with audio array
                    result = self.stt_model.transcribe(
                        audio_float,
                        language="en",
                        task="transcribe"
                    )
                    
                    if result and "text" in result:
                        transcribed_text = str(result["text"]).strip()
                        print(f"Transcription successful: '{transcribed_text}'")
                        return transcribed_text
                    else:
                        print(f"Whisper returned empty result on attempt {attempt + 1}")
                        
                except Exception as e:
                    print(f"Transcription attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)  # Longer wait between retries
                    else:
                        raise e
            
            return ""
            
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
            
        # Handle user authentication
        if "login" in query.lower() or "sign in" in query.lower():
            return self.handle_login(query)
            
        # Handle order status queries
        if any(keyword in query.lower() for keyword in ["order status", "where is my order", "track order", "order tracking"]):
            return self.handle_order_status(query)
            
        # Handle replacement/refund status
        if any(keyword in query.lower() for keyword in ["refund", "replacement", "return", "refund status", "replacement status"]):
            return self.handle_refund_replacement_status(query)
            
        # Handle product recommendations
        if any(keyword in query.lower() for keyword in ["recommend", "recommendation", "suggest", "what should I buy", "what do you recommend"]):
            return self.handle_product_recommendations(query)
            
        # Handle checkout process
        if "checkout" in query.lower() or "place order" in query.lower():
            return self.handle_checkout(query)
            
        # Handle cart operations
        if "add" in query.lower() and "cart" in query.lower():
            return self.handle_add_to_cart(query)
            
        # Handle cart confirmation
        if any(keyword in query.lower() for keyword in ["yes", "add it", "add to cart", "confirm"]):
            return self.handle_cart_confirmation(query)
            
        if "remove" in query.lower() and "cart" in query.lower():
            return self.handle_remove_from_cart(query)
            
        if "show cart" in query.lower() or "my cart" in query.lower():
            return self.handle_show_cart(query)
            
        # Search for products
        products = self.search_products(query)
        
        # Format available products (IN STOCK)
        products_text = ""
        if products:
            if hasattr(products, '__iter__'):
                products_text = "Available products in stock:\n" + "\n".join([doc.page_content for doc in products])
            else:
                products_text = "Available products in stock:\n" + str(products)
        else:
            products_text = "No products found in stock"
            
        # Format cart with explicit state information (USER'S PERSONAL CART)
        cart_text = ""
        cart_count = len(self.cart)
        cart_total = self.get_cart_total()
        
        if self.cart:
            cart_items = []
            for item in self.cart:
                cart_items.append(f"- {item['name']} (₹{item['price']})")
            cart_text = f"YOUR PERSONAL CART has {cart_count} items:\n" + "\n".join(cart_items) + f"\nTotal: ₹{cart_total}"
        else:
            cart_text = "YOUR PERSONAL CART is empty (0 items)"
            
        # Enhanced prompt for better voice responses with explicit separation
        template = """
You are a helpful Walmart shopping assistant with a friendly, conversational voice. 
Help customers find products and manage their cart. Speak naturally and clearly.

CRITICAL RULES - NEVER BREAK THESE:
1. INVENTORY = Available products in stock that customers can buy
2. CART = Items the user has personally added to their shopping cart
3. NEVER mention cart items unless they are actually in the user's cart
4. NEVER confuse inventory items with cart items
5. ONLY recommend products from inventory
6. ONLY mention cart items that exist in the user's personal cart

Current user: {user_info}
AVAILABLE INVENTORY (In Stock Products): {products}
USER'S PERSONAL CART: {cart}

User query: {query}

Provide a helpful, conversational response that sounds natural when spoken aloud.
- If products are found in inventory, mention them clearly with prices
- If the user wants to add items to cart, confirm the action clearly
- ONLY mention cart items that actually exist in the user's personal cart above
- NEVER invent or hallucinate cart contents
- Keep responses concise but informative
- Use natural speech patterns
        """
        
        user_info = f"Logged in as {self.current_user['name']}" if self.current_user else "Not logged in"
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        result = chain.invoke({
            "user_info": user_info,
            "cart": cart_text,
            "products": products_text,
            "query": query
        })
        
        return str(result)
        
    def handle_login(self, query):
        """Handle user login"""
        if not self.current_user:
            return "Please provide your user ID to login. Say something like 'Login with user ID U001'"
        
        return f"Welcome back, {self.current_user['name']}! You are now logged in."
        
    def add_to_cart(self, product_name, price, product_id=None):
        """Add item to shopping cart with validation"""
        if not self.current_user:
            return "Please login first to add items to your cart."
            
        # Validate inputs
        if not product_name or not price:
            return "Invalid product information."
            
        # Create cart item with unique ID
        item = {
            "id": str(uuid.uuid4()),
            "product_id": product_id,
            "name": str(product_name),
            "price": float(price),
            "quantity": 1,
            "added_at": datetime.now().isoformat(),
            "user_id": self.user_session.get("customer_id")
        }
        
        self.cart.append(item)
        self.cart_history.append({"action": "add", "item": item, "timestamp": datetime.now()})
        
        # Save to CSV
        self.save_user_cart(self.user_session.get("customer_id"), self.cart)
        
        print(f"DEBUG: Added to cart - {item['name']} (₹{item['price']})")
        print(f"DEBUG: Cart now has {len(self.cart)} items")
        
        return f"Added {product_name} to your cart for ₹{price:,}. Your cart now has {len(self.cart)} items."
        
    def remove_from_cart(self, item_id=None):
        """Remove item from cart"""
        if not self.cart:
            return "Your cart is empty."
            
        if item_id:
            # Remove specific item by ID
            for i, item in enumerate(self.cart):
                if item["id"] == item_id:
                    removed_item = self.cart.pop(i)
                    self.cart_history.append({"action": "remove", "item": removed_item, "timestamp": datetime.now()})
                    
                    # Save to CSV
                    self.save_user_cart(self.user_session.get("customer_id"), self.cart)
                    
                    return f"Removed {removed_item['name']} from your cart. Your cart now has {len(self.cart)} items."
            return "Item not found in cart."
        else:
            # Remove last item
            removed_item = self.cart.pop()
            self.cart_history.append({"action": "remove", "item": removed_item, "timestamp": datetime.now()})
            
            # Save to CSV
            self.save_user_cart(self.user_session.get("customer_id"), self.cart)
            
            return f"Removed {removed_item['name']} from your cart. Your cart now has {len(self.cart)} items."
        
    def get_cart_contents(self):
        """Get formatted cart contents"""
        if not self.cart:
            return "Your cart is empty."
            
        cart_items = []
        total = 0
        
        for item in self.cart:
            cart_items.append(f"- {item['name']} (₹{item['price']:,}) x{item['quantity']}")
            total += item['price'] * item['quantity']
            
        cart_text = "\n".join(cart_items)
        return f"Your cart contains:\n{cart_text}\nTotal: ₹{total:,}"
        
    def get_cart_total(self):
        """Get cart total"""
        return sum(item['price'] * item['quantity'] for item in self.cart)
        
    def clear_cart(self):
        """Clear shopping cart"""
        self.cart.clear()
        self.cart_history.append({"action": "clear", "timestamp": datetime.now()})
        
        # Save to CSV
        self.save_user_cart(self.user_session.get("customer_id"), self.cart)
        
        return "Cart cleared"
        
    def handle_add_to_cart(self, query):
        """Handle adding items to cart with confirmation"""
        # Validate operation
        is_valid, message = self.validate_cart_operation("add")
        if not is_valid:
            return message
            
        # Extract product name from query
        products = self.search_products(query)
        if not products:
            return "I couldn't find that product. Please try searching for it first."
            
        # Get the first product found
        product_doc = products[0]
        product_name = product_doc.metadata.get('product_name', 'Unknown Product')
        product_price = product_doc.metadata.get('price_inr', 0)
        product_id = product_doc.metadata.get('product_id', '')
        
        # Check if product is in stock
        try:
            inventory_df = pd.read_csv('walmart_inventory.csv')
            product_row = inventory_df[inventory_df['product_id'] == product_id]
            
            if product_row.empty:
                return f"Sorry, {product_name} is not available in our inventory."
                
            stock_quantity = product_row.iloc[0]['stock_quantity']
            if stock_quantity <= 0:
                return f"Sorry, {product_name} is currently out of stock."
                
            # Store pending product for confirmation
            self.pending_product = {
                'name': product_name,
                'price': product_price,
                'id': product_id,
                'stock': stock_quantity
            }
            
            # Confirm with user
            confirmation_response = f"I found {product_name} for ₹{product_price:,} with {stock_quantity} in stock. "
            confirmation_response += "Would you like me to add this to your cart? Say 'yes' to add or 'no' to cancel."
            
            return confirmation_response
            
        except Exception as e:
            return f"Sorry, I couldn't check the product availability. Error: {str(e)}"
            
    def confirm_add_to_cart(self, product_name, product_price, product_id):
        """Confirm and add item to cart"""
        if not self.current_user:
            return "Please login first to add items to your cart."
            
        # Add to cart
        result = self.add_to_cart(product_name, product_price, product_id)
        
        # Ask if user wants to place order or get more recommendations
        result += "\n\nWould you like to place your order now or would you like more product recommendations?"
        
        return result
        
    def handle_cart_confirmation(self, query):
        """Handle cart confirmation responses"""
        if not self.current_user:
            return "Please login first to manage your cart."
            
        # Check if we have a pending product to add
        if hasattr(self, 'pending_product') and self.pending_product:
            product_name = self.pending_product['name']
            product_price = self.pending_product['price']
            product_id = self.pending_product['id']
            
            # Add to cart
            result = self.add_to_cart(product_name, product_price, product_id)
            
            # Clear pending product
            self.pending_product = None
            
            # Ask next steps
            result += "\n\nWould you like to place your order now or would you like more product recommendations?"
            
            return result
        else:
            return "I don't have a product waiting to be added to your cart. Please search for a product first."
        
    def handle_remove_from_cart(self, query):
        """Handle removing items from cart"""
        # Validate operation
        is_valid, message = self.validate_cart_operation("remove")
        if not is_valid:
            return message
            
        result = self.remove_from_cart()
        
        # Debug cart state
        self.debug_cart_state()
        
        return result
        
    def handle_show_cart(self, query):
        """Handle showing cart contents"""
        # Debug cart state
        self.debug_cart_state()
        
        return self.get_cart_contents()
        
    def handle_checkout(self, query):
        """Handle checkout process"""
        # Validate operation
        is_valid, message = self.validate_cart_operation("checkout")
        if not is_valid:
            return message
            
        # Check if cart is empty
        if not self.cart:
            return "Your cart is empty. Please add some items to your cart before placing an order."
            
        # Start checkout process
        self.checkout_state = "address_selection"
        cart_info = self.get_cart_contents()
        cart_total = self.get_cart_total()
        
        response = f"Great! Let's complete your order. {cart_info}\n\n"
        response += f"Total amount: ₹{cart_total:,}\n\n"
        response += "Please select a delivery address. You can say 'use home address' or 'use work address'."
        
        return response
        
    def place_order(self, delivery_address, payment_method):
        """Place order and add to database"""
        if not self.current_user:
            return "Please login first to place an order."
            
        if not self.cart:
            return "Your cart is empty. Please add some items before placing an order."
            
        try:
            # Generate order ID
            order_id = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}"
            cart_id = f"CART{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Calculate total
            total_amount = self.get_cart_total()
            
            # Create order record
            order_data = {
                'order_id': order_id,
                'user_id': self.current_user['user_id'],
                'cart_id': cart_id,
                'total_amount': total_amount,
                'order_date': datetime.now().isoformat(),
                'delivery_address': delivery_address,
                'payment_method': payment_method,
                'order_status': 'Confirmed',
                'delivery_status': 'Processing',
                'delivery_date': (datetime.now() + timedelta(days=3)).isoformat()
            }
            
            # Add to orders CSV
            orders_df = pd.read_csv('user_orders.csv')
            new_order_df = pd.DataFrame([order_data])
            updated_orders_df = pd.concat([orders_df, new_order_df], ignore_index=True)
            updated_orders_df.to_csv('user_orders.csv', index=False)
            
            # Clear cart after successful order
            self.cart.clear()
            self.save_user_cart(self.current_user['user_id'], self.cart)
            
            # Generate confirmation message
            response = f"Order placed successfully!\n\n"
            response += f"Order ID: {order_id}\n"
            response += f"Total Amount: ₹{total_amount:,}\n"
            response += f"Delivery Address: {delivery_address}\n"
            response += f"Payment Method: {payment_method}\n"
            response += f"Expected Delivery: {(datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')}\n\n"
            response += "Your cart has been cleared. Thank you for shopping with Walmart!"
            
            return response
            
        except Exception as e:
            return f"Sorry, there was an error placing your order. Error: {str(e)}"

    def handle_order_status(self, query):
        """Handle order status and tracking queries"""
        if not self.current_user:
            return "Please login first to check your order status."
            
        # Load orders from CSV
        try:
            orders_df = pd.read_csv('user_orders.csv')
            user_orders = orders_df[orders_df['user_id'] == self.current_user['user_id']]
            
            if user_orders.empty:
                return f"Hello {self.current_user['name']}, you don't have any orders yet. Would you like to browse our products?"
                
            # Get the most recent order
            latest_order = user_orders.iloc[-1]
            
            status_response = f"Hello {self.current_user['name']}, here's your order status:\n"
            status_response += f"Order ID: {latest_order['order_id']}\n"
            status_response += f"Total Amount: ₹{latest_order['total_amount']:,}\n"
            status_response += f"Order Status: {latest_order['order_status']}\n"
            status_response += f"Delivery Status: {latest_order['delivery_status']}\n"
            
            if latest_order['delivery_status'] == 'In Transit':
                status_response += f"Expected Delivery: {latest_order['delivery_date']}\n"
            elif latest_order['delivery_status'] == 'Delivered':
                status_response += f"Delivered on: {latest_order['delivery_date']}\n"
                
            status_response += "\nWould you like to place a new order or check other products?"
            
            return status_response
            
        except Exception as e:
            return f"Sorry, I couldn't retrieve your order status. Error: {str(e)}"
            
    def handle_refund_replacement_status(self, query):
        """Handle refund and replacement status queries"""
        if not self.current_user:
            return "Please login first to check your refund or replacement status."
            
        # Mock refund/replacement data
        refund_status = {
            "U001": {
                "has_refund": True,
                "refund_id": "REF001",
                "amount": "₹2,999",
                "status": "Processed",
                "processed_date": "2024-01-20"
            },
            "U002": {
                "has_refund": False,
                "message": "No refund requests found"
            }
        }
        
        user_id = self.current_user['user_id']
        user_refund = refund_status.get(user_id, {"has_refund": False, "message": "No refund requests found"})
        
        if user_refund.get("has_refund", False):
            response = f"Hello {self.current_user['name']}, here's your refund status:\n"
            response += f"Refund ID: {user_refund['refund_id']}\n"
            response += f"Amount: {user_refund['amount']}\n"
            response += f"Status: {user_refund['status']}\n"
            response += f"Processed: {user_refund['processed_date']}\n"
            response += "\nIs there anything else I can help you with?"
        else:
            response = f"Hello {self.current_user['name']}, {user_refund['message']}. "
            response += "Would you like to place a new order or browse our products?"
            
        return response
        
    def handle_product_recommendations(self, query):
        """Handle product recommendations based on user preferences"""
        if not self.current_user:
            return "Please login first to get personalized recommendations."
            
        # Load inventory
        try:
            inventory_df = pd.read_csv('walmart_inventory.csv')
            
            # Simple recommendation logic based on query keywords
            recommendations = []
            
            if any(keyword in query.lower() for keyword in ["electronics", "phone", "headphone", "laptop"]):
                electronics = inventory_df[inventory_df['category'].str.contains('Electronics', case=False, na=False)]
                recommendations = electronics.head(3)
            elif any(keyword in query.lower() for keyword in ["clothing", "shirt", "pants", "dress", "fashion"]):
                clothing = inventory_df[inventory_df['category'].str.contains('Clothing', case=False, na=False)]
                recommendations = clothing.head(3)
            elif any(keyword in query.lower() for keyword in ["home", "kitchen", "furniture"]):
                home = inventory_df[inventory_df['category'].str.contains('Home', case=False, na=False)]
                recommendations = home.head(3)
            else:
                # General recommendations - top products by price (assuming higher price = better quality)
                recommendations = inventory_df.nlargest(5, 'price_inr')
                
            if recommendations.empty:
                return f"Hello {self.current_user['name']}, I couldn't find specific recommendations based on your query. "
                return "Let me show you some of our best products instead."
                
            response = f"Hello {self.current_user['name']}, here are my recommendations for you:\n\n"
            
            for idx, product in recommendations.iterrows():
                response += f"• {product['product_name']} - ₹{product['price_inr']:,}\n"
                response += f"  Category: {product['category']}\n"
                response += f"  Stock: {product['stock_quantity']} available\n\n"
                
            response += "Would you like me to add any of these items to your cart? "
            response += "Just say 'Add [product name] to cart' and I'll help you with that. "
            response += "Or would you like more recommendations?"
            
            return response
            
        except Exception as e:
            return f"Sorry, I couldn't load product recommendations. Error: {str(e)}"
        
    def debug_cart_state(self):
        """Debug function to show current cart state"""
        print(f"\n=== CART DEBUG ===")
        print(f"Cart items: {len(self.cart)}")
        print(f"Cart total: ₹{self.get_cart_total():,}")
        if self.cart:
            for i, item in enumerate(self.cart):
                print(f"  {i+1}. {item['name']} - ₹{item['price']:,} x{item['quantity']} (ID: {item['id']})")
        else:
            print("  Cart is empty")
        print(f"User: {self.current_user['name'] if self.current_user else 'Not logged in'}")
        print(f"==================\n")
        
    def validate_cart_operation(self, operation):
        """Validate cart operations"""
        if not self.current_user:
            return False, "User not logged in"
            
        if operation in ["remove", "checkout"] and not self.cart:
            return False, "Cart is empty"
            
        return True, "OK"

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
        
        if not isinstance(query, str) or not query.strip():
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
        
        if not isinstance(text, str) or not text.strip():
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
            
        # Save uploaded audio temporarily
        audio_path = "web_input.wav"
        audio_file.save(audio_path)
        
        # Read audio file and convert to array
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(frames, dtype=np.int16)
        
        # Transcribe audio
        query = assistant.transcribe(audio_array)
        
        # Clean up temporary file
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        
        if not isinstance(query, str) or not query.strip():
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
    """Command line voice interface with enhanced workflow"""
    print("\nWalmart Voice Assistant")
    print("=" * 50)
    
    # Generic greeting and user authentication
    assistant.speak("Hello! Welcome to Walmart Voice Assistant. I'm here to help you with your shopping needs.", async_mode=False)
    assistant.speak("Please provide your user ID to get started. You can say your user ID or type it.", async_mode=False)
    print("Please provide your user ID to login (e.g., U001, U002)")
    
    # User authentication loop
    while True:
        print("\n" + "="*50)
        try:
            mode = input("Enter your user ID (or type 'q' to quit): ")
            if not isinstance(mode, str) or mode.strip().lower() == "q":
                print("Thank you for shopping with Walmart!")
                break
                
            # Try to authenticate user
            if assistant.authenticate_user(mode.strip().upper()):
                if assistant.current_user:
                    assistant.speak(f"Welcome {assistant.current_user['name']}! You are now logged in.", async_mode=False)
                    assistant.speak("How can I help you today? You can ask me about:", async_mode=False)
                    assistant.speak("Product recommendations, order status, placing orders, or any shopping questions.", async_mode=False)
                    print(f"Logged in as: {assistant.current_user['name']}")
                break
            else:
                assistant.speak("Invalid user ID. Please try again.", async_mode=False)
                print("Invalid user ID. Available IDs: U001, U002")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
    
    print("\nExample queries:")
    print("- 'What do you recommend?'")
    print("- 'Where is my order?'")
    print("- 'Check refund status'")
    print("- 'Find me headphones'")
    print("- 'Add Sony headphones to cart'")
    print("- 'Show my cart'")
    print("- 'Place order' to checkout")
    print("- 'Stop speaking' to interrupt current speech")
    print("- Press Ctrl+C to quit anytime")
    
    # Main interaction loop
    while True:
        try:
            print("\n" + "="*50)
            user_input = input("Press Enter to talk (or type 'q' to quit): ")
            
            if not isinstance(user_input, str) or user_input.strip().lower() == "q":
                print("Thank you for shopping with Walmart!")
                break
                
            # Only record if user pressed Enter (empty input)
            if user_input.strip() == "":
                print("Starting voice recording...")
                print("Speak now, then press Enter to stop recording")
                
                try:
                    # Record and transcribe with manual stop
                    audio_array = assistant.record_audio()
                    query = assistant.transcribe(audio_array)
                    
                    if not isinstance(query, str) or not query.strip():
                        print("No speech detected. Please try again.")
                        continue
                        
                    print(f"You said: {query}")
                    
                    # Check for stop command
                    if isinstance(query, str) and "stop speaking" in query.lower():
                        assistant.stop_speaking()
                        print("Speech stopped")
                        continue
                    
                    # Process query
                    response = assistant.process_query(query)
                    print(f"Assistant: {response}")
                    
                    # Speak response
                    assistant.speak(response, async_mode=False)
                    
                except KeyboardInterrupt:
                    print("\nRecording stopped by user.")
                    assistant.stop_speaking()
                    continue
                    
            else:
                # Handle text input for testing
                query = user_input.strip()
                print(f"You typed: {query}")
                
                if query.lower() == "q":
                    print("Thank you for shopping with Walmart!")
                    break
                    
                # Process query
                response = assistant.process_query(query)
                print(f"Assistant: {response}")
                
                # Speak response
                assistant.speak(response, async_mode=False)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

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