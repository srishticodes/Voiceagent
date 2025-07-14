#!/usr/bin/env python3
"""
Waltz Voice Assistant - Core functionality for AI-powered shopping assistant
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
import re
from typing import cast

# Constants
RECORD_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Mock user data is now replaced by CSV files
# users.csv, user_addresses.csv, user_payment_methods.csv

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
        self.checkout_data: dict = {}
        
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
                self.retriever = None
                return
                
            # Read CSV with proper quoting to handle commas in descriptions
            df = pd.read_csv(inventory_file, quoting=1)  # QUOTE_ALL
            print(f"Loaded {len(df)} products from inventory")
            
            # Filter only in-stock items for search
            in_stock_df = df[df['stock_status'] == 'In Stock'].copy()
            print(f"Found {len(in_stock_df)} in-stock products")
            
        except FileNotFoundError:
            print("Inventory CSV file not found. Please ensure 'walmart_inventory.csv' exists.")
            self.retriever = None
            return
        except Exception as e:
            print(f"Error loading inventory: {e}")
            self.retriever = None
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
        print("Retriever initialized successfully")
        
    def load_user_cart(self, user_id):
        """Load user's cart from CSV"""
        try:
            if not os.path.exists("user_carts.csv"):
                return []
                
            df = pd.read_csv("user_carts.csv")
            # Check if 'status' column exists, if not create empty cart
            if 'status' not in df.columns:
                df['status'] = 'active'
                
            user_cart_df = df[(df['user_id'] == user_id) & (df['status'] == 'active')]
            
            cart_items = []
            for _, row in user_cart_df.iterrows():
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
            cart_file = "user_carts.csv"
            
            # Define columns to ensure consistency
            cart_columns = [
                'cart_id', 'user_id', 'product_id', 'product_name', 
                'price_inr', 'quantity', 'added_at', 'status'
            ]

            # Load existing carts or create a new DataFrame
            if os.path.exists(cart_file):
                df = pd.read_csv(cart_file)
            else:
                df = pd.DataFrame(columns=pd.Index(cart_columns))

            # Ensure all columns are present
            for col in cart_columns:
                if col not in df.columns:
                    df[col] = None

            # Remove old active items for this user
            df = df[~((df['user_id'] == user_id) & (df['status'] == 'active'))]
            
            # Add new cart items from the current session
            new_rows = []
            for item in cart_items:
                new_row = {
                    'cart_id': item.get('id', f"CART-{uuid.uuid4().hex[:8]}"),
                    'user_id': user_id,
                    'product_id': item['product_id'],
                    'product_name': item['name'],
                    'price_inr': item['price'],
                    'quantity': item['quantity'],
                    'added_at': item.get('added_at', datetime.now().isoformat()),
                    'status': 'active'
                }
                new_rows.append(new_row)

            if new_rows:
                new_rows_df = pd.DataFrame(new_rows)
                df = pd.concat([df, new_rows_df], ignore_index=True)
            
            # Save the updated DataFrame
            df.to_csv(cart_file, index=False)
            
        except Exception as e:
            print(f"Error saving user cart: {e}")
            
    def setup_shopping_cart(self):
        """Initialize shopping cart"""
        self.cart = []
        self.user_session = {"customer_id": None}
        self.cart_history = []
        
    def authenticate_user(self, user_id):
        """Authenticate user and load their cart and data from CSV files."""
        try:
            users_df = pd.read_csv("users.csv")
            user_series = users_df[users_df['user_id'] == user_id]

            if user_series.empty:
                return False

            self.current_user = user_series.iloc[0].to_dict()
            
            # Load addresses
            addresses_df = pd.read_csv("user_addresses.csv")
            user_addresses = addresses_df[addresses_df['user_id'] == user_id]
            self.current_user['addresses'] = [row for _, row in user_addresses.iterrows()]
            print(f"[DEBUG] Loaded {len(self.current_user['addresses'])} addresses for user {user_id}")

            # Load payment methods
            payments_df = pd.read_csv("user_payment_methods.csv")
            user_payments = payments_df[payments_df['user_id'] == user_id]
            self.current_user['payment_methods'] = [row for _, row in user_payments.iterrows()]
            print(f"[DEBUG] Loaded {len(self.current_user['payment_methods'])} payment methods for user {user_id}")

            self.user_session["customer_id"] = user_id
            
            # Load user's existing cart
            self.cart = self.load_user_cart(user_id)
            if self.current_user:
                print(f"Loaded {len(self.cart)} items from user's cart for {self.current_user.get('name')}")
            
            return True
        except FileNotFoundError as e:
            print(f"Error: {e}. Make sure users.csv, user_addresses.csv, and user_payment_methods.csv exist.")
            return False
        except Exception as e:
            print(f"An error occurred during authentication: {e}")
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
                    
                    # Use Whisper's transcribe method with audio array, explicitly disabling FP16 on CPU
                    result = self.stt_model.transcribe(
                        audio_float,
                        language="en",
                        task="transcribe",
                        fp16=False
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
        if not self.retriever:
            return []
        try:
            results = self.retriever.invoke(query)
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
            
    def process_query(self, query):
        """Process user query and generate response"""
        print(f"\n[DEBUG] Processing query: '{query}'")
        if not self.llm:
            return "Sorry, the AI model is not available."

        # Prioritize checkout flow if it has started
        if self.checkout_state:
            print(f"[DEBUG] Checkout state is '{self.checkout_state}'. Routing to handle_checkout_interaction.")
            return self.handle_checkout_interaction(query)
            
        # Handle user authentication
        if "login" in query.lower() or "sign in" in query.lower():
            print("[DEBUG] Intent: handle_login")
            return self.handle_login(query)
            
        # Handle order status queries
        if any(keyword in query.lower() for keyword in ["order status", "where is my order", "track order", "order tracking", "when will my order be delivered", "delivery status", "when will it arrive"]):
            print("[DEBUG] Intent: handle_order_status")
            return self.handle_order_status(query)
            
        # Handle replacement/refund status
        if any(keyword in query.lower() for keyword in ["refund", "replacement", "return", "refund status", "replacement status"]):
            print("[DEBUG] Intent: handle_refund_replacement_status")
            return self.handle_refund_replacement_status(query)
            
        # Handle product recommendations
        if any(keyword in query.lower() for keyword in ["recommend", "recommendation", "suggest", "what should I buy", "what do you recommend"]):
            print("[DEBUG] Intent: handle_product_recommendations")
            return self.handle_product_recommendations(query)
            
        # Handle checkout process - broader keywords
        checkout_keywords = ["checkout", "place order", "place my order", "buy now", "please my order", "order from my cart"]
        if any(keyword in query.lower() for keyword in checkout_keywords):
            print("[DEBUG] Intent: handle_checkout")
            return self.handle_checkout(query)
            
        # Handle cart operations
        # Handle mis-transcriptions of "cart" as "card"
        if "add" in query.lower() and ("cart" in query.lower() or "card" in query.lower()):
            print("[DEBUG] Intent: handle_add_to_cart")
            return self.handle_add_to_cart(query)
            
        # Handle cart questions (why is X in my cart, what's in my cart, etc.)
        if any(keyword in query.lower() for keyword in ["why is", "what's in my cart", "what is in my cart", "cart has", "cart contains"]) and "cart" in query.lower():
            print("[DEBUG] Intent: handle_show_cart")
            return self.handle_show_cart(query)
            
        if "remove" in query.lower() and "cart" in query.lower():
            print("[DEBUG] Intent: handle_remove_from_cart")
            return self.handle_remove_from_cart(query)
            
        if "show cart" in query.lower() or "my cart" in query.lower():
            print("[DEBUG] Intent: handle_show_cart")
            return self.handle_show_cart(query)
            
        # Handle order history queries
        if any(keyword in query.lower() for keyword in ["show my orders", "order history", "past orders", "my orders", "list orders"]):
            print("[DEBUG] Intent: list_user_orders")
            return self.list_user_orders()

        # Search for products
        print("[DEBUG] Intent: General product search")
        products = self.search_products(query)
        
        # Format available products (IN STOCK)
        products_text = ""
        if products and len(products) > 0:
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
You are Waltz, a helpful shopping assistant with a friendly, conversational voice. 
Help customers find products, manage their cart, and track orders. Speak naturally and clearly.

CRITICAL RULES - NEVER BREAK THESE:
1.  **INVENTORY vs. CART**: The INVENTORY is all available products. The CART is ONLY what the user has personally selected. Do not confuse them.
2.  **BE FACTUAL**: NEVER invent or "hallucinate" products, prices, order details, or eligibility statuses. If you don't know or can't find the information, say so clearly (e.g., "I could not find an order with that ID.").
3.  **USE YOUR TOOLS**: Rely on the functions provided to get information. Do not make up answers.
4.  **CURRENCY**: ALL prices must be in Indian Rupees (INR) and formatted with the '₹' symbol (e.g., ₹1,299).
5.  **CART CONTENTS**: When asked about the cart, only list items from the "USER'S PERSONAL CART" section. Do not mention inventory items.
6.  **PRODUCT RECOMMENDATIONS**: Only recommend products listed in the "AVAILABLE INVENTORY" section.
7.  **NO MARKDOWN**: Do not use any formatting like asterisks (*), dashes (-), or other markdown. Write responses in plain, natural sentences.

Here are your primary capabilities:
- **Search for products**: Find items in the inventory.
- **Manage Cart**: Add or remove items from the user's personal cart.
- **Checkout**: Place an order from the user's cart.
- **Check Order Status**: Provide updates on past orders using an Order ID.
- **Check Return/Refund Eligibility**: Inform the user if an order is within the 15-day return window.

Current user: {user_info}
AVAILABLE INVENTORY (In Stock Products): {products}
USER'S PERSONAL CART: {cart}

User query: {query}

Provide a helpful, conversational response that sounds natural when spoken aloud.
- If a user asks a question about a product, search the inventory and state what you find. Do not ask to add it to the cart.
- ONLY mention cart items that actually exist in the user's personal cart above
- NEVER invent or hallucinate cart contents or order details.
- Keep responses concise but informative
- Use natural speech patterns and avoid any markdown formatting like asterisks.
- Your entire response should be a single paragraph of conversational text. Do not use lists.
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
            "user_id": self.current_user['user_id']
        }
        
        self.cart.append(item)
        self.cart_history.append({"action": "add", "item": item, "timestamp": datetime.now()})
        
        # Save to CSV
        self.save_user_cart(self.current_user['user_id'], self.cart)
        
        print(f"DEBUG: Added to cart - {item['name']} (₹{item['price']})")
        print(f"DEBUG: Cart now has {len(self.cart)} items")
        
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
                    if self.current_user:
                        self.save_user_cart(self.current_user['user_id'], self.cart)
                    
                    return f"Removed {removed_item['name']} from your cart. Your cart now has {len(self.cart)} items."
            return "Item not found in cart."
        else:
            # Remove last item
            removed_item = self.cart.pop()
            self.cart_history.append({"action": "remove", "item": removed_item, "timestamp": datetime.now()})
            
            # Save to CSV
            if self.current_user:
                self.save_user_cart(self.current_user['user_id'], self.cart)
            
            return f"Removed {removed_item['name']} from your cart. Your cart now has {len(self.cart)} items."
        
    def get_cart_contents(self):
        """Get formatted cart contents"""
        if not self.cart:
            return "Your cart is empty."
            
        cart_items = []
        total = 0
        
        for item in self.cart:
            cart_items.append(f"{item['name']} (₹{item['price']:,}) x{item['quantity']}")
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
        if self.current_user:
            self.save_user_cart(self.current_user['user_id'], self.cart)
        
        return "Cart cleared"
        
    def handle_add_to_cart(self, query):
        """Handles a direct user command to add an item to the cart."""
        # Validate operation
        is_valid, message = self.validate_cart_operation("add")
        if not is_valid:
            return message
            
        if not self.retriever:
            return "Product search is currently unavailable. Please try again later."
            
        # Clean up the query to get just the product name
        cleaned_query = re.sub(r'add|to|my|cart|card|please', '', query, flags=re.IGNORECASE).strip()
        
        if not cleaned_query:
            return "Of course. What product would you like to add?"

        products = self.search_products(cleaned_query)
        if not products:
            return f"I'm sorry, I couldn't find any products matching '{cleaned_query}'. Please try a different name."
            
        # Get the top search result
        product_doc = products[0]
        product_name = product_doc.metadata.get('product_name', 'Unknown Product')
        product_price = product_doc.metadata.get('price', 0)
        product_id = product_doc.metadata.get('product_id', '')
        
        # Check stock before adding
        try:
            inventory_df = pd.read_csv('walmart_inventory.csv', quoting=1)
            product_row = inventory_df[inventory_df['product_id'] == product_id]
            
            if product_row.empty or product_row.iloc[0]['stock_quantity'] <= 0:
                return f"I'm sorry, {product_name} is currently out of stock."
                
            # Directly add the item and confirm.
            self.add_to_cart(product_name, product_price, product_id)
            response = f"Great! I've added {product_name} to your cart for ₹{product_price:,}."
            response += " You can continue shopping or say 'place order' to checkout."
            return response
            
        except Exception as e:
            return f"Sorry, there was an error checking our inventory. Please try again in a moment. Error: {str(e)}"
        
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
        """Handle showing cart contents and answering cart-related questions"""
        # Debug cart state
        self.debug_cart_state()
        
        # Check if user is asking about specific items in cart
        query_lower = query.lower()
        if "why is" in query_lower and "cart" in query_lower:
            # User is asking why something is in their cart
            if not self.cart:
                return "Your cart is empty, so there's nothing to explain."
            
            # Extract potential product name from query
            words = query_lower.split()
            for i, word in enumerate(words):
                if word in ["converse", "headphones", "laptop", "phone", "shoes", "shirt", "pants"]:
                    # Check if this item is actually in cart
                    for item in self.cart:
                        if word.lower() in item['name'].lower():
                            return f"{item['name']} is in your cart because you added it earlier. You can remove it by saying 'remove from cart' if you don't want it."
            
            return "I can see items in your cart, but I'm not sure which specific item you're asking about. Here's what's in your cart: " + self.get_cart_contents()
        
        return self.get_cart_contents()
        
    def handle_checkout(self, query):
        """Handle checkout process"""
        print(f"[DEBUG] handle_checkout called with query: '{query}'")
        
        # Validate operation
        is_valid, message = self.validate_cart_operation("checkout")
        if not is_valid:
            print(f"[DEBUG] Cart validation failed: {message}")
            return message
            
        # Start checkout process
        self.checkout_state = "confirm_order"
        cart_info = self.get_cart_contents()
        cart_total = self.get_cart_total()
        
        print(f"[DEBUG] Starting checkout. State -> {self.checkout_state}. Cart total: ₹{cart_total:,.2f}")
        
        response = f"Perfect! Let me help you place your order. {cart_info}\n"
        response += f"The total amount is ₹{cart_total:,.2f}. "
        response += "Would you like to proceed with checkout? Please say 'yes' to continue or 'no' to cancel."
        
        print(f"[DEBUG] Checkout response: {response}")
        return response

    def handle_checkout_interaction(self, query):
        """Manages the step-by-step checkout process based on the current state."""
        state = self.checkout_state
        print(f"[DEBUG] Handling checkout interaction. Current state: {state}, Query: '{query}'")
        
        if state == "confirm_order":
            if 'yes' in query.lower() or 'proceed' in query.lower() or 'continue' in query.lower():
                self.checkout_state = "awaiting_address"
                print(f"[DEBUG] User confirmed order. Checkout state updated: {self.checkout_state}")
                return self.prompt_for_address()
            else:
                self.checkout_state = None
                self.checkout_data = {}
                print("[DEBUG] Checkout cancelled by user.")
                return "No problem! Your cart is still saved. You can checkout anytime by saying 'place order'."

        elif state == "awaiting_address":
            return self.handle_address_selection(query)

        elif state == "awaiting_payment":
            return self.handle_payment_selection(query)

        else:
            self.checkout_state = None
            self.checkout_data = {}
            print(f"[DEBUG] Unknown checkout state '{state}'. Resetting.")
            return "There was an issue with the checkout process. Let's start over. Say 'place order' to try again."

    def prompt_for_address(self):
        """Generates a prompt asking the user to select a delivery address."""
        print("[DEBUG] Prompting for address selection")
        response = "Great! Now I need a delivery address. "
        
        if not self.current_user:
            return "Error: User not logged in."
        
        addresses = self.current_user.get('addresses', [])
        print(f"[DEBUG] Found {len(addresses)} addresses for user")
        
        if not addresses:
            return "You don't have any saved addresses. Please add an address to continue with your order."

        response += "Please choose from your saved addresses: "
        for i, addr in enumerate(addresses):
            response += f"Say '{i+1}' or '{self.number_to_word(i+1)}' for {addr['type']} address at {addr['street']}, {addr['city']}. "
        
        response += "Or say 'cancel' to stop the order."
        print(f"[DEBUG] Address prompt: {response}")
        return response

    def handle_address_selection(self, query):
        """Processes user's address choice."""
        if not self.current_user:
            return "Error: User not logged in."
        addresses = self.current_user.get('addresses', [])
        print(f"[DEBUG] Handling address selection. Found {len(addresses)} addresses. Query: '{query}'")
        
        try:
            # Handle word numbers (one, two, three, etc.) and ordinal words
            word_to_number = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
                'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
            }
            
            query_lower = query.lower().strip()
            
            # First try to match word numbers or ordinal words
            choice = None
            for word, num in word_to_number.items():
                if word in query_lower:
                    choice = num
                    break
            
            # If no word/ordinal number found, try to find digits
            if choice is None:
                match = re.search(r'(\d+)', query)
                if match:
                    choice = int(match.group())
                else:
                    raise ValueError("No number found in query")
            
            choice = choice - 1  # Convert to 0-based index
            print(f"[DEBUG] User selected address choice: {choice + 1}")

            if addresses and 0 <= choice < len(addresses):
                self.checkout_data['address'] = addresses[choice]
                self.checkout_state = "awaiting_payment"
                print(f"[DEBUG] Address selected: {self.checkout_data['address']}. State -> {self.checkout_state}")
                return self.prompt_for_payment()
            else:
                return f"That's not a valid choice. Please select a number from 1 to {len(addresses)}."
                
        except (ValueError, AttributeError):
            if 'cancel' in query.lower():
                self.checkout_state = None
                self.checkout_data = {}
                print("[DEBUG] Checkout cancelled. State has been reset.")
                return "Order cancelled. Your cart is still saved for later."
            return "I didn't understand that. Please say the number of the address you'd like to use, like 'one' or 'two'."
    
    def prompt_for_payment(self):
        """Generates a prompt for payment method selection."""
        print("[DEBUG] Prompting for payment method selection")
        response = "Perfect! Address selected. Now, how would you like to pay? "
        
        if not self.current_user:
            return "Error: User not logged in."
        payments = self.current_user.get('payment_methods', [])
        print(f"[DEBUG] Found {len(payments)} payment methods for user")

        if not payments:
            return "You don't have any saved payment methods. Please add a payment method to continue."

        response += "Please choose your payment method: "
        for i, pay in enumerate(payments):
            if pay['type'] == 'Credit Card':
                response += f"Say '{i+1}' or '{self.number_to_word(i+1)}' for Credit Card ending in {pay['last4']}. "
            else:
                response += f"Say '{i+1}' or '{self.number_to_word(i+1)}' for {pay['type']}. "

        response += "Or say 'cancel' to stop the order."
        print(f"[DEBUG] Payment prompt: {response}")
        return response

    def handle_payment_selection(self, query):
        """Processes user's payment choice and places the order."""
        if not self.current_user:
            return "Error: User not logged in."
        payments = self.current_user.get('payment_methods', [])
        print(f"[DEBUG] Handling payment selection. Found {len(payments)} methods. Query: '{query}'")

        try:
            # Handle word numbers (one, two, three, etc.) and ordinal words
            word_to_number = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
                'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
            }
            
            query_lower = query.lower().strip()
            
            # First try to match word numbers or ordinal words
            choice = None
            for word, num in word_to_number.items():
                if word in query_lower:
                    choice = num
                    break
            
            # If no word/ordinal number found, try to find digits
            if choice is None:
                match = re.search(r'(\d+)', query)
                if match:
                    choice = int(match.group())
                else:
                    raise ValueError("No number found in query")
            
            choice = choice - 1  # Convert to 0-based index
            print(f"[DEBUG] User selected payment choice: {choice + 1}")
            
            if payments and 0 <= choice < len(payments):
                self.checkout_data['payment'] = payments[choice]
                print(f"[DEBUG] Payment selected: {self.checkout_data['payment']}. Placing order...")
                
                # All data collected, place the order
                address_str = f"{self.checkout_data['address']['street']}, {self.checkout_data['address']['city']}"
                payment_str = self.checkout_data['payment']['type']
                
                response = self.place_order(address_str, payment_str)
                
                # Clear checkout state
                self.checkout_state = None
                self.checkout_data = {}
                print("[DEBUG] Order placed successfully. Checkout state has been reset.")
                return response
            else:
                return f"That's not a valid choice. Please select a number from 1 to {len(payments)}."
                
        except (ValueError, AttributeError):
            if 'cancel' in query.lower():
                self.checkout_state = None
                self.checkout_data = {}
                print("[DEBUG] Checkout cancelled. State has been reset.")
                return "Order cancelled. Your cart is still saved for later."
            return "I didn't understand that. Please say the number for your payment choice, like 'one' or 'two'."

    def place_order(self, delivery_address, payment_method):
        """Place order, save each cart item to orders CSV, and clear the cart."""
        print(f"[DEBUG] place_order called with address: {delivery_address}, payment: {payment_method}")
        
        if not self.current_user:
            return "Please login first to place an order."
            
        if not self.cart:
            return "Your cart is empty. Please add some items before placing an order."
            
        try:
            order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
            order_date = datetime.now()
            
            print(f"[DEBUG] Generated order ID: {order_id}")
            print(f"[DEBUG] Processing {len(self.cart)} items in cart")
            
            # Define order columns
            order_columns = [
                'order_id', 'user_id', 'product_id', 'product_name', 'price_inr', 
                'quantity', 'order_date', 'delivery_address', 'payment_method', 
                'delivery_status', 'replacement_eligible_until', 'refund_eligible_until'
            ]

            # Load existing orders or create a new DataFrame
            if os.path.exists('user_orders.csv'):
                orders_df = pd.read_csv('user_orders.csv')
                print(f"[DEBUG] Loaded existing orders CSV with {len(orders_df)} records")
            else:
                orders_df = pd.DataFrame(columns=pd.Index(order_columns))
                print("[DEBUG] Created new orders DataFrame")

            new_orders = []
            for item in self.cart:
                order_item = {
                    'order_id': order_id,
                    'user_id': self.current_user['user_id'],
                    'product_id': item['product_id'],
                    'product_name': item['name'],
                    'price_inr': item['price'],
                    'quantity': item['quantity'],
                    'order_date': order_date.isoformat(),
                    'delivery_address': delivery_address,
                    'payment_method': payment_method,
                    'delivery_status': 'Order Placed - Processing',
                    'replacement_eligible_until': (order_date + timedelta(days=15)).isoformat(),
                    'refund_eligible_until': (order_date + timedelta(days=15)).isoformat()
                }
                new_orders.append(order_item)
                print(f"[DEBUG] Added order item: {item['name']} (₹{item['price']})")
            
            # Append new orders and save
            if new_orders:
                new_orders_df = pd.DataFrame(new_orders)
                updated_orders_df = pd.concat([orders_df, new_orders_df], ignore_index=True)
                updated_orders_df.to_csv('user_orders.csv', index=False)
                print(f"[DEBUG] Saved {len(new_orders)} new order items to user_orders.csv")

            total_amount = self.get_cart_total()
            
            # Clear cart after successful order
            self.cart.clear()
            self.save_user_cart(self.current_user['user_id'], self.cart)
            print("[DEBUG] Cart cleared after successful order")
            
            # Generate confirmation message
            response = f"Excellent! Your order has been placed successfully. "
            response += f"Order ID: {order_id}. "
            response += f"Total Amount: ₹{total_amount:,}. "
            response += f"Your order will be delivered to {delivery_address}. "
            response += "You can check your order status anytime by asking 'where is my order'."
            
            print(f"[DEBUG] Order placement complete. Response: {response}")
            return response
            
        except Exception as e:
            print(f"[DEBUG] Error placing order: {e}")
            return f"Sorry, there was an error placing your order. Please try again. Error: {str(e)}"

    def handle_order_status(self, query):
        """Handle order status and tracking queries with dynamic delivery estimates."""
        if not self.current_user:
            return "Please login first to check your order status."
            
        try:
            if not os.path.exists('user_orders.csv'):
                return f"Hello {self.current_user['name']}, you have no past orders."

            orders_df = pd.read_csv('user_orders.csv')
            user_orders = orders_df[orders_df['user_id'] == self.current_user['user_id']]

            if user_orders.empty:
                return f"Hello {self.current_user['name']}, you don't have any orders yet."

            # Try to find an order ID in the query
            import re
            match = re.search(r'ORD-[A-F0-9]{8}', query.upper())
            target_order: pd.DataFrame | None = None

            if match:
                order_id = match.group(0)
                order_details = cast(pd.DataFrame, user_orders[user_orders['order_id'] == order_id])
                if not order_details.empty:
                    target_order = order_details
                    response = f"Here is the status for order {order_id}:\n"
                else:
                    return f"I could not find an order with the ID {order_id} for your account."
            else:
                # If no ID is specified, get the latest order
                latest_order_id = cast(pd.Series, user_orders['order_id']).iloc[-1]
                target_order = cast(pd.DataFrame, user_orders[user_orders['order_id'] == latest_order_id])
                response = f"Here is the status for your latest order ({latest_order_id}):\n"

            if target_order is None or target_order.empty:
                return "Could not find the specified order."

            # Get order details and calculate dynamic delivery status
            order_date_str = cast(pd.Series, target_order['order_date']).iloc[0]
            order_date = datetime.fromisoformat(order_date_str)
            days_since_order = (datetime.now() - order_date).days
            
            # Calculate dynamic delivery status based on days since order
            if days_since_order == 0:
                delivery_status = "Order Placed - Processing"
                estimated_delivery = "2-3 business days"
            elif days_since_order == 1:
                delivery_status = "Order Confirmed - Shipped"
                estimated_delivery = "1-2 business days"
            elif days_since_order == 2:
                delivery_status = "In Transit"
                estimated_delivery = "Tomorrow or next business day"
            elif days_since_order >= 3:
                delivery_status = "Out for Delivery"
                estimated_delivery = "Today or tomorrow"
            else:
                delivery_status = "Order Placed"
                estimated_delivery = "3-5 business days"

            total_amount = (cast(pd.Series, target_order['price_inr']) * cast(pd.Series, target_order['quantity'])).sum()

            response += f"Status: {delivery_status}\n"
            response += f"Order Date: {order_date.strftime('%B %d, %Y')}\n"
            response += f"Estimated Delivery: {estimated_delivery}\n"
            response += f"Total: ₹{total_amount:,.2f}\n"
            response += "Products in this order:\n"
            for _, item in target_order.iterrows():
                response += f"- {item['product_name']} (x{item['quantity']})\n"

            return response
            
        except Exception as e:
            print(f"Error getting order status: {e}")
            return f"Sorry, I couldn't retrieve your order status. Error: {str(e)}"
            
    def handle_refund_replacement_status(self, query):
        """Handle refund and replacement status queries based on a 15-day policy."""
        if not self.current_user:
            return "Please login first to check your refund or replacement status."

        try:
            if not os.path.exists('user_orders.csv'):
                return f"Hello {self.current_user['name']}, you have no past orders to check."

            orders_df = pd.read_csv('user_orders.csv')
            user_orders = orders_df[orders_df['user_id'] == self.current_user['user_id']]

            if user_orders.empty:
                return f"Hello {self.current_user['name']}, you don't have any orders yet."

            # Try to find an order ID in the query
            import re
            match = re.search(r'ORD-[A-F0-9]{8}', query.upper())
            target_order_id = None

            if match:
                order_id = match.group(0)
                if order_id in cast(pd.Series, user_orders['order_id']).values:
                    target_order_id = order_id
                else:
                    return f"I could not find an order with the ID {order_id} for your account."
            else:
                # If no ID is specified, use the latest order
                target_order_id = cast(pd.Series, user_orders['order_id']).iloc[-1]

            order_details_row = cast(pd.DataFrame, user_orders[user_orders['order_id'] == target_order_id])
            if order_details_row.empty:
                return f"Could not find details for order {target_order_id}."

            order_details = order_details_row.iloc[0]
            
            eligibility_date_str = order_details['refund_eligible_until']
            eligibility_date = datetime.fromisoformat(eligibility_date_str)
            
            response = f"For order {target_order_id}:\n"
            if datetime.now() <= eligibility_date:
                days_left = (eligibility_date - datetime.now()).days
                response += f"You are eligible for a refund or replacement until {eligibility_date.strftime('%B %d, %Y')}. "
                response += f"You have {days_left} days left to request it."
            else:
                response += f"This order is no longer eligible for a refund or replacement. The window closed on {eligibility_date.strftime('%B %d, %Y')}."

            return response
            
        except Exception as e:
            print(f"Error getting refund/replacement status: {e}")
            return f"Sorry, I couldn't retrieve the status. Error: {str(e)}"
        
    def handle_product_recommendations(self, query):
        """Handle product recommendations using vector search for relevance."""
        if not self.current_user:
            return "Please login first to get personalized recommendations."
            
        if not self.retriever:
            return "I can't access product information right now to make recommendations."

        # Use vector search to find relevant products
        search_results = self.search_products(query)
        
        if not search_results:
            return f"I couldn't find any products matching '{query}' to recommend. You could try a broader search term."

        response = f"Based on your request, here are some recommendations for you: "
        
        recommended_products = []
        for doc in search_results[:3]: # recommend top 3
            product_name = doc.metadata.get('product_name', 'Unknown Product')
            product_price = doc.metadata.get('price', 0)
            recommended_products.append(f"{product_name} for ₹{product_price:,.0f}")
        
        if recommended_products:
            # Join into a natural sentence
            if len(recommended_products) > 1:
                response += ", ".join(recommended_products[:-1]) + f", and {recommended_products[-1]}."
            else:
                response += f"{recommended_products[0]}."
        
            response += " Would you like me to add any of these to your cart?"
        else:
            return f"I found some items related to '{query}' but couldn't formulate a recommendation. Please try searching directly."

        return response
        
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
        print(f"Checkout state: {self.checkout_state}")
        print(f"==================\n")
        
    def number_to_word(self, num):
        """Convert number to word for better voice interaction"""
        word_numbers = {
            1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
            6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'
        }
        return word_numbers.get(num, str(num))
        
    def validate_cart_operation(self, operation):
        """Validate cart operations"""
        if not self.current_user:
            return False, "Please login first to access your cart."
            
        if operation in ["remove", "checkout"] and not self.cart:
            return False, "Your cart is empty."
            
        return True, "OK"

    def list_user_orders(self):
        """Return a summary of all current and past orders for the logged-in user."""
        if not self.current_user:
            return "Please login first to view your orders."
        try:
            if not os.path.exists('user_orders.csv'):
                return f"Hello {self.current_user['name']}, you have no past orders."
            orders_df = pd.read_csv('user_orders.csv')
            user_orders = orders_df[orders_df['user_id'] == self.current_user['user_id']]
            if user_orders.empty:
                return f"Hello {self.current_user['name']}, you don't have any orders yet."
            # Sort by order_date descending
            user_orders = user_orders.copy()
            user_orders['order_date'] = user_orders['order_date'].astype(str)
            user_orders = user_orders.sort_values(by="order_date", ascending=False)  # type: ignore
            response = f"Here are your orders, most recent first:\n"
            for idx, (order_id, group) in enumerate(user_orders.groupby('order_id')):
                order_date = group.iloc[0]['order_date']
                delivery_status = group.iloc[0]['delivery_status']
                total = (group['price_inr'] * group['quantity']).sum()
                response += f"Order ID: {order_id} | Date: {order_date[:10]} | Status: {delivery_status} | Total: ₹{total:,.0f}\n"
                response += "  Products: "
                response += ", ".join([f"{row['product_name']} (x{row['quantity']})" for _, row in group.iterrows()])
                response += "\n"
            return response.strip()
        except Exception as e:
            print(f"Error listing user orders: {e}")
            return "Sorry, I couldn't retrieve your order history. Please try again."

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
    print("\nWaltz Voice Assistant")
    print("=" * 50)
    
    # Generic greeting and user authentication
    assistant.speak("Hello! Welcome to Waltz Voice Assistant. I'm here to help you with your shopping needs.", async_mode=False)
    assistant.speak("To get started, please type your user ID.", async_mode=False)
    print("Please provide your user ID to login (e.g., U001, U002)")
    
    # User authentication loop
    while True:
        print("\n" + "="*50)
        try:
            mode = input("Enter your user ID (or type 'q' to quit): ")
            if not isinstance(mode, str) or mode.strip().lower() == "q":
                print("Thank you for shopping with Waltz!")
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
            assistant.speak("Thank you for shopping with Waltz! Have a great day.", async_mode=False)
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
                assistant.speak("Thank you for shopping with Waltz! Have a great day.", async_mode=False)
                print("Thank you for shopping with Waltz!")
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
                    print("Thank you for shopping with Waltz!")
                    break
                    
                # Process query
                response = assistant.process_query(query)
                print(f"Assistant: {response}")
                
                # Speak response
                assistant.speak(response, async_mode=False)
                
        except KeyboardInterrupt:
            assistant.speak("Thank you for shopping with Waltz! Have a great day.", async_mode=False)
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