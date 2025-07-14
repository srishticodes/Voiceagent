#!/usr/bin/env python3
"""
Streamlit version of Waltz Voice Assistant - Cloud Compatible
"""

import os, json, tempfile, uuid, re, io
from datetime import datetime, timedelta
from typing import cast

# Third-party
import streamlit as st  # type: ignore
import numpy as np  # type: ignore
from audiorecorder import audiorecorder  # type: ignore
from gtts import gTTS  # type: ignore
import whisper  # type: ignore
from scipy.signal import resample  # type: ignore

# Helper imports for audio conversion
import wave


# -----------------------------------------------------------------------------
# StreamlitWaltzAssistant: reuse core logic from walmart_assistant.py but expose
# a few convenience wrappers expected by the UI (transcribe_audio, synthesize_ 
# speech, simple_checkout). This keeps full conversational behaviour while
# satisfying type-checker/linter expectations.
# -----------------------------------------------------------------------------


class StreamlitWaltzAssistant(WalmartAssistant):
    """Thin wrapper around WalmartAssistant for Streamlit-specific helpers."""

    def __init__(self):
        super().__init__()

    # ---- Speech-to-Text helper ------------------------------------------------
    def transcribe_audio(self, audio_input, input_sample_rate: int = 44100) -> str:  # noqa: D401
        """Convert raw NumPy or bytes audio to text via Whisper."""
        try:
            if audio_input is None or len(audio_input) == 0 or self.stt_model is None:
                return ""

            import numpy as np  # local import to keep mypy happy

            # Accept NumPy arrays directly from audiorecorder
            if isinstance(audio_input, np.ndarray):
                audio_array = audio_input.astype(np.int16)
            else:
                with wave.open(io.BytesIO(audio_input), "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)

            # Mono
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            # Resample to 16 kHz for Whisper
            target_sr = 16000
            if input_sample_rate != target_sr:
                audio_array = resample(audio_array, int(len(audio_array) * target_sr / input_sample_rate))

            audio_float = audio_array.astype(np.float32) / 32768.0
            result = self.stt_model.transcribe(audio_float.flatten(), language="en", task="transcribe", fp16=False)
            return str(result.get("text", "")).strip()
        except Exception as exc:  # pragma: no cover
            st.error(f"Transcription error: {exc}")
            return ""

    # ---- Text-to-Speech helper ----------------------------------------------
    def synthesize_speech(self, text: str):  # noqa: D401
        """Return MP3 bytes using gTTS (fallback when Google Cloud unavailable)."""
        try:
            if not text:
                return None
            tts = gTTS(text=text, lang="en")
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as exc:  # pragma: no cover
            st.error(f"TTS error: {exc}")
            return None

    # ---- Simple checkout helper (non-interactive) ---------------------------
    def simple_checkout(self):  # noqa: D401
        """Place the order immediately with default data for demo."""
        return self.handle_checkout("checkout")


# Import the main assistant logic
from walmart_assistant import WalmartAssistant

# Page configuration
st.set_page_config(
    page_title="Waltz AI Shopping Assistant",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitWaltzAssistant(WalmartAssistant):
    """Thin subclass to expose WalmartAssistant inside Streamlit without modification."""
    def __init__(self):
        super().__init__()

# Reuse our audio helper functions below but rely on StreamlitWaltzAssistant for business logic.

def main():
    st.title("Waltz AI Shopping Assistant")
    st.markdown("### AI-powered shopping assistant with product search and cart management")
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = StreamlitWaltzAssistant()
    
    assistant = st.session_state.assistant

    # Global greeting once immediately after login
    if assistant.current_user and st.session_state.get('greet_needed', False):
        greeting_text = "Hello! I'm your Waltz Voice Assistant. How can I help you today?"
        greeting_audio = assistant.synthesize_speech(greeting_text)
        import base64
        if greeting_audio:
            b64 = base64.b64encode(greeting_audio).decode()
            st.markdown(f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
        # Reset flag
        st.session_state.greet_needed = False
    
    # User authentication
    if not assistant.current_user:
        st.sidebar.header("Login")
        user_id = st.sidebar.selectbox("Select User ID", ["", "U001", "U002"])
        
        if user_id and st.sidebar.button("Login"):
            if assistant.authenticate_user(user_id):
                st.sidebar.success(f"Welcome {assistant.current_user['name']}!")
                # Set flag to trigger greeting on rerun
                st.session_state.greet_needed = True
                st.rerun()
            else:
                st.sidebar.error("Login failed")
        
        if not user_id:
            st.info("Please login to use the shopping assistant")
            return
    else:
        st.sidebar.success(f"Logged in as: {assistant.current_user['name']}")
        if st.sidebar.button("Logout"):
            assistant.current_user = None
            assistant.cart = []
            st.rerun()
    
    # Main interface tabs
    tab4, tab1, tab2, tab3 = st.tabs(["Voice", "Search", "Cart", "Orders"])
    
    with tab1:
        st.header("Product Search")
        
        search_query = st.text_input("Search for products:", placeholder="e.g., headphones, laptops, shoes")
        
        if st.button("Search", type="primary") and search_query.strip():
            with st.spinner("Searching products..."):
                products = assistant.search_products(search_query)
                
                if products:
                    st.success(f"Found {len(products)} products")
                    
                    for i, doc in enumerate(products):
                        meta = doc.metadata  # type: ignore[attr-defined]
                        with st.expander(f"{meta.get('product_name')} - ₹{int(meta.get('price', 0)):,}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Category:** {meta.get('category')}")
                                st.write(f"**Description:** {meta.get('description')}")
                                st.write(f"**Stock:** {meta.get('stock_quantity', 0)} units")
                                st.write(f"**Match Score:** {doc.score:.2f}")
                            
                            with col2:
                                if st.button(f"Add to Cart", key=f"add_{i}"):
                                    result = assistant.add_to_cart(
                                        meta.get('product_name'),
                                        meta.get('price'),
                                        meta.get('product_id')
                                    )
                                    st.success(result)
                                    st.rerun()
                else:
                    st.warning("No products found. Try a different search term.")
    
    with tab2:
        st.header("Shopping Cart")
        
        if assistant.cart:
            total = 0
            
            for i, item in enumerate(assistant.cart):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{item['name']}**")
                    with col2:
                        st.write(f"₹{item['price']:,}")
                    with col3:
                        st.write(f"Qty: {item['quantity']}")
                    with col4:
                        if st.button("Remove", key=f"remove_{i}"):
                            assistant.cart.pop(i)
                            assistant.save_user_cart(assistant.current_user['user_id'], assistant.cart)
                            st.rerun()
                
                total += item['price'] * item['quantity']
                st.divider()
            
            st.markdown(f"### **Total: ₹{total:,}**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Checkout", type="primary"):
                    response = assistant.simple_checkout()
                    st.success(response)
            
            with col2:
                if st.button("Clear Cart"):
                    assistant.clear_cart()
                    st.success("Cart cleared!")
                    st.rerun()
        else:
            st.info("Your cart is empty. Add some products to get started!")
    
    with tab3:
        st.header("Order Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show All Orders"):
                orders = assistant.load_user_orders(assistant.current_user['user_id'])
                st.text_area("Order History", orders, height=300)
        
        with col2:
            if st.button("Latest Order Status"):
                status = assistant.handle_order_status("latest order")
                st.text_area("Order Status", status, height=300)
    
    with tab4:
        st.header("Voice Assistant")

        # Initialize conversation history in session_state
        if 'history' not in st.session_state:
            st.session_state.history = []

        # Display existing conversation
        for msg in st.session_state.history:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['text'])
                    if msg.get('audio') is not None:
                        st.audio(msg['audio'], sample_rate=44100 if isinstance(msg['audio'], __import__('numpy').ndarray) else None)
            else:
                with st.chat_message("assistant"):
                    st.write(msg['text'])
                    if msg.get('audio') is not None:
                        st.audio(msg['audio'], format="audio/mp3")

        # One-time greeting
        if 'greeted' not in st.session_state:
            greeting_text = "Hello! I'm your Waltz Voice Assistant. How can I help you today?"
            greeting_audio = assistant.synthesize_speech(greeting_text)
            st.session_state.greeted = True
            import base64
            if greeting_audio:
                b64 = base64.b64encode(greeting_audio).decode()
                audio_html = f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}"></audio>'
                st.markdown(audio_html, unsafe_allow_html=True)
            st.session_state.history.append({
                'role': 'assistant',
                'text': greeting_text,
                'audio': greeting_audio,
            })

        st.divider()
        st.write("Press **Start Recording** and speak your request, then press **Stop Recording**.")

        audio_capture = audiorecorder("Start Recording", "Stop Recording")
        if audio_capture is not None and len(audio_capture) > 0:
            import numpy as np
            # Immediate playback of user audio
            st.audio(audio_capture, sample_rate=44100)

            # Transcribe
            with st.spinner("Transcribing your speech..."):
                user_query = assistant.transcribe_audio(audio_capture)

            if user_query:
                # Append user message to history
                st.session_state.history.append({
                    'role': 'user',
                    'text': user_query,
                    'audio': audio_capture,
                })

                # Generate assistant response
                with st.spinner("Generating response..."):
                    response_text = assistant.process_query(user_query)

                # Synthesize speech
                with st.spinner("Creating voice response..."):
                    tts_audio = assistant.synthesize_speech(response_text)

                # Append assistant message
                st.session_state.history.append({
                    'role': 'assistant',
                    'text': response_text,
                    'audio': tts_audio,
                })

                # Rerun to display updated history immediately
                st.rerun()
    
    # Sidebar information
    st.sidebar.header("Quick Stats")
    if assistant.current_user:
        st.sidebar.metric("Cart Items", len(assistant.cart))
        st.sidebar.metric("Cart Total", f"₹{assistant.get_cart_total():,}")
    
    st.sidebar.header("How to Use")
    st.sidebar.markdown("""
    1. **Login** with U001 or U002
    2. **Search** for products
    3. **Add items** to your cart
    4. **Chat** with the assistant
    5. **Checkout** when ready
    6. **Track orders** in Orders tab
    """)

if __name__ == "__main__":
    main() 