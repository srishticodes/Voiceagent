import os
import io
import uuid
import numpy as np
import wave
import base64
import pandas as pd
from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
from flask_session import Session
from walmart_assistant import WalmartAssistant

app = Flask(__name__)
app.secret_key = "waltz_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
CORS(app)

def get_assistant():
    """Get the assistant instance for the current session."""
    if 'assistant' not in session:
        session['assistant'] = WalmartAssistant()
    return session['assistant']

@app.route("/")
def index():
    return render_template("waltz.html")

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    user_id = data.get("user_id")
    assistant = get_assistant()
    if assistant.authenticate_user(user_id):
        session["user_id"] = user_id
        session['assistant'] = assistant # Save back to session
        return jsonify({"success": True, "name": assistant.current_user["name"]})
    else:
        session.pop('assistant', None)
        session.pop('user_id', None)
        return jsonify({"success": False, "error": "Invalid user ID"})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/chat", methods=["POST"])
def chat():
    assistant = get_assistant()
    if not assistant.current_user:
        return jsonify({"error": "User not logged in"}), 401

    data = request.json
    text = data.get("message", "")
    if not text.strip():
        return jsonify({"error": "No message provided"}), 400

    response = assistant.process_query(text)
    session['assistant'] = assistant # Save state changes

    # Synthesize TTS
    audio_array = assistant.voice_manager.text_to_speech(response)
    audio_b64 = None
    if audio_array is not None:
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(24000) # TTS sample rate
            wf.writeframes(audio_array.tobytes())
        wav_bytes.seek(0)
        audio_b64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes.getvalue()).decode()

    return jsonify({"response": response, "audio": audio_b64})

@app.route("/api/voice", methods=["POST"])
def voice():
    assistant = get_assistant()
    if not assistant.current_user:
        return jsonify({"error": "User not logged in"}), 401

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    # Convert to numpy array
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    
    # Transcribe
    text = assistant.transcribe(audio_array)
    if not text.strip():
        return jsonify({"error": "No speech detected"}), 400
    
    # Process
    response = assistant.process_query(text)
    session['assistant'] = assistant # Save state changes

    # Synthesize TTS
    audio_array = assistant.voice_manager.text_to_speech(response)
    audio_b64 = None
    if audio_array is not None:
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_array.tobytes())
        wav_bytes.seek(0)
        audio_b64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes.getvalue()).decode()

    return jsonify({"query": text, "response": response, "audio": audio_b64})

@app.route("/api/cart", methods=["GET"])
def get_cart():
    assistant = get_assistant()
    cart = assistant.cart if assistant.current_user else []
    total = assistant.get_cart_total() if assistant.current_user else 0
    return jsonify({"cart": cart, "total": total, "count": len(cart)})

@app.route("/api/orders", methods=["GET"])
def get_orders():
    assistant = get_assistant()
    if not assistant.current_user:
        return jsonify({"orders": []})

    # Get structured order data instead of a formatted string
    try:
        if not os.path.exists('user_orders.csv'):
            return jsonify({"orders": []})
        
        orders_df = pd.read_csv('user_orders.csv')
        user_orders = orders_df[orders_df['user_id'] == assistant.current_user['user_id']]
        
        if user_orders.empty:
            return jsonify({"orders": []})
            
        # Group by order_id and format
        orders_list = []
        for order_id, group in user_orders.groupby('order_id'):
            total = (group['price_inr'] * group['quantity']).sum()
            orders_list.append({
                "order_id": order_id,
                "date": group.iloc[0]['order_date'][:10],
                "status": group.iloc[0]['delivery_status'],
                "total": total,
                "items": group[['product_name', 'quantity', 'price_inr']].to_dict('records')
            })
        
        # Sort by date descending
        orders_list.sort(key=lambda x: x['date'], reverse=True)
        return jsonify({"orders": orders_list})

    except Exception:
        return jsonify({"orders": []})

if __name__ == "__main__":
    app.run(debug=True, port=5001) 