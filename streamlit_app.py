#!/usr/bin/env python3
"""
Streamlit version of Walmart Voice Assistant
"""

import streamlit as st
import pandas as pd
import os
from walmart_assistant import WalmartAssistant

# Initialize the assistant
@st.cache_resource
def get_assistant():
    return WalmartAssistant()

def main():
    st.title("Walmart AI Shopping Assistant")
    st.write("AI-powered shopping assistant with product search and cart management")
    
    # Initialize assistant
    assistant = get_assistant()
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    voice_enabled = st.sidebar.checkbox("Enable Voice Output", value=True)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Chat", "Product Search", "Shopping Cart"])
    
    with tab1:
        st.header("Chat with Assistant")
        
        # Chat input
        user_input = st.text_input("Ask me anything about products:", key="chat_input")
        
        if st.button("Send", key="send_button"):
            if user_input.strip():
                with st.spinner("Processing..."):
                    response = assistant.process_query(user_input)
                    st.write("**Assistant:**", response)
                    
                    if voice_enabled:
                        assistant.speak(response, async_mode=True)
    
    with tab2:
        st.header("Product Search")
        
        # Search input
        search_query = st.text_input("Search for products:", key="search_input")
        
        if st.button("Search", key="search_button"):
            if search_query.strip():
                with st.spinner("Searching..."):
                    products = assistant.search_products(search_query)
                    
                    if products:
                        st.write(f"Found {len(products)} products:")
                        for i, product in enumerate(products):
                            with st.expander(f"Product {i+1}: {product.metadata.get('product_name', 'Unknown')}"):
                                st.write(product.page_content)
                    else:
                        st.write("No products found.")
    
    with tab3:
        st.header("Shopping Cart")
        
        # Display cart
        if assistant.cart:
            st.write("**Current Cart:**")
            total = 0
            for i, item in enumerate(assistant.cart):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"{item['name']}")
                with col2:
                    st.write(f"₹{item['price']}")
                with col3:
                    if st.button("Remove", key=f"remove_{i}"):
                        assistant.cart.pop(i)
                        st.rerun()
                total += item['price']
            
            st.write(f"**Total: ₹{total}**")
            
            if st.button("Clear Cart"):
                assistant.clear_cart()
                st.rerun()
        else:
            st.write("Your cart is empty.")
    
    # Voice input section
    st.sidebar.header("Voice Input")
    if st.sidebar.button("Record Voice"):
        st.sidebar.write("Recording... (5 seconds)")
        # Note: Streamlit doesn't support direct microphone access
        # This would need to be implemented with additional libraries
        st.sidebar.write("Voice recording not available in Streamlit demo")

if __name__ == "__main__":
    main() 