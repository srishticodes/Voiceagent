#!/usr/bin/env python3
"""
Streamlit version of Walmart Voice Assistant
"""

import streamlit as st
import pandas as pd
import os
import json
import tempfile
from walmart_assistant import WalmartAssistant

# Page configuration
st.set_page_config(
    page_title="Walmart AI Shopping Assistant",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the assistant
@st.cache_resource
def get_assistant():
    # Handle Google Cloud credentials for Streamlit hosting
    try:
        if "google_cloud" in st.secrets:
            # Create temporary credentials file from secrets
            creds = dict(st.secrets["google_cloud"])
            temp_creds = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(creds, temp_creds)
            temp_creds.close()
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds.name
            st.success("Google Cloud TTS configured from secrets")
    except Exception as e:
        st.warning(f"Google Cloud TTS not configured: {e}")
    
    return WalmartAssistant()

def main():
    st.title("Walmart AI Shopping Assistant")
    st.markdown("### AI-powered shopping assistant with product search and cart management")
    
    # Initialize assistant
    assistant = get_assistant()
    
    # User authentication
    if not assistant.current_user:
        st.sidebar.header("Login")
        user_id = st.sidebar.selectbox("Select User ID", ["", "U001", "U002"])
        
        if user_id and st.sidebar.button("Login"):
            if assistant.authenticate_user(user_id):
                st.sidebar.success(f"Welcome {assistant.current_user['name']}!")
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
    tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Search", "Cart", "Orders"])
    
    with tab1:
        st.header("Chat with Assistant")
        
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about products..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = assistant.process_query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with tab2:
        st.header("Product Search")
        
        search_query = st.text_input("Search for products:", placeholder="e.g., headphones, laptops, shoes")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("Search", type="primary")
        
        if search_button and search_query.strip():
            with st.spinner("Searching products..."):
                products = assistant.search_products(search_query)
                
                if products:
                    st.success(f"Found {len(products)} products")
                    
                    for i, product in enumerate(products):
                        with st.expander(f"{product.metadata.get('product_name', 'Unknown Product')}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Price:** ₹{product.metadata.get('price', 0):,}")
                                st.write(f"**Category:** {product.metadata.get('category', 'Unknown')}")
                                st.write(f"**Description:** {product.metadata.get('description', 'No description')}")
                                st.write(f"**Stock:** {product.metadata.get('stock_quantity', 0)} units")
                            
                            with col2:
                                if st.button(f"Add to Cart", key=f"add_{i}"):
                                    assistant.add_to_cart(
                                        product.metadata.get('product_name'),
                                        product.metadata.get('price'),
                                        product.metadata.get('product_id')
                                    )
                                    st.success("Added to cart!")
                                    st.rerun()
                else:
                    st.warning("No products found. Try a different search term.")
    
    with tab3:
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
                    response = assistant.handle_checkout("place order")
                    st.success(response)
            
            with col2:
                if st.button("Clear Cart"):
                    assistant.clear_cart()
                    st.success("Cart cleared!")
                    st.rerun()
        else:
            st.info("Your cart is empty. Add some products to get started!")
    
    with tab4:
        st.header("Order Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show All Orders"):
                orders = assistant.list_user_orders()
                st.text_area("Order History", orders, height=300)
        
        with col2:
            if st.button("Latest Order Status"):
                status = assistant.handle_order_status("latest order")
                st.text_area("Order Status", status, height=300)
    
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