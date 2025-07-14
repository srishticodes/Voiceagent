#!/usr/bin/env python3
"""
Streamlit version of Walmart Voice Assistant - Cloud Compatible
"""

import streamlit as st
import pandas as pd
import os
import json
import tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from datetime import datetime, timedelta
import re
from typing import cast

# Page configuration
st.set_page_config(
    page_title="Walmart AI Shopping Assistant",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CloudWalmartAssistant:
    def __init__(self):
        self.current_user = None
        self.cart = []
        self.checkout_state = None
        self.checkout_data = {}
        self.setup_models()
        self.setup_product_search()
        
    @st.cache_resource
    def setup_models(_self):
        """Initialize AI models for cloud deployment"""
        try:
            # Use sentence-transformers for embeddings (cloud compatible)
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            return None
    
    def setup_product_search(self):
        """Setup product search with sentence transformers"""
        try:
            # Load inventory
            if os.path.exists("walmart_inventory.csv"):
                self.inventory_df = pd.read_csv("walmart_inventory.csv", quoting=1)
                self.inventory_df = self.inventory_df[self.inventory_df['stock_status'] == 'In Stock']
                
                # Create embeddings for products
                self.embedding_model = self.setup_models()
                if self.embedding_model:
                    product_texts = []
                    for _, row in self.inventory_df.iterrows():
                        text = f"{row['product_name']} {row['product_category']} {row['product_description']}"
                        product_texts.append(text)
                    
                    self.product_embeddings = self.embedding_model.encode(product_texts)
                    st.success("Product search initialized successfully")
                else:
                    st.error("Failed to initialize embedding model")
            else:
                st.error("walmart_inventory.csv not found")
                self.inventory_df = pd.DataFrame()
        except Exception as e:
            st.error(f"Error setting up product search: {e}")
            self.inventory_df = pd.DataFrame()
    
    def search_products(self, query, top_k=5):
        """Search products using sentence transformers"""
        try:
            if self.embedding_model is None or self.inventory_df.empty:
                return []
            
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    product = self.inventory_df.iloc[idx]
                    results.append({
                        'product_id': product['product_id'],
                        'product_name': product['product_name'],
                        'category': product['product_category'],
                        'price': product['price_inr'],
                        'description': product['product_description'],
                        'stock_quantity': product['stock_quantity'],
                        'similarity': similarities[idx]
                    })
            
            return results
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def authenticate_user(self, user_id):
        """Authenticate user and load their data"""
        try:
            if not os.path.exists("users.csv"):
                return False
                
            users_df = pd.read_csv("users.csv")
            user_series = users_df[users_df['user_id'] == user_id]

            if user_series.empty:
                return False

            self.current_user = user_series.iloc[0].to_dict()
            
            # Load addresses
            if os.path.exists("user_addresses.csv"):
                addresses_df = pd.read_csv("user_addresses.csv")
                user_addresses = addresses_df[addresses_df['user_id'] == user_id]
                self.current_user['addresses'] = [row for _, row in user_addresses.iterrows()]
            else:
                self.current_user['addresses'] = []

            # Load payment methods
            if os.path.exists("user_payment_methods.csv"):
                payments_df = pd.read_csv("user_payment_methods.csv")
                user_payments = payments_df[payments_df['user_id'] == user_id]
                self.current_user['payment_methods'] = [row for _, row in user_payments.iterrows()]
            else:
                self.current_user['payment_methods'] = []

            # Load user's cart
            self.cart = self.load_user_cart(user_id)
            
            return True
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return False
    
    def load_user_cart(self, user_id):
        """Load user's cart from CSV"""
        try:
            if not os.path.exists("user_carts.csv"):
                return []
                
            df = pd.read_csv("user_carts.csv")
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
            st.error(f"Error loading cart: {e}")
            return []
    
    def save_user_cart(self, user_id, cart_items):
        """Save user's cart to CSV"""
        try:
            cart_file = "user_carts.csv"
            cart_columns = [
                'cart_id', 'user_id', 'product_id', 'product_name', 
                'price_inr', 'quantity', 'added_at', 'status'
            ]

            if os.path.exists(cart_file):
                df = pd.read_csv(cart_file)
            else:
                df = pd.DataFrame(columns=pd.Index(cart_columns))

            for col in cart_columns:
                if col not in df.columns:
                    df[col] = None

            df = df[~((df['user_id'] == user_id) & (df['status'] == 'active'))]
            
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
            
            df.to_csv(cart_file, index=False)
            
        except Exception as e:
            st.error(f"Error saving cart: {e}")
    
    def add_to_cart(self, product_name, price, product_id=None):
        """Add item to cart"""
        if not self.current_user:
            return "Please login first"
            
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
        self.save_user_cart(self.current_user['user_id'], self.cart)
        return f"Added {product_name} to cart"
    
    def get_cart_total(self):
        """Get cart total"""
        return sum(item['price'] * item['quantity'] for item in self.cart)
    
    def clear_cart(self):
        """Clear cart"""
        self.cart.clear()
        if self.current_user:
            self.save_user_cart(self.current_user['user_id'], self.cart)
        return "Cart cleared"
    
    def simple_checkout(self):
        """Simplified checkout for demo"""
        if not self.current_user or not self.cart:
            return "Cannot checkout - please login and add items to cart"
        
        try:
            order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
            order_date = datetime.now()
            
            # Simple order creation
            order_columns = [
                'order_id', 'user_id', 'product_id', 'product_name', 'price_inr', 
                'quantity', 'order_date', 'delivery_address', 'payment_method', 
                'delivery_status', 'replacement_eligible_until', 'refund_eligible_until'
            ]

            if os.path.exists('user_orders.csv'):
                orders_df = pd.read_csv('user_orders.csv')
            else:
                orders_df = pd.DataFrame(columns=pd.Index(order_columns))

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
                    'delivery_address': 'Default Address',
                    'payment_method': 'Credit Card',
                    'delivery_status': 'Order Placed - Processing',
                    'replacement_eligible_until': (order_date + timedelta(days=15)).isoformat(),
                    'refund_eligible_until': (order_date + timedelta(days=15)).isoformat()
                }
                new_orders.append(order_item)
            
            if new_orders:
                new_orders_df = pd.DataFrame(new_orders)
                updated_orders_df = pd.concat([orders_df, new_orders_df], ignore_index=True)
                updated_orders_df.to_csv('user_orders.csv', index=False)

            total_amount = self.get_cart_total()
            
            # Clear cart
            self.cart.clear()
            self.save_user_cart(self.current_user['user_id'], self.cart)
            
            return f"Order placed successfully! Order ID: {order_id}. Total: ₹{total_amount:,}"
            
        except Exception as e:
            return f"Checkout error: {str(e)}"

def main():
    st.title("Walmart AI Shopping Assistant")
    st.markdown("### AI-powered shopping assistant with product search and cart management")
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = CloudWalmartAssistant()
    
    assistant = st.session_state.assistant
    
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
    tab1, tab2, tab3 = st.tabs(["Search", "Cart", "Orders"])
    
    with tab1:
        st.header("Product Search")
        
        search_query = st.text_input("Search for products:", placeholder="e.g., headphones, laptops, shoes")
        
        if st.button("Search", type="primary") and search_query.strip():
            with st.spinner("Searching products..."):
                products = assistant.search_products(search_query)
                
                if products:
                    st.success(f"Found {len(products)} products")
                    
                    for i, product in enumerate(products):
                        with st.expander(f"{product['product_name']} - ₹{product['price']:,}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Category:** {product['category']}")
                                st.write(f"**Description:** {product['description']}")
                                st.write(f"**Stock:** {product['stock_quantity']} units")
                                st.write(f"**Match Score:** {product['similarity']:.2f}")
                            
                            with col2:
                                if st.button(f"Add to Cart", key=f"add_{i}"):
                                    result = assistant.add_to_cart(
                                        product['product_name'],
                                        product['price'],
                                        product['product_id']
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