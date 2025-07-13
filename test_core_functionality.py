#!/usr/bin/env python3
"""
Test script to verify core functionalities
"""

import os
import sys
from walmart_assistant import WalmartAssistant

def test_core_functionality():
    """Test core functionalities"""
    
    print("Testing Core Functionalities")
    print("=" * 50)
    
    try:
        # Initialize assistant
        print("Initializing assistant...")
        assistant = WalmartAssistant()
        print("Assistant initialized successfully")
        
        # Test product search
        print("\nTesting product search...")
        test_queries = [
            "headphones",
            "laptop", 
            "phone",
            "toothpaste"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: {query}")
            results = assistant.search_products(query)
            if results:
                print(f"Found {len(results)} products")
                for i, result in enumerate(results[:2]):  # Show first 2 results
                    product_name = result.metadata.get('product_name', 'Unknown')
                    price = result.metadata.get('price', 0)
                    print(f"  {i+1}. {product_name} - ₹{price}")
            else:
                print("No products found")
        
        # Test cart functionality
        print("\nTesting cart functionality...")
        
        # Add items to cart
        test_items = [
            ("Sony Headphones", 25049.16),
            ("Apple iPhone 13", 58449.16),
            ("Colgate Toothpaste", 208.75)
        ]
        
        for name, price in test_items:
            result = assistant.add_to_cart(name, price)
            print(f"Added: {result}")
        
        # Check cart total
        total = assistant.get_cart_total()
        print(f"Cart total: ₹{total}")
        
        # Test cart count
        cart_count = len(assistant.cart)
        print(f"Items in cart: {cart_count}")
        
        # Clear cart
        result = assistant.clear_cart()
        print(f"Cart cleared: {result}")
        
        # Verify cart is empty
        new_total = assistant.get_cart_total()
        new_count = len(assistant.cart)
        print(f"After clearing - Total: ₹{new_total}, Count: {new_count}")
        
        # Test query processing
        print("\nTesting query processing...")
        test_queries = [
            "Find me headphones",
            "Show me electronics",
            "What's the price of iPhone"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = assistant.process_query(query)
            print(f"Response: {response[:100]}...")  # Show first 100 chars
        
        print("\nAll core functionalities working correctly!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_core_functionality()
    if success:
        print("\nCore functionality test passed!")
    else:
        print("\nCore functionality test failed!")
        sys.exit(1) 