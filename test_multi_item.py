#!/usr/bin/env python3
"""
Test script specifically for multi-item matching using the tiny_example_multi_input.json
and tmp/154.json files.
"""

import json
import os
from llm_verifier import LLMVerifier, SubstringEditDistance


def test_multi_item_matching():
    """Test multi-item matching with real data."""
    print("=== Multi-Item Matching Test ===")
    
    # Load the input data
    with open('tiny_example_multi_input.json', 'r') as f:
        input_data = json.load(f)
    
    item = input_data[0]  # Get the first (and only) item
    print(f"Testing item ID: {item['id']}")
    print(f"Question: {item['question']}")
    print(f"Expected titles: {item['titles']}")
    
    # Load the cached response
    with open('tmp/154.json', 'r') as f:
        api_response = json.load(f)
    
    print(f"API response status: {api_response.get('status')}")
    
    # Create verifier and extract response text
    verifier = LLMVerifier("test_key", "tmp", 1)
    response_text = verifier.extract_assistant_response(api_response)
    
    print(f"\nExtracted response text (first 500 chars):")
    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
    
    # Test verification with expected titles only
    print("\n=== Testing with paper titles ===")
    expected_titles = item['titles']
    result_titles = verifier.verify_answer(response_text, expected_titles)
    print(f"Result with titles: {'PASS' if result_titles else 'FAIL'}")
    
    # Test with partial/keyword matches that should work
    print("\n=== Testing with partial matches ===")
    partial_matches = ["Missing Premise", "CORD-19"]
    result_partial = verifier.verify_answer(response_text, partial_matches)
    print(f"Result with partial matches: {'PASS' if result_partial else 'FAIL'}")
    
    # Test individual substring matching for key terms
    print("\n=== Individual Substring Matching Tests ===")
    test_targets = [
        "AbstentionBench",
        "FreshLLMs", 
        "Missing Premise",
        "CORD-19",
        "Overthinking"
    ]
    
    for target in test_targets:
        print(f"\nTesting target: '{target}'")
        distance, best_match = verifier.substring_calculator.calculate(response_text, target)
        threshold = max(1, len(target) * 0.2)
        is_match = distance <= threshold
        print(f"  Best match: '{best_match}'")
        print(f"  Edit distance: {distance} (threshold: {threshold:.1f})")
        print(f"  Result: {'MATCH' if is_match else 'NO MATCH'}")
    
    # Test the actual process_item method
    print("\n=== Testing process_item method ===")
    processed_item = verifier.process_item(item.copy())
    print(f"Final result: {processed_item.get('trial_search_correct', 'NOT SET')}")


def test_substring_performance():
    """Test the performance of the optimized substring matching."""
    print("\n=== Substring Matching Performance Test ===")
    
    # Test with a long text and various targets
    long_text = "This is a very long text that contains many words and phrases. " * 100
    targets = ["very long text", "contains many", "nonexistent phrase", "words and phrases"]
    
    import time
    calculator = SubstringEditDistance()
    
    for target in targets:
        start_time = time.time()
        distance, best_match = calculator.calculate(long_text, target)
        end_time = time.time()
        
        print(f"Target: '{target}'")
        print(f"  Time: {end_time - start_time:.4f}s")
        print(f"  Best match: '{best_match}'")
        print(f"  Distance: {distance}")


if __name__ == "__main__":
    # Check if required files exist
    if not os.path.exists('tiny_example_multi_input.json'):
        print("Error: tiny_example_multi_input.json not found")
        exit(1)
    
    if not os.path.exists('tmp/154.json'):
        print("Error: tmp/154.json not found")
        exit(1)
    
    test_multi_item_matching()
    test_substring_performance()
    
    print("\n=== Test Complete ===")