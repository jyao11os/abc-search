#!/usr/bin/env python3
"""
LLM Data Verification Tool
Verifies data by querying LLM search API and comparing results with expected answers.
"""

import argparse
import json
import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Optional
import time


class SubstringEditDistance:
    """Calculate minimal edit distance for substring matching with optimized matrix reuse."""
    
    def __init__(self, initial_size: int = 1024):
        """Initialize with a default matrix size
        
        Args:
            initial_size: Initial size for both dimensions of the dp matrix
        """
        self.size = initial_size
        self.dp = [[0] * initial_size for _ in range(initial_size)]
    
    def _resize_matrix(self, required_rows: int, required_cols: int):
        """Resize the dp matrix if current size is insufficient
        
        Args:
            required_rows: Number of rows needed
            required_cols: Number of columns needed
        """
        new_size = max(required_rows, required_cols)
        if new_size > self.size:
            # Double the size until it's sufficient
            while self.size < new_size:
                self.size *= 2
            # Create new matrix
            self.dp = [[0] * self.size for _ in range(self.size)]
    
    def calculate(self, source: str, target: str) -> tuple[int, str]:
        """Calculate minimal edit distance between target string and any substring of source string
        
        Args:
            source: The main text to search within (e.g., system output or document text)
            target: The text pattern to search for (e.g., ground truth or query text)
            
        Returns:
            A tuple of (minimal edit distance, best matching substring from source)
        """
        source, target = source.lower(), target.lower()
        m, n = len(source), len(target)
        
        # Resize matrix if necessary
        self._resize_matrix(n + 1, m + 1)
        
        # Initialize first row
        for j in range(m + 1):
            self.dp[0][j] = 0
        
        # Initialize first column
        for i in range(1, n + 1):
            self.dp[i][0] = i
        
        # Fill the dp table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if target[i - 1] == source[j - 1]:
                    self.dp[i][j] = self.dp[i - 1][j - 1]
                else:
                    self.dp[i][j] = min(
                        self.dp[i - 1][j - 1] + 1,  # replace
                        self.dp[i - 1][j] + 1,      # delete
                        self.dp[i][j - 1] + 1,      # insert
                    )
        
        # Find minimum in the last row and its position
        min_dist = float("inf")
        end_pos = 0
        for j in range(m + 1):
            if self.dp[n][j] < min_dist:
                min_dist = self.dp[n][j]
                end_pos = j
        
        # Backtrack to find the start position
        start_pos = end_pos
        curr_row = n
        curr_col = end_pos
        
        while curr_row > 0:
            candidates = [
                (curr_row - 1, curr_col - 1),  # diagonal
                (curr_row - 1, curr_col),      # up
                (curr_row, curr_col - 1),      # left
            ]
            
            next_pos = min(
                (pos for pos in candidates if pos[1] >= 0),
                key=lambda pos: self.dp[pos[0]][pos[1]],
            )
            
            if next_pos[1] < curr_col:
                start_pos = next_pos[1]
            
            curr_row, curr_col = next_pos
        
        return min_dist, source[start_pos:end_pos]
    
    def get_matrix_size(self) -> int:
        """Returns current matrix size"""
        return self.size


class LLMVerifier:
    """Main class for LLM-based data verification."""
    
    def __init__(self, api_key: str, tmp_path: str = "tmp", parallel: int = 1, config: Dict[str, Any] = None):
        # Load config defaults
        default_config = {
            "base_url": "https://openrouter.ai/api/v1/responses",
            "model": "openai/o4-mini:online",
            "max_output_tokens": 30000,
            "timeout": 60,
            "similarity_threshold": 0.2
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        self.api_key = api_key
        self.tmp_path = tmp_path
        self.parallel = parallel
        self.base_url = default_config["base_url"]
        self.model = default_config["model"]
        self.max_output_tokens = default_config["max_output_tokens"]
        self.timeout = default_config["timeout"]
        self.similarity_threshold = default_config["similarity_threshold"]
        
        # Initialize the substring edit distance calculator
        self.substring_calculator = SubstringEditDistance()
        
        # Create tmp directory if it doesn't exist
        os.makedirs(tmp_path, exist_ok=True)
    
    def load_cached_response(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Load cached API response if it exists and is completed."""
        cache_file = os.path.join(self.tmp_path, f"{item_id}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached_response = json.load(f)
            
            # Check if the response is completed
            if cached_response.get('status') == 'completed':
                print(f"Using cached response for item {item_id}")
                return cached_response
            else:
                print(f"Cached response for item {item_id} is not completed, will re-query")
                return None
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading cached response for {item_id}: {e}")
            return None
    
    def query_llm(self, question: str, item_id: str) -> Dict[str, Any]:
        """Query the LLM API with a question, using cached response if available."""
        # Check for cached response first
        cached_response = self.load_cached_response(item_id)
        if cached_response:
            return cached_response
        
        print(f"Making new API request for item {item_id}...")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'model': self.model,
            'input': question,
            'max_output_tokens': self.max_output_tokens,
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            # Save full response to tmp file
            with open(os.path.join(self.tmp_path, f"{item_id}.json"), 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
        
        except requests.exceptions.RequestException as e:
            print(f"API request failed for {item_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error for {item_id}: {e}")
            return None
    
    def extract_assistant_response(self, api_response: Dict[str, Any]) -> str:
        """Extract the assistant's text response from API response."""
        try:
            if 'output' in api_response:
                for output_item in api_response['output']:
                    if output_item.get('role') == 'assistant' and output_item.get('type') == 'message':
                        content = output_item.get('content', [])
                        for content_item in content:
                            if content_item.get('type') == 'output_text':
                                return content_item.get('text', '')
            return ""
        except (KeyError, TypeError):
            return ""
    
    def verify_answer(self, response_text: str, expected_answers: List[str]) -> bool:
        """Verify if the response contains the expected answer(s)."""
        if not response_text or not expected_answers:
            return False
        
        # Count how many expected answers are found
        found_count = 0
        
        for expected in expected_answers:
            # Use the optimized substring edit distance calculator
            edit_distance, best_match = self.substring_calculator.calculate(response_text, expected)
            
            # Consider "very close" if edit distance is small relative to target length
            threshold = max(1, len(expected) * self.similarity_threshold)
            if edit_distance <= threshold:
                found_count += 1
                print(f"Found match for '{expected}': '{best_match}' (distance: {edit_distance})")
        
        ## For multi-item answers, require at least half of the expected answers to be found
        #required_matches = max(1, len(expected_answers) // 2) if len(expected_answers) > 1 else 1
        required_matches = len(expected_answers)
        success = found_count >= required_matches
        
        print(f"Found {found_count}/{len(expected_answers)} expected answers (required: {required_matches})")
        return success
    
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item from the input data."""
        item_id = item.get('id', 'unknown')
        question = item.get('question', '')
        
        # Use titles if available, otherwise fall back to answer field
        expected_answers = item.get('titles', item.get('answer', []))
        
        print(f"Processing item {item_id}...")
        print(f"Expected answers: {expected_answers}")
        
        # Query the LLM
        api_response = self.query_llm(question, item_id)
        
        if api_response is None:
            item['trial_search_correct'] = 'unverified'
            return item
        
        # Extract assistant response
        response_text = self.extract_assistant_response(api_response)
        
        if not response_text:
            item['trial_search_correct'] = 'unverified'
            return item
        
        # Verify the answer
        is_correct = self.verify_answer(response_text, expected_answers)
        item['trial_search_correct'] = 'yes' if is_correct else 'no'
        
        return item
    
    def load_existing_results(self, output_file: str) -> Dict[str, str]:
        """Load existing results for incremental mode."""
        if not os.path.exists(output_file):
            return {}
        
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            results = {}
            for item in data:
                if 'trial_search_correct' in item:
                    results[item.get('id', '')] = item['trial_search_correct']
            
            return results
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def process_data(self, input_file: str, output_file: str, incremental: bool = False):
        """Process the input data and generate output."""
        # Load input data
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
        
        # Load existing results for incremental mode
        existing_results = {}
        if incremental:
            existing_results = self.load_existing_results(output_file)
            print(f"Loaded {len(existing_results)} existing results")
        
        # Filter items to process
        items_to_process = []
        for item in data:
            item_id = item.get('id', '')
            if incremental and item_id in existing_results:
                # Skip if already processed successfully
                if existing_results[item_id] in ['yes', 'no']:
                    item['trial_search_correct'] = existing_results[item_id]
                    continue
            items_to_process.append(item)
        
        print(f"Processing {len(items_to_process)} items...")
        
        # Process items
        if self.parallel > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                future_to_item = {executor.submit(self.process_item, item): item 
                                for item in items_to_process}
                
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        # Update the original item in data
                        for i, item in enumerate(data):
                            if item.get('id') == result.get('id'):
                                data[i] = result
                                break
                    except Exception as e:
                        print(f"Error processing item: {e}")
        else:
            # Sequential processing
            for i, item in enumerate(data):
                if item in items_to_process:
                    data[i] = self.process_item(item)
        
        # Save results
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        except IOError as e:
            print(f"Error saving output file: {e}")
            sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if (not config_path) or (not os.path.exists(config_path)):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Verify data via LLM search")
    parser.add_argument("-i", "--input_file", required=True, 
                       help="Input JSON file path")
    parser.add_argument("-o", "--output_file", required=True,
                       help="Output JSON file path")
    parser.add_argument("-t", "--tmp_path", default="tmp",
                       help="Temporary path for storing API responses")
    parser.add_argument("-p", "--parallel", type=int, default=1,
                       help="Number of parallel queries")
    parser.add_argument("-I", "--incremental", action="store_true",
                       help="Incremental mode - only rerun failed queries")
    parser.add_argument("-k", "--api_key", 
                       help="Search API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("-c", "--config",# default="config.json",
                       help="Configuration file path (default: config.json)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get API key (command line > config file > environment variable)
    api_key = args.api_key or config.get('api_key') or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: API key required. Use -k flag, set in config.json, or set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)
    
    # Override config with command line arguments
    if hasattr(args, 'parallel') and args.parallel != 1:
        config['parallel_queries'] = args.parallel
    if hasattr(args, 'tmp_path') and args.tmp_path != "tmp":
        config['tmp_path'] = args.tmp_path
    
    # Create verifier and process data
    parallel_queries = config.get('parallel_queries', args.parallel)
    tmp_path = config.get('tmp_path', args.tmp_path)
    
    verifier = LLMVerifier(api_key, tmp_path, parallel_queries, config)
    verifier.process_data(args.input_file, args.output_file, args.incremental)


if __name__ == "__main__":
    main()
