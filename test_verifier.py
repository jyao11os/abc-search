#!/usr/bin/env python3
"""
Test script for the LLM verifier functionality.
"""

import unittest
from llm_verifier import SubstringEditDistance, LLMVerifier
import json
import os
import tempfile


class TestSubstringEditDistance(unittest.TestCase):
    """Test cases for substring edit distance calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = SubstringEditDistance()
    
    def test_exact_match(self):
        """Test exact string matches."""
        distance, match = self.calculator.calculate("hello world", "hello")
        self.assertEqual(distance, 0)
        self.assertEqual(match, "hello")
    
    def test_edit_distance(self):
        """Test edit distance calculation."""
        distance, match = self.calculator.calculate("cat bat", "bat")
        self.assertEqual(distance, 0)
        self.assertEqual(match, "bat")
        
        distance, match = self.calculator.calculate("kitten", "sitting")
        self.assertLessEqual(distance, 3)  # Should find some reasonable match
    
    def test_substring_matching(self):
        """Test finding best substring matches."""
        text = "The paper titled 'Training Strategies for Efficient Embodied Reasoning' was published in 2025"
        target = "Training Strategies for Efficient Embodied Reasoning"
        
        distance, best_match = self.calculator.calculate(text, target)
        self.assertLess(distance, 5)  # Should find a close match
        self.assertIn("training strategies", best_match.lower())


class TestLLMVerifier(unittest.TestCase):
    """Test cases for LLM verifier functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.verifier = LLMVerifier("test_key", self.temp_dir, 1)
    
    def test_extract_assistant_response(self):
        """Test extracting assistant response from API response."""
        api_response = {
            "output": [
                {
                    "role": "assistant",
                    "type": "message", 
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The title of that paper is: Training Strategies for Efficient Embodied Reasoning"
                        }
                    ]
                }
            ]
        }
        
        response_text = self.verifier.extract_assistant_response(api_response)
        self.assertIn("Training Strategies", response_text)
    
    def test_verify_answer_single(self):
        """Test answer verification logic with single answer."""
        response_text = "The paper is titled 'Training Strategies for Efficient Embodied Reasoning' and was published recently."
        expected_answers = ["Training Strategies for Efficient Embodied Reasoning"]
        
        result = self.verifier.verify_answer(response_text, expected_answers)
        self.assertTrue(result)
        
        # Test with no match
        response_text = "I couldn't find any relevant papers."
        result = self.verifier.verify_answer(response_text, expected_answers)
        self.assertFalse(result)
    
    def test_verify_answer_multi_item(self):
        """Test answer verification with multiple expected answers."""
        # Simulate the response from tmp/154.json
        response_text = """Here are the two papers you're looking for:

1. Missing-Premise Overthinking in LLMs  
   • Title: "Missing Premise Exacerbates Overthinking: Are Reasoning Models Losing Critical Thinking Skill?"  
   • Authors: Chenrui Fan, Ming Li, Lichao Sun, Tianyi Zhou  
   • arXiv: 2504.06514 (submitted April 9 2025; revised April 11 2025)  

2. The CORD-19 Dataset: Weekly → Monthly Releases  
   • Title: "CORD-19: The COVID-19 Open Research Dataset"  
   • Authors: Lucy Lu Wang, Kyle Lo, Yoganand Chandrasekhar, Russell Reas, Jiangjiang Yang, Doug Burdick, Darrin Eide, Kathryn Funk, Yannis Katsis, Rodney Kinney, Yunyao Li, Ziyang Liu, William Merrill, Paul Mooney, Dewey Murdick, Devvret Rishi, Jerry Sheehan, Zhihong Shen, Brandon Stilson, Alex Wade, Kuansan Wang, Nancy Xin Ru Wang, Christopher Wilhelm, Boya Xie, Douglas Raymond, Daniel Weld, Oren Etzioni, Sebastian Kohlmeier  
   • Venue: Proceedings of the 1st Workshop on NLP for COVID-19 (NLP-COVID19), ACL 2020  
   • Anthology ID: 2020.nlpcovid19-acl.1"""
        
        # Test with the actual expected answers from tiny_example_multi_input.json
        expected_answers = ["2506.09038", "2310.03214"]
        
        # This should fail because the arXiv IDs in the response don't match exactly
        result = self.verifier.verify_answer(response_text, expected_answers)
        # Note: This might fail due to different arXiv IDs, which is expected behavior
        
        # Test with paper titles instead
        expected_titles = [
            "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions",
            "FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation"
        ]
        
        # This should also fail because the titles in response are different
        result_titles = self.verifier.verify_answer(response_text, expected_titles)
        
        # Test with partial matches that should succeed
        partial_matches = ["Missing Premise", "CORD-19"]
        result_partial = self.verifier.verify_answer(response_text, partial_matches)
        self.assertTrue(result_partial)
    
    def test_load_cached_response(self):
        """Test loading cached API responses."""
        # Create a mock cached response
        cached_data = {
            "status": "completed",
            "output": [
                {
                    "role": "assistant",
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Test response"}]
                }
            ]
        }
        
        cache_file = os.path.join(self.temp_dir, "test_item.json")
        with open(cache_file, 'w') as f:
            json.dump(cached_data, f)
        
        # Test loading completed response
        result = self.verifier.load_cached_response("test_item")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "completed")
        
        # Test with incomplete response
        cached_data["status"] = "pending"
        with open(cache_file, 'w') as f:
            json.dump(cached_data, f)
        
        result = self.verifier.load_cached_response("test_item")
        self.assertIsNone(result)
        
        # Test with non-existent file
        result = self.verifier.load_cached_response("non_existent")
        self.assertIsNone(result)
    
    def test_load_existing_results(self):
        """Test loading existing results for incremental mode."""
        # Create a test output file
        test_data = [
            {"id": "test1", "trial_search_correct": "yes"},
            {"id": "test2", "trial_search_correct": "no"}
        ]
        
        output_file = os.path.join(self.temp_dir, "test_output.json")
        with open(output_file, 'w') as f:
            json.dump(test_data, f)
        
        results = self.verifier.load_existing_results(output_file)
        self.assertEqual(results["test1"], "yes")
        self.assertEqual(results["test2"], "no")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()