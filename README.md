# LLM Data Verifier

A lightweight Python tool for verifying data via LLM search queries. This tool takes JSON input with questions and expected answers, queries an LLM API, and verifies if the responses match the expected results.

## Features

- **Parallel Processing**: Support for concurrent API queries
- **Incremental Mode**: Only reprocess failed queries from previous runs
- **Flexible API Key Management**: Via command line or environment variable
- **Detailed Logging**: Saves full API responses for debugging
- **Smart Answer Matching**: Uses optimized substring edit distance algorithm
- **Cached Response Reuse**: Automatically reuses completed API responses from tmp directory

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python llm_verifier.py -i input.json -o output.json -k YOUR_API_KEY
```

### Command Line Arguments

- `-i, --input_file`: Input JSON file path (required)
- `-o, --output_file`: Output JSON file path (required)
- `-t, --tmp_path`: Temporary path for API responses (default: "tmp")
- `-p, --parallel`: Number of parallel queries (default: 1)
- `-I, --incremental`: Incremental mode - only rerun failed queries
- `-k, --api_key`: Search API key (or set `OPENROUTER_API_KEY` env var)
- `-c, --config`: Configuration file path (default: "config.json")

### Configuration File

Create a `config.json` file to set default values:

```json
{
  "api_key": "your_openrouter_api_key_here",
  "base_url": "https://openrouter.ai/api/v1/responses",
  "model": "openai/o4-mini:online",
  "max_output_tokens": 30000,
  "timeout": 60,
  "parallel_queries": 3,
  "tmp_path": "tmp",
  "similarity_threshold": 0.2
}
```

**Configuration Priority** (highest to lowest):
1. Command line arguments
2. Configuration file values
3. Environment variables
4. Default values

### Environment Variables

Set your API key as an environment variable:
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

## Input Format

```json
[
  {
    "id": "example_1",
    "type": "research",
    "question": "Find a paper published in 2025 that discusses training strategies for embodied reasoning",
    "modality": "text",
    "title": "Training Strategies for Efficient Embodied Reasoning",
    "answer": ["2505.08243"],
    "reason": "This paper introduces novel training methodologies..."
  }
]
```

## Output Format

The output maintains the same structure but adds a `trial_search_correct` field:

```json
[
  {
    "id": "example_1",
    "type": "research",
    "question": "Find a paper published in 2025 that discusses training strategies for embodied reasoning",
    "modality": "text", 
    "title": "Training Strategies for Efficient Embodied Reasoning",
    "answer": ["2505.08243"],
    "reason": "This paper introduces novel training methodologies...",
    "trial_search_correct": "yes"
  }
]
```

### Success Values

- `"yes"`: Query was correctly answered (found sufficient matching content)
- `"no"`: Query was answered but incorrectly (no sufficient matches found)
- `"unverified"`: Query failed or couldn't be processed

### Matching Algorithm

The tool uses an optimized substring edit distance algorithm that:

- **Finds the best matching substring** in the response text for each expected answer
- **Uses dynamic programming** with matrix reuse for efficiency
- **Supports configurable similarity threshold** (default 20% of target length)
- **Multi-answer items**: Requires ~~at least 50%~~ all of expected answers to match
- **Prioritizes titles over arXiv IDs** for verification

## Examples

### Basic Usage
```bash
python llm_verifier.py -i data.json -o results.json -k sk-your-key
```

### Parallel Processing
```bash
python llm_verifier.py -i data.json -o results.json -p 5 -k sk-your-key
```

### Incremental Mode
```bash
python llm_verifier.py -i data.json -o results.json -I -k sk-your-key
```

### Custom Temporary Directory
```bash
python llm_verifier.py -i data.json -o results.json -t ./temp_responses -k sk-your-key
```

## API Response Storage

Full API responses are saved as `{id}.json` files in the temporary directory for debugging and analysis purposes.

## Error Handling

The tool handles various error conditions gracefully:
- Network timeouts and API errors
- JSON parsing errors
- File I/O errors
- Missing or invalid API keys

Failed queries are marked as "unverified" and can be reprocessed using incremental mode.
