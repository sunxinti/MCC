#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author. Xinti Sun

# ===================== Configuration Section =====================
import os
import re
import json
import time
import sys
import random
import traceback
import requests
import pandas as pd
import argparse
import datetime
import configparser
import concurrent.futures 
from typing import Tuple, List, Dict, Any, Optional
from openai import OpenAI 
import blobfile as bf  

# Custom output class for simultaneous output to console and log file
class TeeOutput:
    """Class for sending output to both terminal and log file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a", encoding="utf-8")
        # Write timestamp at the beginning of log file
        separator = "=" * 50
        self.logfile.write(f"\n{separator}\n")
        self.logfile.write(f"Log start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logfile.write(f"{separator}\n\n")
        self.logfile.flush()

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()  # Ensure immediate file write

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
        
    def close(self):
        """Close log file"""
        if self.logfile:
            # Write timestamp at the end of log file
            separator = "=" * 50
            self.logfile.write(f"\n{separator}\n")
            self.logfile.write(f"Log end time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.logfile.write(f"{separator}\n\n")
            self.logfile.close()
            self.logfile = None

# API configuration file path
CONFIG_FILE = "api_config.ini"
def setup_api_config():
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        print("Found API configuration file, loading...")
        config.read(CONFIG_FILE, encoding='utf-8')
        
        required_sections = {
            'GPT': ['api_key', 'api_url'],
            'QWEN': ['api_key', 'api_url'], 
            'DEEPSEEK': ['api_key', 'api_url']
        }
        
        config_complete = True
        for section, keys in required_sections.items():
            if not config.has_section(section):
                config_complete = False
                break
            for key in keys:
                if not config.has_option(section, key) or not config.get(section, key).strip():
                    config_complete = False
                    break
        
        if config_complete:
            print("API configuration loaded successfully!")
            return config
        else:
            print("Configuration file incomplete, reconfiguration needed...")
    
    print("="*60)
    print("Welcome to MCC!")
    print("First run requires API key configuration, please enter your API information as prompted.")
    print("Press ENTER to use default URLs for quick testing.")
    print("Configuration will be saved locally, no need to re-enter for subsequent runs.")
    print("="*60)
    
    # GPT API configuration
    print("\n【GPT API Configuration】")
    print("Please enter your GPT API configuration information:")
    gpt_api_url = input("GPT API URL (Press ENTER for default: https://api.chatanywhere.tech/v1/chat/completions) API documentation: @https://chatanywhere.apifox.cn/: ").strip()
    if not gpt_api_url:
        gpt_api_url = "https://api.chatanywhere.tech/v1/chat/completions"
    gpt_api_key = input("GPT API Key: ").strip()
    
    # Qwen API configuration
    print("\n【Qwen API Configuration】")
    print("Please enter your Qwen API configuration information:")
    qwen_api_url = input("Qwen API URL (Press ENTER for default: https://api.siliconflow.cn/v1/chat/completions) API documentation: @https://cloud.siliconflow.cn/me/models: ").strip()
    if not qwen_api_url:
        qwen_api_url = "https://api.siliconflow.cn/v1/chat/completions"
    qwen_api_key = input("Qwen API Key: ").strip()
    
    # DeepSeek API configuration
    print("\n【DeepSeek API Configuration】")
    print("Please enter your DeepSeek API configuration information:")
    deepseek_api_url = input("DeepSeek API URL (Press ENTER for default: https://api.deepseek.com) API documentation: @https://api-docs.deepseek.com/zh-cn/: ").strip()
    if not deepseek_api_url:
        deepseek_api_url = "https://api.deepseek.com"
    deepseek_api_key = input("DeepSeek API Key: ").strip()
    
    if not all([gpt_api_key, qwen_api_key, deepseek_api_key]):
        print("\nError: API Key cannot be empty! Please restart the program and enter complete API configuration.")
        sys.exit(1)
    
    # Store configuration
    config['GPT'] = {
        'api_key': gpt_api_key,
        'api_url': gpt_api_url
    }
    config['QWEN'] = {
        'api_key': qwen_api_key,
        'api_url': qwen_api_url
    }
    config['DEEPSEEK'] = {
        'api_key': deepseek_api_key,
        'api_url': deepseek_api_url
    }
    
    # Write configuration file
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            config.write(f)
        print(f"\n✓ API configuration saved to {CONFIG_FILE}")
        print("Configuration complete! Starting HealthBench model debate system...")
    except Exception as e:
        print(f"\nWarning: Failed to save configuration file: {e}")
        print("Program will continue running, but configuration will need to be re-entered on next startup.")
    
    return config

def get_api_config():
    """Get API configuration"""
    config = setup_api_config()
    
    gpt_config = {
        'api_key': config.get('GPT', 'api_key'),
        'api_url': config.get('GPT', 'api_url'),
        'headers': {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.get('GPT', 'api_key')}"
        }
    }
    
    qwen_config = {
        'api_key': config.get('QWEN', 'api_key'),
        'api_url': config.get('QWEN', 'api_url'),
        'headers': {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.get('QWEN', 'api_key')}"
        }
    }
    
    deepseek_config = {
        'api_key': config.get('DEEPSEEK', 'api_key'),
        'api_url': config.get('DEEPSEEK', 'api_url'),
        'headers': {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.get('DEEPSEEK', 'api_key')}"
        }
    }
    
    return gpt_config, qwen_config, deepseek_config


GPT_CONFIG, QWEN_CONFIG, DEEPSEEK_CONFIG = get_api_config()

GPT_API_KEY = GPT_CONFIG['api_key']
GPT_API_URL = GPT_CONFIG['api_url']
GPT_HEADERS = GPT_CONFIG['headers']

QWEN_API_KEY = QWEN_CONFIG['api_key']
QWEN_API_URL = QWEN_CONFIG['api_url']
QWEN_HEADERS = QWEN_CONFIG['headers']

DEEPSEEK_API_KEY = DEEPSEEK_CONFIG['api_key']
DEEPSEEK_API_URL = DEEPSEEK_CONFIG['api_url']
DEEPSEEK_HEADERS = DEEPSEEK_CONFIG['headers']


# Check if file exists
def check_file_exists(file_path):
    """Check if file exists"""
    if not os.path.exists(file_path):
        print("Error: File '{}' does not exist!".format(file_path))
        return False
    return True

# HealthBench data loading related constants
HEALTHBENCH_INPUT_PATH = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
HEALTHBENCH_INPUT_PATH_HARD = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
HEALTHBENCH_INPUT_PATH_CONSENSUS = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"

# HealthBench data structure class
class RubricItem:
    """HealthBench evaluation criterion item"""
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )

# Local HealthBench data file paths
LOCAL_HEALTHBENCH_DIR = "../benchmarks/HealthBench"
LOCAL_FILE_MAPPING = {
    None: "2025-05-07-06-14-12_oss_eval.jsonl",  # Main dataset
    "hard": "hard_2025-05-08-21-00-10.jsonl",   # Hard subset
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",  # Consensus subset
    "meta": "2025-05-07-06-14-12_oss_meta_eval.jsonl"  # Meta dataset
}

# Load HealthBench data
def load_healthbench_data(subset_name: Optional[str] = None, num_examples: Optional[int] = None, use_local: bool = True):
    """Load HealthBench dataset
    
    Args:
        subset_name: Subset name ("hard", "consensus", "meta" or None)
        num_examples: Limit number of samples to load
        use_local: Whether to prioritize local files
        
    Returns:
        List[Dict]: HealthBench data sample list
    """
    try:
        # Try local files first
        if use_local:
            local_file = LOCAL_FILE_MAPPING.get(subset_name)
            if local_file:
                local_path = os.path.join(LOCAL_HEALTHBENCH_DIR, local_file)
                if os.path.exists(local_path):
                    print(f"Loading local HealthBench dataset: {local_path}")
                    return _load_from_local_file(local_path, num_examples)
                else:
                    print(f"Local file does not exist: {local_path}")
                    print("Will attempt to download from network...")
        
        # If local file doesn't exist, try network download
        return _load_from_remote_url(subset_name, num_examples)
        
    except Exception as e:
        print(f"Error loading HealthBench dataset: {str(e)}")
        traceback.print_exc()
        raise

def _load_from_local_file(file_path: str, num_examples: Optional[int] = None):
    """Load data from local file"""
    try:
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        
        # Convert rubric format
        for example in examples:
            example["rubrics"] = [RubricItem.from_dict(d) for d in example["rubrics"]]
        
        # Limit sample count - select in order rather than random sampling
        if num_examples is not None and num_examples < len(examples):
            examples = examples[:num_examples]  # Select first num_examples samples
        
        print(f"Successfully loaded local HealthBench dataset with {len(examples)} questions")
        return examples
        
    except Exception as e:
        print(f"Error loading local file: {str(e)}")
        raise

def _load_from_remote_url(subset_name: Optional[str] = None, num_examples: Optional[int] = None):
    """Download data from remote URL (fallback option)"""
    try:
        # Select data path based on subset
        if subset_name == "hard":
            input_path = HEALTHBENCH_INPUT_PATH_HARD
        elif subset_name == "consensus":
            input_path = HEALTHBENCH_INPUT_PATH_CONSENSUS
        elif subset_name is None:
            input_path = HEALTHBENCH_INPUT_PATH
        else:
            raise ValueError(f"Invalid subset name: {subset_name}")
        
        print(f"Loading HealthBench dataset from network: {input_path}")
        
        # Download and parse data
        with bf.BlobFile(input_path, "rb") as f:
            examples = [json.loads(line) for line in f]
        
        # Convert rubric format
        for example in examples:
            example["rubrics"] = [RubricItem.from_dict(d) for d in example["rubrics"]]
        
        # Limit sample count - select in order rather than random sampling
        if num_examples is not None and num_examples < len(examples):
            examples = examples[:num_examples]  # Select first num_examples samples
        
        print(f"Successfully loaded network HealthBench dataset with {len(examples)} questions")
        return examples
        
    except Exception as e:
        print(f"Error loading dataset from network: {str(e)}")
        raise




# ===================== GPT Model Section =====================
def get_gpt_prompt(question):
    system_prompt = """You are a knowledgeable medical assistant helping someone with a health-related question. Your goal is to provide helpful, accurate responses that address their specific concerns in a natural, conversational manner.

Please ensure your responses are:
1. Helpful and appropriately detailed for the conversation context
2. Clear and easy to understand, using appropriate language for the audience
3. Naturally conversational, adapting to who you're speaking with
4. Proactive in asking relevant follow-up questions when more context would be helpful

Important guidelines:
- Respond naturally to the conversation flow rather than providing exhaustive medical information
- When appropriate, ask about relevant details that could help provide better guidance
- Maintain appropriate medical caution while being helpful and reassuring when suitable
- Adapt your response style and detail level to what the person actually needs

Health-related inquiry or conversation:
"""
    system_prompt += f"{question}"
    
    return system_prompt

# Use ChatGPT to generate answer
def generate_gpt_answer(question):
    """Use ChatGPT to generate answer
    
    Args:
        question: Medical question
        
    Returns:
        str: Model generated answer
    """
    try:
        prompt = get_gpt_prompt(question)
        
        print("GPT is generating answer...")
        t_generate_start = time.time()
        
        data = {
            "model": "o1-mini", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are GPT-o1, a medical expert with extensive knowledge in healthcare and medicine."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 8000,
        }

        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
        t_generate = time.time() - t_generate_start
        print(f"GPT API response status code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"GPT answer generation completed, time elapsed: {t_generate:.2f} seconds")
            print("\nGPT answer content:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            return answer
        else:
            print(f"GPT API error: {response.status_code}")
            print(f"Error details: {response.text}")
            error_message = f"Sorry, an error occurred while processing your question. Error code: {response.status_code}"
            print("\nGPT error information:")
            print("-" * 80)
            print(error_message)
            print("-" * 80)
            return error_message

    except Exception as e:
        print(f"Error occurred while generating GPT answer: {str(e)}")
        traceback.print_exc()
        error_message = f"Sorry, an error occurred while processing your question. Error message: {str(e)}"
        print("\nGPT error information:")
        print("-" * 80)
        print(error_message)
        print("-" * 80)
        return error_message



# ===================== Qwen Model Section =====================
def get_qwen_prompt(question):
    """Build Qwen medical Q&A prompt"""
    system_prompt = """You are a knowledgeable medical assistant helping someone with a health-related question. Your goal is to provide helpful, accurate responses that address their specific concerns in a natural, conversational manner.

Please ensure your responses are:
1. Helpful and appropriately detailed for the conversation context
2. Clear and easy to understand, using appropriate language for the audience
3. Naturally conversational, adapting to who you're speaking with
4. Proactive in asking relevant follow-up questions when more context would be helpful

Important guidelines:
- Respond naturally to the conversation flow rather than providing exhaustive medical information
- When appropriate, ask about relevant details that could help provide better guidance
- Maintain appropriate medical caution while being helpful and reassuring when suitable
- Adapt your response style and detail level to what the person actually needs

Health-related inquiry or conversation:
"""
    system_prompt += f"{question}"
    
    return system_prompt

# Use Qwen to generate answer
def generate_qwen_answer(question):
    """Use Qwen to generate answer
    
    Args:
        question: Medical question
        
    Returns:
        str: Model generated answer
    """
    try:
        prompt = get_qwen_prompt(question)
        
        print("Qwen is generating answer...")
        t_generate_start = time.time()
        
        # Build request data
        data = {
            "model": "Qwen/QwQ-32B", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are Qwen-QwQ, a medical expert with extensive knowledge in healthcare and medicine."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }

        response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data, timeout=300)
        t_generate = time.time() - t_generate_start
        
        if response.status_code == 200:
            response_data = response.json()
            
            if 'choices' in response_data and response_data['choices']:
                if 'message' in response_data['choices'][0]:
                    message = response_data['choices'][0]['message']
                    
                    # Extract content
                    if 'content' in message and message['content'] and len(message['content'].strip()) > 0:
                        answer = message['content'].strip()
                    elif 'reasoning_content' in message and message['reasoning_content']:
                        answer = message['reasoning_content'].strip()
                    else:
                        print("Warning: API returned empty content")
                        answer = "API returned empty content"
                    
                    print(f"Qwen answer generation completed, time elapsed: {t_generate:.2f} seconds")
                    print("\nQwen answer content:")
                    print("-" * 80)
                    print(answer)
                    print("-" * 80)
                    return answer
                else:
                    print("Error: message field does not exist")
                    error_message = "API response structure error: missing message field"
                    print("\nQwen error information:")
                    print("-" * 80)
                    print(error_message)
                    print("-" * 80)
                    return error_message
            else:
                print("Error: choices field does not exist or is empty")
                error_message = "API response structure error: missing choices field"
                print("\nQwen error information:")
                print("-" * 80)
                print(error_message)
                print("-" * 80)
                return error_message
        else:
            print(f"Qwen API error: {response.status_code}")
            print(f"Error details: {response.text}")
            error_message = f"Sorry, an error occurred while processing your question. Error code: {response.status_code}"
            print("\nQwen error information:")
            print("-" * 80)
            print(error_message)
            print("-" * 80)
            return error_message

    except Exception as e:
        print(f"Error occurred while generating Qwen answer: {str(e)}")
        traceback.print_exc()
        error_message = f"Sorry, an error occurred while processing your question. Error message: {str(e)}"
        print("\nQwen error information:")
        print("-" * 80)
        print(error_message)
        print("-" * 80)
        return error_message


# ===================== DeepSeek Model Section =====================
def get_deepseek_prompt(question):
    system_prompt = """You are a knowledgeable medical assistant helping someone with a health-related question. Your goal is to provide helpful, accurate responses that address their specific concerns in a natural, conversational manner.

Please ensure your responses are:
1. Helpful and appropriately detailed for the conversation context
2. Clear and easy to understand, using appropriate language for the audience
3. Naturally conversational, adapting to who you're speaking with
4. Proactive in asking relevant follow-up questions when more context would be helpful

Important guidelines:
- Respond naturally to the conversation flow rather than providing exhaustive medical information
- When appropriate, ask about relevant details that could help provide better guidance
- Maintain appropriate medical caution while being helpful and reassuring when suitable
- Adapt your response style and detail level to what the person actually needs

Health-related inquiry or conversation:
"""
    system_prompt += f"{question}"
    
    return system_prompt

# Use DeepSeek to generate answer
def generate_deepseek_answer(question):
    """Use DeepSeek to generate answer
    
    Args:
        question: Medical question
        
    Returns:
        str: Model generated answer
    """
    try:
        prompt = get_deepseek_prompt(question)
        
        print("DeepSeek is generating answer...")
        t_generate_start = time.time()
        
        # Try using OpenAI client for API call
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model="deepseek-reasoner", # Note, use the same as the original study; you can replace it with other LLMs. 
                messages=[
                    {"role": "system", "content": "You are DeepSeek-R1, a medical expert with extensive knowledge in healthcare and medicine."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )
            
            answer = response.choices[0].message.content
            t_generate = time.time() - t_generate_start
            print(f"DeepSeek answer generation completed, time elapsed: {t_generate:.2f} seconds")
            print("\nDeepSeek answer content:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            return answer
            
        except Exception as e:
            print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
            print("Trying to call API directly using requests...")
            
            # Fallback: directly use requests to call SiliconFlow API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1", # Note, use the same as the original study; you can replace it with other LLMs. 
                "messages": [
                    {"role": "system", "content": "You are DeepSeek-R1, a medical expert with extensive knowledge in healthcare and medicine."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 8000
            }
            url = "https://api.siliconflow.cn/v1/chat/completions" # SiliconFlow URL
            headers = {
                "Authorization": "Bearer sk-egbetwgfnaopvplrtpenocsbhsmferlbiyggubouibdpwulm",
                "Content-Type": "application/json"
            }
    
            response = requests.post(url, headers=headers, json=data)
            t_generate = time.time() - t_generate_start
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['choices'][0]['message']['content'].strip()
                print(f"DeepSeek answer generation completed, time elapsed: {t_generate:.2f} seconds")
                print("\nDeepSeek answer content:")
                print("-" * 80)
                print(answer)
                print("-" * 80)
                return answer
            else:
                print(f"DeepSeek API error: {response.status_code}")
                print(f"Error details: {response.text}")
                error_message = f"Sorry, an error occurred while processing your question. Error code: {response.status_code}"
                print("\nDeepSeek error information:")
                print("-" * 80)
                print(error_message)
                print("-" * 80)
                return error_message
        
    except Exception as e:
        print(f"Error occurred while generating DeepSeek answer: {str(e)}")
        traceback.print_exc()
        error_message = f"Sorry, an error occurred while processing your question. Error message: {str(e)}"
        print("\nDeepSeek error information:")
        print("-" * 80)
        print(error_message)
        print("-" * 80)
        return error_message




# ===================== Consistency Evaluation Section =====================
def structured_model_evaluation(question, model_name, answer_to_evaluate, evaluator_model_name, evaluator_previous_answer, max_retries=3):
    """Conduct structured multi-dimensional evaluation of model answers
    
    Args:
        question: Original question
        model_name: Name of model being evaluated
        answer_to_evaluate: Answer content to be evaluated
        evaluator_model_name: Name of evaluator model
        evaluator_previous_answer: Evaluator's own answer
        max_retries: Maximum retry attempts when API fails
        
    Returns:
        Dict: Contains scores and evaluations for three dimensions
    """
    
    # Implement 3-retry mechanism
    for attempt in range(max_retries):
        try:
            print(f"Attempting evaluation {evaluator_model_name}→{model_name} (attempt {attempt+1})")
            
            # Three core dimensions from the MCC study
            evaluation_dimensions = [
                {"name": "factual_accuracy", "description": "Evaluate the objective correctness of medical information. Check for incorrect medical information, inaccurate dosages/numbers, outdated medical practices, misused medical terminology, and ensure alignment with current medical consensus and guidelines. Focus on factual accuracy without considering completeness."},
                {"name": "completeness", "description": "Evaluate the comprehensiveness and communication effectiveness of the response. Check whether all necessary information points of the question are covered without key omissions; whether the language is clear, empathetic, well-structured, and easy to understand; whether it accurately understands the dialogue context and user intent, and appropriately seeks additional background information when needed."},
                {"name": "safety", "description": "Evaluate the safety of medical advice and compliance with guidelines. Check for harmful medical advice, assess whether urgency level is appropriate for the described situation, ensure recommendations are proportionate to the described symptoms, avoid unnecessary alarmism while maintaining appropriate caution; also evaluate whether the response strictly follows the user's specific instructions or role-playing requirements."}
            ]

            # Build evaluation prompt
            dimensions_text = ""
            for i, dim in enumerate(evaluation_dimensions, 1):
                dimensions_text += f"{i}. {dim['name'].upper()}: {dim['description']}\n"
            
            prompt = f"""As {evaluator_model_name}, you need to conduct a focused evaluation of {model_name}'s answer to a medical question. Rate the answer on three key dimensions:

{dimensions_text}

Original question:
{question}

{model_name}'s answer:
{answer_to_evaluate}

Your own answer:
{evaluator_previous_answer}

For each dimension, provide:
1. A score from 1-10 (1 being lowest, 10 being highest)
2. Reasoning for your score (concise, focusing on key strengths or issues)
3. Brief suggestions for improvement

Be objective and fair in your evaluation. Keep your assessment concise and focused.

Response format:
FACTUAL_ACCURACY_SCORE: [score 1-10]
FACTUAL_ACCURACY_REASON: [brief reasoning]
FACTUAL_ACCURACY_SUGGESTION: [brief suggestion]

COMPLETENESS_SCORE: [score 1-10]
COMPLETENESS_REASON: [brief reasoning]
COMPLETENESS_SUGGESTION: [brief suggestion]

SAFETY_SCORE: [score 1-10]
SAFETY_REASON: [brief reasoning]
SAFETY_SUGGESTION: [brief suggestion]

COMPARATIVE_ANALYSIS: [very brief comparison with your own answer]
"""
            
            # Select corresponding API based on evaluator model
            if evaluator_model_name == "GPT":
                data = {
                    "model": "o1-mini",
                    "messages": [
                        {"role": "system", "content": "You are a medical evaluation expert. Provide concise, accurate evaluations focusing only on the most important aspects. Always respond in English."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_completion_tokens": 10000,
                }
                response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data, timeout=120)
                
                if response.status_code == 200:
                    response_data = response.json()
                    evaluation_text = response_data['choices'][0]['message']['content'].strip()
                else:
                    print(f"GPT evaluation API error: {response.status_code}")
                    raise Exception(f"GPT evaluation API call failed: {response.status_code}")
                    
            elif evaluator_model_name == "Qwen":
                data = {
                    "model": "Qwen/QwQ-32B", 
                    "messages": [
                        {"role": "system", "content": "You are a medical evaluation expert. Provide concise, accurate evaluations focusing only on the most important aspects. Always respond in English."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 10000
                }
                response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data, timeout=300)
                
                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and response_data['choices']:
                        if 'message' in response_data['choices'][0]:
                            message = response_data['choices'][0]['message']
                            evaluation_text = message['content'].strip() if 'content' in message else ""
                        else:
                            raise Exception("Qwen API response format error: missing message field")
                    else:
                        raise Exception("Qwen API response format error: missing choices field")
                else:
                    print(f"Qwen evaluation API error: {response.status_code}")
                    raise Exception(f"Qwen evaluation API call failed: {response.status_code}")
                    
            elif evaluator_model_name == "DeepSeek":
                try:
                    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
                    
                    response = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": "You are a medical evaluation expert. Provide concise, accurate evaluations focusing only on the most important aspects. Always respond in English."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=10000
                    )
                    
                    evaluation_text = response.choices[0].message.content
                    
                except Exception as e:
                    print(f"DeepSeek OpenAI client failed: {str(e)}")
                    # Try fallback API
                    data = {
                        "model": "Pro/deepseek-ai/DeepSeek-R1",
                        "messages": [
                            {"role": "system", "content": "You are a medical evaluation expert. Provide concise, accurate evaluations focusing only on the most important aspects. Always respond in English."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 10000
                    }
                    url = "https://api.siliconflow.cn/v1/chat/completions"
                    headers = {
                        "Authorization": "Bearer sk-egbetwgfnaopvplrtpenocsbhsmferlbiyggubouibdpwulm",
                        "Content-Type": "application/json"
                    }
                    
                    response = requests.post(url, headers=headers, json=data, timeout=300)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        evaluation_text = response_data['choices'][0]['message']['content'].strip()
                    else:
                        print(f"DeepSeek fallback API error: {response.status_code}")
                        raise Exception(f"All DeepSeek APIs failed")
            else:
                raise Exception(f"Unknown evaluator model: {evaluator_model_name}")
            
            # Check evaluation text validity
            if not evaluation_text or len(evaluation_text) < 50:
                raise Exception(f"{evaluator_model_name} returned invalid or too short evaluation text")
                
            # Print original evaluation text
            print(f"\n{evaluator_model_name} evaluation results for {model_name}:")
            print(f"{'-'*40}")
            print(evaluation_text)
            print(f"{'-'*40}")
            
            # Parse evaluation results
            evaluation_result = parse_evaluation_text(evaluation_text)
            
            # Print parsed evaluation scores
            print(f"\n{evaluator_model_name}'s dimension scores for {model_name}:")
            for dim in ["factual_accuracy", "completeness", "safety"]:
                if dim in evaluation_result:
                    print(f"- {dim}: {evaluation_result[dim]['score']}/10")
            
            print(f"{evaluator_model_name} completed evaluation of {model_name} results")
            print(f"{'-'*80}\n")
            return evaluation_result
            
        except Exception as e:
            print(f"Evaluation attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting 2 seconds before attempt {attempt+2}...")
                time.sleep(2)
            else:
                print(f"⚠️ Critical error: {evaluator_model_name} evaluation of {model_name} still failed after {max_retries} attempts!")
                traceback.print_exc()
                raise Exception(f"Evaluation completely failed: {evaluator_model_name}→{model_name} - retried {max_retries} times")



def parse_evaluation_text(evaluation_text):
    """Parse evaluation text and extract scores and evaluations for each dimension"""
    try:
        evaluation_result = {}
        
        # Extract scores and evaluations from each dimension - improved regex to handle various markdown formats
        # Handle possible markdown formats like **FACTUAL_ACCURACY_SCORE:** or FACTUAL_ACCURACY_SCORE:
        factual_accuracy_score = re.search(r'FACTUAL_ACCURACY_SCORE[:\*\s]*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
        factual_accuracy_reason = re.search(r'FACTUAL_ACCURACY_REASON[:\*\s]*(.*?)(?=FACTUAL_ACCURACY_SUGGESTION|COMPLETENESS_SCORE|$)', evaluation_text, re.DOTALL | re.IGNORECASE)
        factual_accuracy_suggestion = re.search(r'FACTUAL_ACCURACY_SUGGESTION[:\*\s]*(.*?)(?=COMPLETENESS_SCORE|$)', evaluation_text, re.DOTALL | re.IGNORECASE)
        
        completeness_score = re.search(r'COMPLETENESS_SCORE[:\*\s]*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
        completeness_reason = re.search(r'COMPLETENESS_REASON[:\*\s]*(.*?)(?=COMPLETENESS_SUGGESTION|SAFETY_SCORE|$)', evaluation_text, re.DOTALL | re.IGNORECASE)
        completeness_suggestion = re.search(r'COMPLETENESS_SUGGESTION[:\*\s]*(.*?)(?=SAFETY_SCORE|$)', evaluation_text, re.DOTALL | re.IGNORECASE)
        
        safety_score = re.search(r'SAFETY_SCORE[:\*\s]*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
        safety_reason = re.search(r'SAFETY_REASON[:\*\s]*(.*?)(?=SAFETY_SUGGESTION|BIAS_FAIRNESS_SCORE|COMPARATIVE_ANALYSIS|$)', evaluation_text, re.DOTALL | re.IGNORECASE)
        safety_suggestion = re.search(r'SAFETY_SUGGESTION[:\*\s]*(.*?)(?=BIAS_FAIRNESS_SCORE|COMPARATIVE_ANALYSIS|$)', evaluation_text, re.DOTALL | re.IGNORECASE)
        
        comparative_analysis = re.search(r'COMPARATIVE_ANALYSIS:\s*(.*?)($)', evaluation_text, re.DOTALL)
        
        # Build evaluation result dictionary - throw exception if parsing fails
        if not factual_accuracy_score:
            print(f"Parsing error: Unable to extract FACTUAL_ACCURACY_SCORE")
            print(f"First 500 characters of evaluation text: {evaluation_text[:500]}")
            raise Exception("FACTUAL_ACCURACY_SCORE parsing failed")
        
        if not completeness_score:
            print(f"Parsing error: Unable to extract COMPLETENESS_SCORE")
            print(f"First 500 characters of evaluation text: {evaluation_text[:500]}")
            raise Exception("COMPLETENESS_SCORE parsing failed")
            
        if not safety_score:
            print(f"Parsing error: Unable to extract SAFETY_SCORE")
            print(f"First 500 characters of evaluation text: {evaluation_text[:500]}")
            raise Exception("SAFETY_SCORE parsing failed")
        
        evaluation_result["factual_accuracy"] = {
            "score": float(factual_accuracy_score.group(1)),
            "reason": factual_accuracy_reason.group(1).strip() if factual_accuracy_reason else "No reason provided",
            "suggestion": factual_accuracy_suggestion.group(1).strip() if factual_accuracy_suggestion else "No suggestion provided"
        }
        
        evaluation_result["completeness"] = {
            "score": float(completeness_score.group(1)),
            "reason": completeness_reason.group(1).strip() if completeness_reason else "No reason provided",
            "suggestion": completeness_suggestion.group(1).strip() if completeness_suggestion else "No suggestion provided"
        }
        
        evaluation_result["safety"] = {
            "score": float(safety_score.group(1)),
            "reason": safety_reason.group(1).strip() if safety_reason else "No reason provided",
            "suggestion": safety_suggestion.group(1).strip() if safety_suggestion else "No suggestion provided"
        }
        
        evaluation_result["comparative_analysis"] = comparative_analysis.group(1).strip() if comparative_analysis else "No comparative analysis provided"
        
        # Calculate total score
        total_score = (
            evaluation_result["factual_accuracy"]["score"] +
            evaluation_result["completeness"]["score"] +
            evaluation_result["safety"]["score"]
        ) / 3.0
        
        evaluation_result["total_score"] = total_score
        
        return evaluation_result
        
    except Exception as e:
        print(f"Error parsing evaluation text: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Evaluation text parsing failed: {str(e)}")




def model_improvement(question, model_name, current_answer, evaluations_received):
    """Improve model answer based on received evaluations
    
    Args:
        question: Original question
        model_name: Model name
        current_answer: Current answer
        evaluations_received: Received evaluation results
        
    Returns:
        Dict: Contains critique summary and improved answer
    """
    try:   
        # Summarize evaluations from all models
        evaluation_summary = ""
        for evaluator, evaluation in evaluations_received.items():
            evaluation_summary += f"\n--- Evaluation from {evaluator} for {model_name} ---\n"
            
            # Add scores and evaluations for each dimension
            for dimension in ["factual_accuracy", "completeness", "safety"]:
                if dimension in evaluation:
                    dim_data = evaluation[dimension]
                    evaluation_summary += f"{dimension.upper()} Score: {dim_data['score']}/10\n"
                    evaluation_summary += f"Reason: {dim_data['reason']}\n"
                    evaluation_summary += f"Suggestion: {dim_data['suggestion']}\n\n"
            
            # Add comparative analysis
            if "comparative_analysis" in evaluation:
                evaluation_summary += f"Comparative Analysis: {evaluation['comparative_analysis']}\n"
            
        # Build prompt
        prompt = f"""As {model_name}, you need to improve your answer to a medical question based on the evaluation feedback from other models.

Original question:
{question}

Your current answer:
{current_answer}

Evaluation feedback from other models:
{evaluation_summary}

Please follow these guidelines to improve your answer:
1. Provide a brief summary of the criticisms, indicating which points you agree with and which you disagree with
2. Improve your answer by addressing the issues pointed out in the evaluations
3. Maintain the accurate and valuable parts of your original answer
4. Add important information mentioned by other models that you might have missed
5. Correct any inaccuracies or potentially risky content

First provide a critique summary, then provide your complete improved answer. Use this format:

CRITIQUE_SUMMARY: [Brief summary of evaluation feedback, including points you agree and disagree with]

IMPROVED_ANSWER: [Your complete improved answer]
"""
        
        # Select corresponding API based on model
        if model_name == "GPT":
            data = {
                "model": "o1-mini",
                "messages": [
                    {"role": "system", "content": "You are GPT-o1, a medical expert who is improving your answer based on feedback."},
                    {"role": "user", "content": prompt}
                ],
                "max_completion_tokens": 10000,
            }
            response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                improvement_text = response_data['choices'][0]['message']['content'].strip()
            else:
                print(f"Improvement API error: {response.status_code}")
                return {"critique": "Could not generate critique summary", "improved_answer": current_answer}
                
        elif model_name == "Qwen":
            data = {
                "model": "Qwen/QwQ-32B", 
                "messages": [
                    {"role": "system", "content": "You are Qwen-QwQ, a medical expert who is improving your answer based on feedback."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10000
            }
            response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data, timeout=300)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    if 'message' in response_data['choices'][0]:
                        message = response_data['choices'][0]['message']
                        improvement_text = message['content'].strip() if 'content' in message else "Improvement failed"
                    else:
                        return {"critique": "Could not generate critique summary", "improved_answer": current_answer}
                else:
                    return {"critique": "Could not generate critique summary", "improved_answer": current_answer}
            else:
                print(f"Improvement API error: {response.status_code}")
                return {"critique": "Could not generate critique summary", "improved_answer": current_answer}
                
        elif model_name == "DeepSeek":
            try:
                client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
                
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are DeepSeek-R1, a medical expert who is improving your answer based on feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10000
                )
                
                improvement_text = response.choices[0].message.content
                
            except Exception as e:
                print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
                # Try fallback API
                data = {
                    "model": "Pro/deepseek-ai/DeepSeek-R1",
                    "messages": [
                        {"role": "system", "content": "You are DeepSeek-R1, a medical expert who is improving your answer based on feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 10000
                }
                url = "https://api.siliconflow.cn/v1/chat/completions"
                headers = {
                    "Authorization": "Bearer sk-egbetwgfnaopvplrtpenocsbhsmferlbiyggubouibdpwulm",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    improvement_text = response_data['choices'][0]['message']['content'].strip()
                else:
                    print(f"Improvement API error: {response.status_code}")
                    return {"critique": "Could not generate critique summary", "improved_answer": current_answer}
        else:
            print(f"Unknown model: {model_name}")
            return {"critique": "Could not generate critique summary", "improved_answer": current_answer}
        
        # Print model improvement results
        print(f"\n{model_name} optimized output:")
        print(f"{'-'*40}")
        print(improvement_text)
        print(f"{'-'*40}")
        
        # Parse improvement results
        critique_match = re.search(r'CRITIQUE_SUMMARY:\s*(.*?)(?=IMPROVED_ANSWER:|$)', improvement_text, re.DOTALL)
        improved_answer_match = re.search(r'IMPROVED_ANSWER:\s*(.*?)($)', improvement_text, re.DOTALL)
        
        critique = critique_match.group(1).strip() if critique_match else "No critique summary provided"
        improved_answer = improved_answer_match.group(1).strip() if improved_answer_match else current_answer
            
        print(f"{model_name} answer optimization completed")
        print(f"{'-'*100}\n")
        return {"critique": critique, "improved_answer": improved_answer}
        
    except Exception as e:
        print(f"Error while improving answer: {str(e)}")
        traceback.print_exc()
        return {"critique": "Error during improvement process", "improved_answer": current_answer}



def calculate_consensus_metrics(evaluations_history):
    """Calculate consensus metrics between models
    
    Args:
        evaluations_history: Evaluation history records
        
    Returns:
        Dict: Dictionary containing consensus metrics
    """
    # If no evaluation history, return default values
    if not evaluations_history or len(evaluations_history) == 0:
        return {
            "consensus_score": 0.0,
            "requires_further_debate": True,
            "dimension_agreement": {}
        }
    
    # Get latest round of evaluations
    latest_round = evaluations_history[-1]
    
    # Calculate score differences for each dimension
    dimension_agreement = {}
    dimensions = ["factual_accuracy", "completeness", "safety"] # Three core dimensions from the MCC study
    
    for dimension in dimensions:
        scores = []
        for model, evals in latest_round.items():
            for target, eval_data in evals.items():
                if isinstance(eval_data, dict):
                    if dimension in eval_data and isinstance(eval_data[dimension], dict) and "score" in eval_data[dimension]:
                        score = eval_data[dimension]["score"]
                        # Handle possible decimal scores
                        if isinstance(score, (int, float)):
                            scores.append(float(score))
        
        # Calculate score standard deviation as consistency indicator
        if scores:
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            std_dev = variance ** 0.5
            
            # Smaller standard deviation means higher consistency
            agreement_score = max(0, 1 - (std_dev / 4))  
            dimension_agreement[dimension] = agreement_score
        else:
            print(f"Warning: {dimension} dimension has no valid evaluation scores")
            dimension_agreement[dimension] = 0.0
    
    # Calculate overall consensus score
    if dimension_agreement:
        consensus_score = sum(dimension_agreement.values()) / len(dimension_agreement)
    else:
        consensus_score = 0.0
    
    # Determine if further debate is needed
    requires_further_debate = consensus_score < 0.8 # The criteria of MCC study
    
    return {
        "consensus_score": consensus_score,
        "requires_further_debate": requires_further_debate,
        "dimension_agreement": dimension_agreement
    }


# ===================== Model Debate Section =====================
def circular_criticism_improvement(question, initial_answers, max_rounds=3):
    """Use circular criticism-improvement mode for model debate
    
    Args:
        question: Original medical question
        initial_answers: Initial answers from three models
        max_rounds: Maximum debate rounds
        
    Returns:
        Dict: Debate results, including final answer and debate history
    """
    print("="*50)
    print("Starting medical Q&A debate (Circular Criticism-Improvement Mode)")
    print("="*80)
    
    # Model list
    models = ["GPT", "Qwen", "DeepSeek"]
    
    # Current answer versions
    current_answers = initial_answers.copy()
    
    # Store debate history
    debate_history = [{
        "round": 0,
        "answers": current_answers.copy(),
        "evaluations": {},
        "improvements": {}
    }]
    
    # Store evaluation history
    evaluations_history = []
    
    # Start debate loop
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*40} Debate Round {round_num} {'='*40}")
        
        # Step 1: Each model evaluates other models' answers (parallel)
        print(f"\nRound {round_num} - Phase 1: Model confrontation phase begins")
        round_evaluations = {model: {} for model in models}
        
        # Create evaluation task list
        evaluation_tasks = []
        for evaluator in models:
            for target in models:
                if evaluator != target:  # Don't evaluate self
                    evaluation_tasks.append((evaluator, target))
        
        # Execute evaluation tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(evaluation_tasks)) as executor:
            # Submit all evaluation tasks
            future_to_task = {
                executor.submit(
                    structured_model_evaluation,
                    question, 
                    target, 
                    current_answers[target],
                    evaluator,
                    current_answers[evaluator]
                ): (evaluator, target) for evaluator, target in evaluation_tasks
            }
            
            # Collect evaluation results
            for future in concurrent.futures.as_completed(future_to_task):
                evaluator, target = future_to_task[future]
                try:
                    evaluation = future.result()
                    round_evaluations[evaluator][target] = evaluation
                    #print(f"{evaluator}'s evaluation of {target} completed")
                except Exception as e:
                    print(f"❌ Critical error: {evaluator} evaluation of {target} completely failed")
                    print(f"Error details: {str(e)}")
                    traceback.print_exc()
                    print(f"🛑 Stop evaluation process - cannot use fake scores")
                    raise Exception(f"Critical evaluation failure: {evaluator} unable to evaluate {target} - failed after 3 retries")
        
        # Store current round evaluations
        evaluations_history.append(round_evaluations)
        
        # Calculate current consensus metrics
        consensus_metrics = calculate_consensus_metrics(evaluations_history)
        print(f"\nCurrent consensus status:")
        print(f"- Overall consensus score: {consensus_metrics['consensus_score']:.2f}")
        print("- Dimension consensus scores:")
        for dim, score in consensus_metrics.get("dimension_agreement", {}).items():
            print(f"  * {dim}: {score:.2f}")
        
        # If sufficient consensus is reached, can end debate early
        if not consensus_metrics["requires_further_debate"] and round_num > 1:
            print("\nModels have reached sufficient consensus, ending debate early")
            # Before ending early, save current round evaluation data
            debate_history.append({
                "round": round_num,
                "answers": current_answers.copy(),
                "evaluations": round_evaluations,
                "improvements": {},  # No improvement phase when ending early
                "consensus_metrics": consensus_metrics
            })
            break
        
        # Step 2: Each model improves its own answer based on evaluation results (parallel)
        print(f"\nRound {round_num} - Phase 2: Models improve their answers based on evaluation results")
        round_improvements = {}
        improved_answers = current_answers.copy()
        
        # Execute improvement tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
            # Prepare evaluation data for each model
            model_to_evaluations = {model: {} for model in models}
            for evaluator in models:
                for target in models:
                    if evaluator != target and target in round_evaluations.get(evaluator, {}):
                        if target not in model_to_evaluations:
                            model_to_evaluations[target] = {}
                        model_to_evaluations[target][evaluator] = round_evaluations[evaluator][target]
            
            # Submit all improvement tasks
            future_to_model = {
                executor.submit(
                    model_improvement,
                    question,
                    model,
                    current_answers[model],
                    model_to_evaluations[model]
                ): model for model in models
            }
            
            # Collect improvement results
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    improvement_result = future.result()
                    round_improvements[model] = improvement_result
                    improved_answers[model] = improvement_result["improved_answer"]
                    #print(f"{model}'s answer improvement completed")
                except Exception as e:
                    print(f"Error in {model}'s answer improvement: {str(e)}")
                    round_improvements[model] = {"critique": f"Error: {str(e)}", "improved_answer": current_answers[model]}
                    improved_answers[model] = current_answers[model]
        
        # Update current answers
        current_answers = improved_answers.copy()
        
        # Record current round results
        debate_history.append({
            "round": round_num,
            "answers": current_answers.copy(),
            "evaluations": round_evaluations,
            "improvements": round_improvements,
            "consensus_metrics": consensus_metrics
        })
        
        print(f"\nRound {round_num} debate completed")
        print(f"{'='*50}")
    
    # Final consensus evaluation
    final_consensus_metrics = calculate_consensus_metrics(evaluations_history)
    print("\nDebate finished, final consensus status:")
    print(f"- Overall consensus score: {final_consensus_metrics['consensus_score']:.2f}")
    print("- Dimension consensus scores:")
    for dim, score in final_consensus_metrics.get("dimension_agreement", {}).items():
        print(f"  * {dim}: {score:.2f}")
    
    # Use expert integration model to generate final answer
    final_answer = consensus_driven_integration(
        question,
        current_answers,
        debate_history,
        final_consensus_metrics
    )
    
    return {
        "consensus_score": final_consensus_metrics["consensus_score"],
        "requires_further_debate": final_consensus_metrics["requires_further_debate"],
        "final_answer": final_answer,
        "debate_history": debate_history
    }


# ===================== Answer Integration Section =====================
def consensus_driven_integration(question, final_answers, debate_history, consensus_metrics):
    """Generate final consensus answer based on debate results
    
    Args:
        question: Original question
        final_answers: Final answers from each model
        debate_history: Debate history
        consensus_metrics: Consensus metrics
        
    Returns:
        str: Integrated final answer
    """
    try:
        print(f"\n{'-'*80}")
        print("\nIntegrating expert final answer...")
        
        # Extract only the last round of debate information (most relevant and recent)
        debate_summary = ""
        if debate_history:
            # Get last round debate data
            last_round = None
            for round_data in reversed(debate_history):
                if round_data["round"] > 0:  # Skip initial answer round
                    last_round = round_data
                    break
            
            if last_round:
                round_num = last_round["round"]
                debate_summary += f"Final round ({round_num}) consensus summary:\n"
                
                # Add last round evaluation details
                if "evaluations" in last_round:
                    debate_summary += "Final round evaluations:\n"
                    for evaluator, targets in last_round["evaluations"].items():
                        for target, evaluation in targets.items():
                            if isinstance(evaluation, dict):
                                fa_score = evaluation.get("factual_accuracy", {}).get("score", "N/A")
                                comp_score = evaluation.get("completeness", {}).get("score", "N/A") 
                                safety_score = evaluation.get("safety", {}).get("score", "N/A")
                                debate_summary += f"  {evaluator}→{target}: FA={fa_score}, Comp={comp_score}, Safety={safety_score}\n"
                
                # Add last round model improvement information
                if "improvements" in last_round:
                    debate_summary += "Final round improvements:\n"
                    for model, improvement in last_round["improvements"].items():
                        if "critique" in improvement:
                            debate_summary += f"  {model} final critique: {improvement['critique']}\n"
                
                # Add final consensus information
                if "consensus_metrics" in last_round:
                    consensus = last_round["consensus_metrics"]
                    debate_summary += f"  Final consensus score: {consensus.get('consensus_score', 'N/A'):.3f}\n"
                
                debate_summary += "\n"
        
        # Print debate summary
        print("\nLast round debate records:")
        print(f"{'-'*40}")
        print(debate_summary)
        print(f"{'-'*40}")
        
        # Build dimension score information
        dimension_scores = ""
        if "dimension_agreement" in consensus_metrics:
            dimension_scores += "Dimension consensus scores:\n"
            for dimension, score in consensus_metrics["dimension_agreement"].items():
                dimension_scores += f"- {dimension}: {score:.2f}\n"
        
        # Check if it's a multi-turn conversation, if so extract original conversation
        if "You are participating in a multi-turn conversation" in question:
            # Extract original conversation part
            conversation_start = question.find("conversation history below:") + len("conversation history below:")
            conversation_end = question.find("\n\nPlease provide your response")
            if conversation_start > 0 and conversation_end > 0:
                original_conversation = question[conversation_start:conversation_end].strip()
                question_display = f"Multi-turn conversation:\n{original_conversation}"
            else:
                question_display = question
        else:
            question_display = question

        prompt = f"""You are helping someone with a health-related question. You have three high-quality reference responses from medical AI models that have already discussed and refined their answers.

Your task: Create the most helpful response by thoughtfully combining the best insights from all three models.

Original question/conversation:
{question_display}

Reference responses:

GPT's response:
{final_answers["GPT"]}

Qwen's response:
{final_answers["Qwen"]}

DeepSeek's response:
{final_answers["DeepSeek"]}

Discussion summary:
{debate_summary}

INTEGRATION GUIDELINES:
- Combine the most helpful and relevant parts from all three responses
- Keep the response natural and conversational, not academic
- Match the response length to what this person actually needs
- Remove redundancies while preserving important information

CRITICAL REQUIREMENTS:
- If any model asked relevant follow-up questions, you MUST include appropriate questions in your response
- Maintain the same level of medical caution as the reference responses
- Only include medical information that appears in at least one reference response
- Do not add new medical facts, numbers, or recommendations

RESPONSE STYLE:
- Write as if speaking naturally to the person asking the question
- Use clear, accessible language appropriate for the audience
- Be helpful and reassuring while maintaining appropriate medical caution

IMPORTANT: Focus on being helpful and conversational while preserving the key medical guidance and any important follow-up questions from the reference responses.
"""
        
        data = {
            "model": "o1-mini",
            "messages": [
                {"role": "system", "content": "You are a medical professional who specializes in patient communication. Your role is to create the most helpful and appropriate response for patients by thoughtfully integrating insights from multiple medical AI consultations."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 32000,
        }
        
        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            integrated_answer = response_data['choices'][0]['message']['content'].strip()
            
            print("Final integrated answer generation complete")
            print(f"{'-'*80}\n")
            return integrated_answer
        else:
            print(f"Integration answer API error: {response.status_code}")
            # If API error occurs, use DeepSeek's answer as fallback
            print("Using DeepSeek's answer as fallback")
            return final_answers["DeepSeek"]
            
    except Exception as e:
        print(f"Error while generating integrated answer: {str(e)}")
        traceback.print_exc()
        # Use DeepSeek's answer as fallback when error occurs
        print("Using DeepSeek's answer as fallback")
        return final_answers["DeepSeek"]


# ===================== HealthBench Adapter =====================

# Official HealthBench scorer (strictly follows Paper: HealthBench: Evaluating Large Language Models Towards Improved Human Health)
def call_gpt_api(messages, model="gpt-4.1", temperature=0.0, max_tokens=32000):
    """Call GPT API - using MCC framework configuration"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"GPT API call failed: {e}")
        return None

GRADER_TEMPLATE_OFFICIAL = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
Note: Ignore any bracketed model names like [GPT, Qwen, DeepSeek] - these are source indicators for research purposes, not medical references.
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

def parse_json_to_dict_official(json_string: str) -> dict:
    """Parse JSON string - official implementation"""
    import re
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return {}

def grade_rubric_item_official(response_text: str, rubric_item: Dict, prompt: List[Dict]) -> Dict:
    """Use official method to evaluate single rubric item - fixed version"""
    # Build conversation string - completely following official method
    convo_with_response = prompt + [{"content": response_text, "role": "assistant"}]
    convo_str = "\\n\\n".join([f"{m['role']}: {m['content']}" for m in convo_with_response])
    
    # Build rubric_item string
    rubric_str = f"[{rubric_item['points']}] {rubric_item['criterion']}"
    
    # Use official GRADER_TEMPLATE
    grader_prompt = GRADER_TEMPLATE_OFFICIAL.replace(
        "<<conversation>>", convo_str
    ).replace("<<rubric_item>>", rubric_str)
    
    # Completely follow official implementation - use official parameters and logic
    messages = [{"role": "user", "content": grader_prompt}]  # Official only uses user role
    
    # Official infinite retry logic
    while True:
        try:
            response = call_gpt_api(
                [{"role": "system", "content": "You are a helpful assistant."}] + messages,
                model="gpt-4.1",  # Use model consistent with official
                temperature=0.0, 
                max_tokens=32000
            )
            
            if response is None:
                print("API call returned None, retrying...")
                time.sleep(2)
                continue
                
            grading_response_dict = parse_json_to_dict_official(response)
            
            if "criteria_met" in grading_response_dict:
                label = grading_response_dict["criteria_met"]
                if label is True or label is False:
                    return grading_response_dict
            
            print("Grading failed due to bad JSON output, retrying...")
            time.sleep(1)
            
        except Exception as e:
            print(f"Evaluation API call failed: {e}")
            time.sleep(2)

def grade_sample_official(prompt: List[Dict], response_text: str, rubric_items: List[Dict]) -> tuple[float, List[Dict]]:
    """Official sample scoring method"""
    print(f"Starting evaluation of {len(rubric_items)} rubric items...")
    
    grading_response_list = []
    
    for i, rubric_item in enumerate(rubric_items):
        criterion = rubric_item["criterion"]
        points = rubric_item["points"]
        
        print(f"\\n[{i+1}/{len(rubric_items)}] {criterion}")  # Show complete criteria, no abbreviation
        
        # Use official method for evaluation
        grading_response = grade_rubric_item_official(response_text, rubric_item, prompt)
        grading_response_list.append(grading_response)
        
        criteria_met = grading_response["criteria_met"]
        explanation = grading_response.get("explanation", "")
        
        # Strictly follow official format output
        print(f"   criteria_met: {criteria_met}")
        print(f"   explanation: {explanation}")
        if criteria_met:
            print(f"   Score: {points:+d} points")
        else:
            print(f"   Score: 0 points")
        
        # Avoid API rate limiting
        time.sleep(0.5)
    
    # Calculate score - official method
    total_possible_points = sum(
        rubric_item["points"] for rubric_item in rubric_items if rubric_item["points"] > 0
    )
    
    if total_possible_points == 0:
        return 0.0, grading_response_list
    
    achieved_points = sum(
        rubric_item["points"]
        for rubric_item, grading_response in zip(rubric_items, grading_response_list)
        if grading_response["criteria_met"]
    )
    
    overall_score = achieved_points / total_possible_points
    
    print(f"\\nSample scoring completed:")
    print(f"   Achieved points: {achieved_points}")
    print(f"   Total possible points: {total_possible_points}")
    print(f"   HealthBench score: {overall_score:.3f} ({overall_score*100:.1f}%)")
    
    return overall_score, grading_response_list
class SamplerResponse:
    """Simulate HealthBench's SamplerResponse"""
    def __init__(self, response_text: str, response_metadata: Dict = None, actual_queried_message_list: List = None):
        self.response_text = response_text
        self.response_metadata = response_metadata or {}
        self.actual_queried_message_list = actual_queried_message_list or []

class MCCSampler:
    """MCC framework's HealthBench Sampler adapter"""
    
    def __init__(self, max_rounds: int = 3):
        """
        Initialize MCC Sampler
        
        Args:
            max_rounds: Maximum debate rounds for MCC framework
        """
        self.max_rounds = max_rounds
    
    def __call__(self, prompt_messages: List[Dict[str, str]]) -> SamplerResponse:
        """
        Implement HealthBench's Sampler interface - completely following official standards
        
        Args:
            prompt_messages: Complete message list in HealthBench format
            
        Returns:
            SamplerResponse: Contains final answer generated by MCC framework
        """
        try:
            # Completely follow official ChatCompletionSampler approach
            # Use prompt_messages directly without any modification
            
            print(f"MCC Sampler received messages:")
            print(f"  Message count: {len(prompt_messages)}")
            for i, msg in enumerate(prompt_messages):
                role = msg.get("role", "")
                content = msg.get("content", "")  # Don't truncate, show complete content
                print(f"  [{i+1}] {role}: {content}")
            
            # Convert multi-turn conversation to format that MCC framework can handle
            # Key: We need to simulate ChatCompletionSampler behavior
            
            if len(prompt_messages) == 1 and prompt_messages[0].get("role") == "user":
                # Single-turn conversation
                question = prompt_messages[0].get("content", "")
            else:
                # Multi-turn conversation: Following official approach, use entire conversation as context
                # Build a question that lets MCC understand complete conversation
                conversation_str = "\\n\\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt_messages])
                
                question = f"""You are participating in a multi-turn conversation. Please provide an appropriate response based on the conversation history below:

{conversation_str}

Please provide your response as the assistant in this conversation."""
            
            # Use MCC framework for processing
            result = process_single_question(
                question=question,
                reference_answer=None,
                max_rounds=self.max_rounds,
                force_debate=False
            )
            
            # Return final answer
            final_answer = result.get("final_answer", "Sorry, unable to generate answer")
            
            # Return according to official standards
            return SamplerResponse(
                response_text=final_answer,
                response_metadata={"mcc_result": result},
                actual_queried_message_list=prompt_messages  # Return original message list, consistent with official
            )
            
        except Exception as e:
            print(f"MCC Sampler processing error: {str(e)}")
            traceback.print_exc()
            error_message = f"Sorry, an error occurred while processing your question. Error message: {str(e)}"
            return SamplerResponse(
                response_text=error_message,
                response_metadata={"error": str(e)},
                actual_queried_message_list=prompt_messages
            )
    

# HealthBench evaluation processing function
def process_healthbench_question(prompt_messages: List[Dict[str, str]], max_rounds: int = 3) -> str:
    """
    Process single HealthBench question
    
    Args:
        prompt_messages: Message list in HealthBench format
        max_rounds: Maximum debate rounds
        
    Returns:
        str: Final answer generated by MCC framework
    """
    sampler = MCCSampler(max_rounds=max_rounds)
    response = sampler(prompt_messages)
    return response.response_text

# ===================== Main Function and Running Logic =====================
def process_single_question(question, reference_answer=None, max_rounds=3, force_debate=False):
    """Process single medical question
    
    Args:
        question: Medical question
        reference_answer: Reference answer (optional)
        max_rounds: Maximum debate rounds
        force_debate: Whether to force debate (even if initial answers are consistent)
        
    Returns:
        Dict: Processing results
    """
    try:
        print(f"Processing question: {question}")
        
        # Get initial answers from three models in parallel
        print("Starting parallel acquisition of initial answers from three models")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks and associate Future objects with model names
            future_to_model = {
                executor.submit(generate_gpt_answer, question): "GPT",
                executor.submit(generate_qwen_answer, question): "Qwen",
                executor.submit(generate_deepseek_answer, question): "DeepSeek"
            }
            
            # Initialize result dictionary
            initial_answers = {}
            
            # Collect completed results
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    answer = future.result()
                    initial_answers[model_name] = answer
                    print(f"{model_name} has been recorded")
                except Exception as e:
                    print(f"{model_name} answer generation failed: {str(e)}")
                    initial_answers[model_name] = f"Error generating {model_name} answer: {str(e)}"
        
        print("\n" + "="*100)
        print("All models have generated initial answers")
        print("="*100)
        
        # Start circular criticism-improvement debate process
        debate_result = circular_criticism_improvement(
            question,
            initial_answers,
            max_rounds=max_rounds
        )
        
        print("\nFinal response:")
        print("="*100)
        print(debate_result["final_answer"])
        print("="*100)
        
        # Build result
        result = {
            "question": question,
            "initial_answers": initial_answers,
            "reference_answer": reference_answer,
            "final_answer": debate_result["final_answer"],
            "debate_history": debate_result["debate_history"],
            "consensus_score": debate_result["consensus_score"]
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        traceback.print_exc()
        
        # Build error result
        return {
            "question": question,
            "error": str(e),
            "error_traceback": traceback.format_exc()
        }

# Compatibility functions
def gpt_responds_to_others(question, qwen_answer, deepseek_answer, debate_round, self_previous_answer=None):
    """Compatibility function for old code calls, will be deprecated in new version
    
    This function is no longer used in current circular criticism-improvement mode, kept only for compatibility
    """
    print("Warning: gpt_responds_to_others function is deprecated, please use new circular criticism-improvement mode")
    return {
        "critique": "This function is deprecated",
        "improved_answer": self_previous_answer or "GPT answer"
    }

def qwen_responds_to_others(question, gpt_answer, deepseek_answer, debate_round, self_previous_answer=None):
    """Compatibility function for old code calls, will be deprecated in new version
    
    This function is no longer used in current circular criticism-improvement mode, kept only for compatibility
    """
    print("Warning: qwen_responds_to_others function is deprecated, please use new circular criticism-improvement mode")
    return {
        "critique": "This function is deprecated",
        "improved_answer": self_previous_answer or "Qwen answer"
    }

def deepseek_responds_to_others(question, gpt_answer, qwen_answer, debate_round, self_previous_answer=None):
    """Compatibility function for old code calls, will be deprecated in new version
    
    This function is no longer used in current circular criticism-improvement mode, kept only for compatibility
    """
    print("Warning: deepseek_responds_to_others function is deprecated, please use new circular criticism-improvement mode")
    return {
        "critique": "This function is deprecated",
        "improved_answer": self_previous_answer or "DeepSeek answer"
    }



# HealthBench evaluation main function
def run_healthbench_evaluation(subset_name: Optional[str] = None, num_examples: Optional[int] = None, max_rounds: int = 3, sample_indices: Optional[List[int]] = None):
    """
    Run HealthBench evaluation
    
    Args:
        subset_name: HealthBench subset name ("hard", "consensus" or None)
        num_examples: Limit number of evaluation samples
        max_rounds: Maximum debate rounds for MCC framework
    """
    try:
        # Set output redirection
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        subset_suffix = f"_{subset_name}" if subset_name else ""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # Add timestamp to main log file to avoid overwriting
        log_file = os.path.join(log_dir, f"healthbench{subset_suffix}_mcc_log_{timestamp}.txt")
        sys.stdout = TeeOutput(log_file)
        
        print("="*80)
        print(f"Starting HealthBench{subset_suffix} evaluation - using MCC framework")
        print("="*80)
        
        # Load HealthBench data
        examples = load_healthbench_data(subset_name=subset_name, num_examples=num_examples)
        
        original_indices = None
        if sample_indices is not None:
            print(f"Filtering specified sample indices: {sample_indices}")
            original_count = len(examples)
            original_indices = [i for i in sample_indices if i < len(examples)]
            examples = [examples[i] for i in sample_indices if i < len(examples)]
            print(f"Filtered {len(examples)} specified samples from {original_count} samples")
        else:
            # If no sample indices specified, original indices are sequential indices
            original_indices = list(range(len(examples)))
        
        # Create MCC Sampler
        mcc_sampler = MCCSampler(max_rounds=max_rounds)
        
        # Create results directory
        results_dir = "healthbench_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Process each sample
        results = []
        for i, example in enumerate(examples):
            # Get current sample's index in original dataset
            original_index = original_indices[i]
            print(f"\n{'='*50}")
            print(f"Processing HealthBench question {i+1}/{len(examples)}")
            print(f"{'='*50}")
            
            # Create separate log file for each sample - named using original index in dataset
            sample_log_file = os.path.join(log_dir, f"{subset_name or 'healthbench'}_sample_{original_index}.txt")
            
            # Create sample-specific log output
            sample_stdout = TeeOutput(sample_log_file)
            original_stdout = sys.stdout
            sys.stdout = sample_stdout
            
            print(f"{'='*80}")
            print(f"HealthBench sample log - {subset_name or 'healthbench'}_sample_{original_index}")
            print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Subset: {subset_name or 'Full dataset'}")
            print(f"Original index in dataset: {original_index}")
            print(f"Index in current batch: {i}")
            print(f"{'='*80}")
            
            try:
                # Get prompt messages
                prompt_messages = example["prompt"]
                
                # Use MCC framework to generate answer
                print("Generating MCC answer...")
                sampler_response = mcc_sampler(prompt_messages)
                response_text = sampler_response.response_text
                
                print(f"MCC answer generation completed, length: {len(response_text)} characters")
                
                # Get individual model initial answers (extracted from MCC results)
                mcc_result = sampler_response.response_metadata.get("mcc_result", {})
                initial_answers = mcc_result.get("initial_answers", {})
                
                # Evaluate individual model initial answers
                individual_scores = {}
                if initial_answers:
                    print("\nStarting evaluation of individual model initial answers...")
                    for model_name, initial_answer in initial_answers.items():
                        if initial_answer and "Error" not in initial_answer:
                            print(f"\nEvaluating {model_name} initial answer...")
                            try:
                                rubric_items = [rubric.to_dict() for rubric in example["rubrics"]]
                                model_score, model_grading = grade_sample_official(
                                    prompt=prompt_messages,
                                    response_text=initial_answer,
                                    rubric_items=rubric_items
                                )
                                individual_scores[model_name] = {
                                    "score": model_score,
                                    "grading_details": model_grading,
                                    "response_text": initial_answer
                                }
                                print(f"{model_name} initial answer HealthBench score: {model_score:.3f} ({model_score*100:.1f}%)")
                            except Exception as e:
                                print(f"Error evaluating {model_name} initial answer: {str(e)}")
                                individual_scores[model_name] = {"error": str(e)}
                
                # Evaluate MCC final answer
                print("\nStarting evaluation of MCC final integrated answer...")
                rubric_items = [rubric.to_dict() for rubric in example["rubrics"]]
                mcc_score, mcc_grading_details = grade_sample_official(
                    prompt=prompt_messages,
                    response_text=response_text,
                    rubric_items=rubric_items
                )
                
                # Calculate MCC improvement relative to individual models
                improvement_analysis = {}
                if individual_scores:
                    individual_model_scores = [v["score"] for v in individual_scores.values() if "score" in v]
                    if individual_model_scores:
                        best_individual_score = max(individual_model_scores)
                        avg_individual_score = sum(individual_model_scores) / len(individual_model_scores)
                        improvement_analysis = {
                            "best_individual_score": best_individual_score,
                            "avg_individual_score": avg_individual_score,
                            "mcc_score": mcc_score,
                            "improvement_vs_best": mcc_score - best_individual_score,
                            "improvement_vs_avg": mcc_score - avg_individual_score
                        }
                
                # Save complete results
                result = {
                    "prompt_id": example.get("prompt_id", f"sample_{original_index}"),
                    "sample_index": original_index,  # Use original index in dataset
                    "prompt": prompt_messages,
                    "individual_models": individual_scores,  # Initial answers and scores from individual models
                    "mcc_response": response_text,
                    "mcc_healthbench_score": mcc_score,
                    "mcc_grading_details": mcc_grading_details,
                    "improvement_analysis": improvement_analysis,
                    "rubrics": rubric_items,
                    "example_tags": example.get("example_tags", []),
                    "mcc_metadata": sampler_response.response_metadata
                }
                results.append(result)
                
                # Display detailed comparison results
                print(f"\n{'='*60}")
                print(f"Sample {i+1} evaluation completed")
                print(f"{'='*60}")
                
                if individual_scores:
                    print("Individual model initial answer HealthBench scores:")
                    for model_name, score_data in individual_scores.items():
                        if "score" in score_data:
                            score = score_data["score"]
                            print(f"  {model_name:8}: {score:.3f} ({score*100:.1f}%)")
                        else:
                            print(f"  {model_name:8}: Evaluation failed")
                
                print(f"MCC final answer HealthBench score: {mcc_score:.3f} ({mcc_score*100:.1f}%)")
                
                if improvement_analysis:
                    print(f"\nMCC framework improvement analysis:")
                    print(f"  vs best individual model: {improvement_analysis['improvement_vs_best']:+.3f}")
                    print(f"  vs average individual model: {improvement_analysis['improvement_vs_avg']:+.3f}")
                    
                    if improvement_analysis['improvement_vs_best'] > 0:
                        print("  ✓ MCC framework surpassed best individual model!")
                    elif improvement_analysis['improvement_vs_avg'] > 0:
                        print("  ✓ MCC framework surpassed average individual model performance")
                    else:
                        print("  - MCC framework did not significantly improve individual model performance")
                
                # Close sample log, restore main log
                sample_stdout.close()
                sys.stdout = original_stdout
                
                # Immediately generate CSV file for current sample - named using original index in dataset
                sample_csv_file = os.path.join(results_dir, f"{subset_name or 'healthbench'}_sample_{original_index}.csv")
                try:
                    export_to_csv([result], sample_csv_file, append_mode=False, sample_index=original_index)
                    print(f"✓ Sample {i+1} CSV file saved to: {sample_csv_file}")
                except Exception as csv_error:
                    print(f"✗ Sample {i+1} CSV save failed: {str(csv_error)}")
                
                print(f"✓ Sample {i+1} processing completed, detailed log saved to: {sample_log_file}")
                
            except Exception as e:
                print(f"Error processing question {i+1}: {str(e)}")
                traceback.print_exc()
                
                # Close sample log, restore main log
                if 'sample_stdout' in locals():
                    sample_stdout.close()
                    sys.stdout = original_stdout
                
                print(f"✗ Sample {i+1} processing failed, error log saved to: {sample_log_file}")
                
                # Add error result
                error_result = {
                    "prompt_id": example.get("prompt_id", f"sample_{original_index}"),
                    "sample_index": original_index,  # Use original index in dataset
                    "prompt": example["prompt"],
                    "mcc_response": f"Processing error: {str(e)}",
                    "rubrics": [rubric.to_dict() for rubric in example["rubrics"]],
                    "example_tags": example.get("example_tags", []),
                    "error": str(e)
                }
                results.append(error_result)
                
                # Try to generate CSV file even on error (containing error information) - named using original index in dataset
                sample_csv_file = os.path.join(results_dir, f"{subset_name or 'healthbench'}_sample_{original_index}_ERROR.csv")
                try:
                    export_to_csv([error_result], sample_csv_file, append_mode=False, sample_index=original_index)
                    print(f"✓ Sample {i+1} error CSV file saved to: {sample_csv_file}")
                except Exception as csv_error:
                    print(f"✗ Sample {i+1} error CSV save failed: {str(csv_error)}")
        
        # Calculate detailed HealthBench scoring statistics
        successful_results = [r for r in results if 'error' not in r and 'mcc_healthbench_score' in r]
        
        # MCC scoring statistics
        if successful_results:
            mcc_scores = [r['mcc_healthbench_score'] for r in successful_results]
            mcc_avg_score = max(0.0, min(1.0, sum(mcc_scores) / len(mcc_scores)))  # Clip to [0,1] range
            mcc_min_score = min(mcc_scores)
            mcc_max_score = max(mcc_scores)
        else:
            mcc_avg_score = mcc_min_score = mcc_max_score = 0.0
        
        # Individual model scoring statistics
        individual_model_stats = {}
        models = ["GPT", "Qwen", "DeepSeek"]
        
        for model in models:
            model_scores = []
            for r in successful_results:
                if 'individual_models' in r and model in r['individual_models']:
                    if 'score' in r['individual_models'][model]:
                        model_scores.append(r['individual_models'][model]['score'])
            
            if model_scores:
                individual_model_stats[model] = {
                    "avg_score": max(0.0, min(1.0, sum(model_scores) / len(model_scores))),  # Clip to [0,1] range
                    "min_score": min(model_scores),
                    "max_score": max(model_scores),
                    "num_samples": len(model_scores)
                }
            else:
                individual_model_stats[model] = {
                    "avg_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0,
                    "num_samples": 0
                }
        
        # Calculate overall improvement statistics
        improvement_stats = {}
        if successful_results:
            improvements_vs_best = []
            improvements_vs_avg = []
            for r in successful_results:
                if 'improvement_analysis' in r:
                    analysis = r['improvement_analysis']
                    if 'improvement_vs_best' in analysis:
                        improvements_vs_best.append(analysis['improvement_vs_best'])
                    if 'improvement_vs_avg' in analysis:
                        improvements_vs_avg.append(analysis['improvement_vs_avg'])
            
            if improvements_vs_best:
                improvement_stats['vs_best'] = {
                    "avg_improvement": sum(improvements_vs_best) / len(improvements_vs_best),
                    "positive_improvements": sum(1 for x in improvements_vs_best if x > 0),
                    "total_comparisons": len(improvements_vs_best)
                }
            
            if improvements_vs_avg:
                improvement_stats['vs_avg'] = {
                    "avg_improvement": sum(improvements_vs_avg) / len(improvements_vs_avg),
                    "positive_improvements": sum(1 for x in improvements_vs_avg if x > 0),
                    "total_comparisons": len(improvements_vs_avg)
                }
        
        # Save all results with summary information
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        final_results = {
            "evaluation_start_time": datetime.datetime.now().isoformat(),
            "evaluation_summary": {
                "subset": subset_name or "Full dataset",
                "num_samples": len(examples),
                "successful_samples": len(successful_results),
                "failed_samples": len([r for r in results if 'error' in r]),
                "mcc_max_rounds": max_rounds,
                "timestamp": timestamp,
                "mcc_performance": {
                    "avg_score": mcc_avg_score,
                    "min_score": mcc_min_score,
                    "max_score": mcc_max_score
                },
                "individual_models_performance": individual_model_stats,
                "improvement_analysis": improvement_stats
            },
            "samples": results
        }
        
        result_file = os.path.join(results_dir, f"healthbench{subset_suffix}_mcc_results_{timestamp}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # Optional: Export summary CSV file with detailed information for all samples, add timestamp to avoid overwriting
        summary_csv_file = os.path.join(results_dir, f"healthbench{subset_suffix}_mcc_summary_{timestamp}.csv")
        if successful_results:
            try:
                # Summary CSV doesn't need sample_index parameter, will use enumerate index
                export_to_csv(successful_results, summary_csv_file, append_mode=False)
                print(f"✓ Summary CSV file saved to: {summary_csv_file}")
            except Exception as e:
                print(f"✗ Summary CSV save failed: {str(e)}")
        
        print(f"\n{'='*80}")
        print(f"HealthBench{subset_suffix} evaluation completed - detailed comparative analysis")
        print(f"{'='*80}")
        print(f"Total processed: {len(examples)} questions")
        print(f"Successfully processed: {len(successful_results)} questions")
        print(f"Failed processed: {len([r for r in results if 'error' in r])} questions")
        
        print(f"\n📊 MCC framework vs individual model performance comparison:")
        print(f"{'='*50}")
        
        # Display individual model performance
        print("Individual model HealthBench average scores:")
        for model, stats in individual_model_stats.items():
            if stats['num_samples'] > 0:
                print(f"  {model:8}: {stats['avg_score']:.3f} ({stats['avg_score']*100:.1f}%) [{stats['num_samples']} samples]")
            else:
                print(f"  {model:8}: No valid evaluation")
        
        print(f"\nMCC framework HealthBench average score:")
        print(f"  MCC framework: {mcc_avg_score:.3f} ({mcc_avg_score*100:.1f}%) [{len(successful_results)} samples]")
        
        # Display improvement analysis
        if improvement_stats:
            print(f"\n🚀 MCC framework improvement effects:")
            if 'vs_best' in improvement_stats:
                vs_best = improvement_stats['vs_best']
                print(f"  vs best individual model:")
                print(f"    Average improvement: {vs_best['avg_improvement']:+.3f}")
                print(f"    Improved samples: {vs_best['positive_improvements']}/{vs_best['total_comparisons']} ({vs_best['positive_improvements']/vs_best['total_comparisons']*100:.1f}%)")
            
            if 'vs_avg' in improvement_stats:
                vs_avg = improvement_stats['vs_avg']
                print(f"  vs average individual model:")
                print(f"    Average improvement: {vs_avg['avg_improvement']:+.3f}")
                print(f"    Improved samples: {vs_avg['positive_improvements']}/{vs_avg['total_comparisons']} ({vs_avg['positive_improvements']/vs_avg['total_comparisons']*100:.1f}%)")
        
        print(f"\n📁 File saves:")
        print(f"  JSON result file: {result_file}")
        print(f"  Summary CSV file: {summary_csv_file}")
        print(f"  Main log file: {log_file}")
        print(f"  Sample log files: logs/{subset_name or 'healthbench'}_sample_[dataset_index].txt")
        print(f"  Sample CSV files: healthbench_results/{subset_name or 'healthbench'}_sample_[dataset_index].csv")
        print(f"  Generated {len(successful_results)} sample log files and {len(results)} sample CSV files")
        print("="*80)
        
        # Close log file
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            
        return results
        
    except Exception as e:
        print(f"Error during HealthBench evaluation: {str(e)}")
        traceback.print_exc()
        
        # Ensure log file is closed
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        
        raise

# Main function
def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='MCC Medical Q&A Framework - Support HealthBench Evaluation')
        parser.add_argument('-r', '--max_rounds', type=int, default=3, help='Maximum debate rounds (default: 3)')
        parser.add_argument('-q', '--question', type=str, help='Directly specify question to process')
        
        # HealthBench related parameters
        parser.add_argument('--healthbench', action='store_true', help='Run HealthBench evaluation mode')
        parser.add_argument('--healthbench_subset', type=str, choices=['hard', 'consensus'], help='HealthBench subset (hard, consensus)')
        parser.add_argument('--healthbench_examples', type=int, help='Number of HealthBench evaluation samples')
        
        args = parser.parse_args()
        
        # If HealthBench mode is specified
        if args.healthbench:
            return run_healthbench_evaluation(
                subset_name=args.healthbench_subset,
                num_examples=args.healthbench_examples,
                max_rounds=args.max_rounds
            )
        
        # Original processing logic
        # Set output redirection to terminal and log file
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"mcc_direct_log.txt")
        sys.stdout = TeeOutput(log_file)
        
        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Process directly specified question
        if args.question:
            print("\n" + "="*50)
            print(f"Processing specified question: {args.question}")
            print("="*50)
            
            # Process single question
            result = process_single_question(
                args.question,
                reference_answer=None,
                max_rounds=args.max_rounds,
                force_debate=False
            )
            
            # Save result to JSON file
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = os.path.join(results_dir, f"mcc_direct_results.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\nProcessing results saved to {result_file}")
            print(f"Log file saved to {log_file}")
            
        
        # Close log file
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            # Restore standard output
            sys.stdout = sys.__stdout__
    
    except Exception as e:
        print(f"Error during program execution: {str(e)}")
        traceback.print_exc()
        
        # Ensure log file is closed
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            sys.stdout = sys.__stdout__

def export_to_csv(results, csv_file_path, append_mode=False, sample_index=None):
    """
    Export results to CSV file
    
    Args:
        results: Evaluation result list (can be single result or result list)
        csv_file_path: CSV file save path
        append_mode: Whether in append mode (don't write header when True)
        sample_index: Sample index (for single sample export)
    """
    try:
        import csv
        
        # Ensure results is in list format
        if not isinstance(results, list):
            results = [results]
        
        print(f"Exporting CSV file: {csv_file_path} ({'Append mode' if append_mode else 'New file mode'})")
        
        def clean_text_for_csv(text, preserve_structure=False):
            """Clean text to make it suitable for CSV format"""
            if not text:
                return ""
            
            if preserve_structure:
                # Maintain paragraph and title structure for Excel viewing
                # Keep title lines as separate lines
                text = text.replace("### ", "\n### ")  # Line break before titles
                text = text.replace("## ", "\n## ")   # Line break before level-2 titles
                text = text.replace("# ", "\n# ")     # Line break before level-1 titles
                text = text.replace("---", "\n---\n")  # Line breaks before and after separators
                
                # Maintain list structure
                text = text.replace("\n- ", "\n  • ")  # List items
                text = text.replace("\n* ", "\n  • ")  # List items
                
                # Clean excessive line breaks
                while "\n\n\n" in text:
                    text = text.replace("\n\n\n", "\n\n")
            else:
                # Remove all line breaks, replace with spaces
                text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            
            # Remove excessive spaces
            text = " ".join(text.split())
            # Escape double quotes
            text = text.replace('"', '""')
            
            return text
        
        # Define CSV columns - restore original format but optimize visualization
        csv_columns = [
            "Sample Index",
            "Question ID", 
            "Official Question", 
            "GPT Initial Response",
            "Qwen Initial Response", 
            "DeepSeek Initial Response",
            "MCC Final Response",
            "Scoring Criteria",
            "GPT Initial Response Scoring Output",
            "Qwen Initial Response Scoring Output",
            "DeepSeek Initial Response Scoring Output", 
            "MCC Response Scoring Output",
            "GPT Score",
            "Qwen Score",
            "DeepSeek Score",
            "MCC Score",
            "Best Individual Model Score",
            "MCC Improvement Effect"
        ]
        
        # Select file opening mode based on mode
        file_mode = 'a' if append_mode else 'w'
        
        with open(csv_file_path, file_mode, newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)  # Add quotes to all fields to avoid format issues
            
            # Write header only in non-append mode
            if not append_mode:
                writer.writerow(csv_columns)
            
            for idx, result in enumerate(results):
                # Extract basic information
                prompt_id = result.get("prompt_id", "")
                
                # Determine sample index
                if sample_index is not None:
                    # Use passed index when exporting single sample
                    current_sample_index = sample_index
                elif "sample_index" in result:
                    # Use original sample index stored in result
                    current_sample_index = result["sample_index"]
                else:
                    # Use loop index as backup when batch exporting
                    current_sample_index = idx
                
                # Extract complete conversation structure for CSV
                question = ""
                if "prompt" in result and result["prompt"]:
                    # Build complete conversation history
                    conversation_parts = []
                    last_user_message = ""
                    
                    for msg in result["prompt"]:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        conversation_parts.append(f"{role.title()}: {content}")
                        
                        if role == "user":
                            last_user_message = content
                    
                    # Always save complete conversation context for understanding question background
                    if len(result["prompt"]) > 1:
                        question = f"Multi-turn conversation - Final question: {last_user_message} | Complete conversation: {' || '.join(conversation_parts)}"
                    else:
                        # Even for single-turn conversation, save complete role and content information
                        question = f"Single-turn conversation: {' || '.join(conversation_parts)}"
                
                mcc_response = result.get("mcc_response", "")
                
                # Extract initial responses from each model
                individual_models = result.get("individual_models", {})
                gpt_initial = individual_models.get("GPT", {}).get("response_text", "")
                qwen_initial = individual_models.get("Qwen", {}).get("response_text", "")
                deepseek_initial = individual_models.get("DeepSeek", {}).get("response_text", "")
                
                # Handle error cases
                if 'error' in result:
                    gpt_initial = qwen_initial = deepseek_initial = f"Processing error: {result['error']}"
                
                # Extract scoring criteria
                rubrics = result.get("rubrics", [])
                rubrics_text = "\\n".join([f"[{r.get('points', 0):+d} points] {r.get('criterion', '')}" for r in rubrics])
                
                # Extract scoring output
                def format_grading_output(grading_details):
                    if not grading_details:
                        return ""
                    output_lines = []
                    for detail in grading_details:
                        criterion = detail.get("criterion", "")
                        criteria_met = detail.get("criteria_met", False)
                        explanation = detail.get("explanation", "")
                        output_lines.append(f"Criterion: {criterion}\\ncriteria_met: {criteria_met}\\nexplanation: {explanation}\\n")
                    return "\\n".join(output_lines)
                
                gpt_grading_output = ""
                qwen_grading_output = ""
                deepseek_grading_output = ""
                if individual_models:
                    gpt_grading_output = format_grading_output(individual_models.get("GPT", {}).get("grading_details", []))
                    qwen_grading_output = format_grading_output(individual_models.get("Qwen", {}).get("grading_details", []))
                    deepseek_grading_output = format_grading_output(individual_models.get("DeepSeek", {}).get("grading_details", []))
                
                mcc_grading_output = format_grading_output(result.get("mcc_grading_details", []))
                
                # Extract scores (handle error cases)
                if 'error' in result:
                    gpt_score = qwen_score = deepseek_score = mcc_score = 0.0
                    best_individual = improvement_effect = 0.0
                else:
                    gpt_score = individual_models.get("GPT", {}).get("score", 0.0)
                    qwen_score = individual_models.get("Qwen", {}).get("score", 0.0) 
                    deepseek_score = individual_models.get("DeepSeek", {}).get("score", 0.0)
                    mcc_score = result.get("mcc_healthbench_score", 0.0)
                    
                    # Extract improvement analysis
                    improvement_analysis = result.get("improvement_analysis", {})
                    best_individual = improvement_analysis.get("best_individual_score", 0.0)
                    improvement_effect = improvement_analysis.get("improvement_vs_best", 0.0)
                
                # Optimize scoring output format - add score display and better structure
                def format_grading_enhanced(grading_details, rubric_items):
                    if not grading_details:
                        return ""
                    
                    parts = []
                    for i, detail in enumerate(grading_details):
                        criteria_met = detail.get("criteria_met", False)
                        explanation = detail.get("explanation", "")
                        
                        # Get corresponding score
                        points = 0
                        if i < len(rubric_items):
                            points = rubric_items[i].get("points", 0)
                        
                        # Calculate actual score
                        actual_score = points if criteria_met else 0
                        
                        met_text = "Met" if criteria_met else "Not met"
                        criterion = detail.get("criterion", "")
                        
                        # Build enhanced evaluation information with scores
                        criterion_short = criterion[:100] + "..." if len(criterion) > 100 else criterion
                        explanation_clean = explanation.replace("\\n", " ").replace("\\r", " ")
                        explanation_clean = " ".join(explanation_clean.split())
                        
                        parts.append(f"[Criterion {i+1}] {criterion_short}\n  Result: {met_text} ({actual_score:+d} points)\n  Explanation: {explanation_clean}\n")
                    
                    return "\n".join(parts)  # Use real line breaks, Excel will display correctly
                
                # Generate scoring output for each model - use rubrics from results
                rubrics = result.get("rubrics", [])
                
                if 'error' in result:
                    # Scoring output for error cases
                    error_msg = f"Evaluation failed: {result['error']}"
                    gpt_grading = qwen_grading = deepseek_grading = mcc_grading = error_msg
                else:
                    gpt_grading = format_grading_enhanced(individual_models.get("GPT", {}).get("grading_details", []), rubrics)
                    qwen_grading = format_grading_enhanced(individual_models.get("Qwen", {}).get("grading_details", []), rubrics)
                    deepseek_grading = format_grading_enhanced(individual_models.get("DeepSeek", {}).get("grading_details", []), rubrics)
                    mcc_grading = format_grading_enhanced(result.get("mcc_grading_details", []), rubrics)
                
                # Write CSV row - use different formatting for different content
                row = [
                    current_sample_index,  # Add sample index
                    clean_text_for_csv(prompt_id),
                    clean_text_for_csv(question),  # Keep questions concise
                    clean_text_for_csv(gpt_initial, preserve_structure=True),  # Preserve structure for responses
                    clean_text_for_csv(qwen_initial, preserve_structure=True),
                    clean_text_for_csv(deepseek_initial, preserve_structure=True),
                    clean_text_for_csv(mcc_response, preserve_structure=True),
                    clean_text_for_csv(rubrics_text),  # Keep criteria concise
                    gpt_grading,  # Scoring output already formatted, use directly
                    qwen_grading,
                    deepseek_grading,
                    mcc_grading,
                    f"{gpt_score:.3f}",
                    f"{qwen_score:.3f}",
                    f"{deepseek_score:.3f}",
                    f"{mcc_score:.3f}",
                    f"{best_individual:.3f}",
                    f"{improvement_effect:+.3f}"
                ]
                writer.writerow(row)
        
        print(f"✓ Enhanced CSV file export completed: {csv_file_path}")
        
    except Exception as e:
        print(f"Error exporting CSV file: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 