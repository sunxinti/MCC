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
import concurrent.futures  # Import concurrent execution module
from typing import Tuple, List, Dict, Any
from openai import OpenAI
import numpy as np

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
        print("Configuration complete! Starting MMLU Anatomy model debate system...")
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

# Custom output class that outputs to both console and log file
class TeeOutput:
    """Class that sends output to both terminal and log file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a", encoding="utf-8")
        # Write timestamp at the beginning of log file
        self.logfile.write(f"\n{'='*50}\n")
        self.logfile.write(f"Log start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logfile.write(f"{'='*50}\n\n")
        self.logfile.flush()

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
        
    def close(self):
        """Close log file"""
        if self.logfile:
            # Write timestamp at the end of log file
            self.logfile.write(f"\n{'='*50}\n")
            self.logfile.write(f"Log end time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.logfile.write(f"{'='*50}\n\n")
            self.logfile.close()
            self.logfile = None

# Check if file exists
def check_file_exists(file_path):
    """Check if file exists"""
    if not os.path.exists(file_path):
        print("Error: File '{}' does not exist!".format(file_path))
        return False
    return True

# Load MMLU anatomy data
def load_medical_mcq_data(file_path):
    """Load MMLU anatomy dataset"""
    try:
        if not check_file_exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        print(f"Loading dataset: {file_path}")
        
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line.strip()))
        
        dataset = pd.DataFrame(data_list)
        print(f"Successfully loaded dataset with {len(dataset)} cases")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
        raise

# Get options list
def get_choices(dataset, case_idx):
    """Get options list for specified case from dataset
    
    Args:
        dataset: Dataset
        case_idx: Case index
        
    Returns:
        str: Formatted options list, like "1. Option1\n2. Option2"
    """
    try:
        # Get options
        options = dataset.iloc[case_idx]['options']
        
        if isinstance(options, dict):
            choices_list = []
            for i, (key, value) in enumerate(options.items(), 1):
                choices_list.append(f"{i}. {value}")
            return "\n".join(choices_list)
        
        elif isinstance(options, list):
            choices_list = [f"{i+1}. {option}" for i, option in enumerate(options)]
            return "\n".join(choices_list)
        
        elif isinstance(options, str) and options.startswith('{'):
            try:
                options_dict = json.loads(options)
                choices_list = []
                for i, (key, value) in enumerate(options_dict.items(), 1):
                    choices_list.append(f"{i}. {value}")
                return "\n".join(choices_list)
            except:
                pass
        
        # If no options found
        print(f"Warning: Unable to extract choices list from case index {case_idx}")
        print(f"Dataset columns: {list(dataset.columns)}")
        raise ValueError(f"Unable to extract choices list for case {case_idx}, please check data format")
        
    except Exception as e:
        print(f"Error occurred while getting choices list: {str(e)}")
        print(f"Case index: {case_idx}")
        print(f"Dataset columns: {list(dataset.columns) if hasattr(dataset, 'columns') else 'Unable to get column names'}")
        traceback.print_exc()
        raise e




# ===================== GPT Model Section =====================
def get_gpt_prompt(case_vignette, choices):
    system_prompt = """You are analyzing an anatomy multiple-choice question. Your task is to use your knowledge of human anatomy to select the most accurate answer from the given options.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Question】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Options】
{choices}

Please provide a structured analysis using the following format:

**1. Question Analysis**  
- Analyze what the question is asking about in terms of anatomical structures or concepts.  

**2. Key Anatomical Considerations**  
- Discuss the relevant anatomical knowledge needed to answer this question.
- Explain the anatomical structures, relationships, or concepts involved.

**3. Analysis of Options**  
- Systematically evaluate each option based on anatomical facts.
- Determine whether each option is correct or incorrect based on anatomical knowledge.

**4. Final Selection**  
- Clearly state the option you believe is correct.
- Explain why this option is the most accurate answer.
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X**  

Note: You must choose one option from the provided list and clearly indicate the option number as per the format above.  
Ensure your analysis is based on accurate anatomical knowledge."""
    return system_prompt

# Use ChatGPT for reasoning
def generate_gpt_answer(case_vignette, choices):
    """Use ChatGPT to generate multiple choice answer"""
    try:
        prompt = get_gpt_prompt(case_vignette, choices)
        
        print("\nGPT is reasoning the answer...")
        t_generate_start = time.time()
        
        data = {
            "model": "o1-mini", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are a top-tier anatomist with exceptional clinical knowledge of human anatomy and physiology. Your primary task is to maximize accuracy in answering anatomical questions."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 8000
        }

        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
        t_generate = time.time() - t_generate_start
        print(f"GPT API response status code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"GPT answer generation completed, time taken: {t_generate:.2f} seconds")
            return answer
        else:
            print(f"GPT API error: {response.status_code}")
            print(f"Error details: {response.text}")
            return f"Sorry, an error occurred while processing your question. Error code: {response.status_code}"

    except Exception as e:
        print(f"Error generating GPT answer: {str(e)}")
        traceback.print_exc()
        return f"Sorry, an error occurred while processing your question. Error message: {str(e)}"



# ===================== Qwen Model Section =====================
def get_qwen_prompt(case_vignette, choices):
    system_prompt = """You are analyzing an anatomy multiple-choice question. Your task is to use your knowledge of human anatomy to select the most accurate answer from the given options.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Question】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Options】
{choices}

Please provide a structured analysis using the following format:

**1. Question Analysis**  
- Analyze what the question is asking about in terms of anatomical structures or concepts.  

**2. Key Anatomical Considerations**  
- Discuss the relevant anatomical knowledge needed to answer this question.
- Explain the anatomical structures, relationships, or concepts involved.

**3. Analysis of Options**  
- Systematically evaluate each option based on anatomical facts.
- Determine whether each option is correct or incorrect based on anatomical knowledge.

**4. Final Selection**  
- Clearly state the option you believe is correct.
- Explain why this option is the most accurate answer.
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X**  

Note: You must choose one option from the provided list and clearly indicate the option number as per the format above.  
Ensure your analysis is based on accurate anatomical knowledge."""
    return system_prompt

# Use Qwen for reasoning
def generate_qwen_answer(case_vignette, choices):
    """Use Qwen to generate multiple choice answer"""
    try:
        prompt = get_qwen_prompt(case_vignette, choices)
        
        print("\nQwen is generating answer...")
        t_generate_start = time.time()
        
        # Build request data
        data = {
            "model": "Qwen/QwQ-32B", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are a top-tier anatomist with exceptional clinical knowledge of human anatomy and physiology. Your primary task is to maximize accuracy in answering anatomical questions."},
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
                    
                    # Get content
                    if 'content' in message and message['content'] and len(message['content'].strip()) > 0:
                        answer = message['content'].strip()
                    elif 'reasoning_content' in message and message['reasoning_content']:
                        answer = message['reasoning_content'].strip()
                    else:
                        print("Warning: API returned empty content")
                        answer = "API returned empty content"
                    
                    print(f"Qwen answer generation completed, time taken: {t_generate:.2f} seconds")
                    return answer
                else:
                    print("Error: message field does not exist")
                    return "API response structure abnormal: missing message field"
            else:
                print("Error: choices field does not exist or is empty")
                return "API response structure abnormal: missing choices field"
        else:
            print(f"Qwen API error: {response.status_code}")
            print(f"Error details: {response.text}")
            print("Unable to call Qwen API, task terminated")
            sys.exit(1)

    except Exception as e:
        print(f"Error generating Qwen answer: {str(e)}")
        traceback.print_exc()
        print("Unable to generate Qwen answer, task terminated")
        sys.exit(1)



# ===================== DeepSeek Model Section =====================
def get_deepseek_prompt(case_vignette, choices):
    system_prompt = """You are analyzing an anatomy multiple-choice question. Your task is to use your knowledge of human anatomy to select the most accurate answer from the given options.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Question】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Options】
{choices}

Please provide a structured analysis using the following format:

**1. Question Analysis**  
- Analyze what the question is asking about in terms of anatomical structures or concepts.  

**2. Key Anatomical Considerations**  
- Discuss the relevant anatomical knowledge needed to answer this question.
- Explain the anatomical structures, relationships, or concepts involved.

**3. Analysis of Options**  
- Systematically evaluate each option based on anatomical facts.
- Determine whether each option is correct or incorrect based on anatomical knowledge.

**4. Final Selection**  
- Clearly state the option you believe is correct.
- Explain why this option is the most accurate answer.
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X**  

Note: You must choose one option from the provided list and clearly indicate the option number as per the format above.  
Ensure your analysis is based on accurate anatomical knowledge."""
    return system_prompt

# Use DeepSeek for reasoning
def generate_deepseek_answer(case_vignette, choices):
    """Use DeepSeek to generate multiple choice answer"""
    try:
        prompt = get_deepseek_prompt(case_vignette, choices)
        
        print("\nDeepSeek is generating answer...")
        t_generate_start = time.time()
        
        # Try using OpenAI client for API call
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model="deepseek-reasoner", # Note, use the same as the original study; you can replace it with other LLMs. 
                messages=[
                    {"role": "system", "content": "You are a top-tier anatomist with exceptional clinical knowledge of human anatomy and physiology. Your primary task is to maximize accuracy in answering anatomical questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )
            
            answer = response.choices[0].message.content
            t_generate = time.time() - t_generate_start
            print(f"DeepSeek answer generation completed, time taken: {t_generate:.2f} seconds")
            return answer
            
        except Exception as e:
            print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
            print("Trying to call API directly using requests...")
            
            # Fallback: direct requests call using SiliconFlow API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1", # Note, use the same as the original study; you can replace it with other LLMs. 
                "messages": [
                    {"role": "system", "content": "You are a top-tier anatomist with exceptional clinical knowledge of human anatomy and physiology. Your primary task is to maximize accuracy in answering anatomical questions."},
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
                print(f"DeepSeek answer generation completed, time taken: {t_generate:.2f} seconds")
                return answer
            else:
                print(f"DeepSeek API error: {response.status_code}")
                print(f"Error details: {response.text}")
                print("Unable to call DeepSeek API, task terminated")
                sys.exit(1)
        
    except Exception as e:
        print(f"Error generating DeepSeek answer: {str(e)}")
        traceback.print_exc()
        print("Unable to generate DeepSeek answer, task terminated")
        sys.exit(1) 



# ===================== Model Debate Section =====================
# Extract selected option number from response
def extract_model_choice(answer_text, choices_text=None):
    """Extract final choice from model response
    
    Args:
        answer_text: Complete response text from model
        choices_text: Options text for dynamic matching (optional)
    
    Returns:
        int: Extracted option number (1-n), raises ValueError if extraction fails
    """
    
    # Debug information
    print("\nStarting model choice extraction...")
    
    clean_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer_text)
    
    # Priority extraction of final conclusion section
    conclusion_markers = [
        "final decision", "final selection", "final choice", "final answer", "final determination",
        "in conclusion", "to conclude", "my conclusion", "my answer"
    ]
    
    # Find final conclusion paragraph
    conclusion_text = ""
    lines = clean_answer.split("\n")
    for i, line in enumerate(lines):
        if any(marker.lower() in line.lower() for marker in conclusion_markers):
            # Get this line and subsequent lines as conclusion text
            conclusion_text = "\n".join(lines[i:min(i+5, len(lines))])
            break
    
    # If no clear conclusion paragraph found, use last few lines
    if not conclusion_text:
        conclusion_text = "\n".join(lines[-15:])
    
    # 1. First match standard format patterns
    final_choice_patterns = [
        r'my final selection is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'my final choice is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'my final answer is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'final (?:selection|choice|answer) is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'(?:selection|choice|answer) is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'i select (?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'i choose (?:\*\*)?option\s*(\d+)(?:\*\*)?',
    ]
    
    # Priority search in conclusion text
    search_texts = [conclusion_text, clean_answer]
    
    # Match most explicit conclusion patterns
    for text in search_texts:
        for pattern in final_choice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    option_num = int(match.group(1))
                    if 1 <= option_num <= 10:  # Expand option range to 10
                        print(f"[Extract] Found final selection: Option {option_num}")
                        return option_num
                except (ValueError, IndexError):
                    continue
    
    # 2. If standard format not found, look for other explicit choice expressions
    option_explicit_patterns = [
        r'(?:choose|select)[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'(?:\*\*)?option\s*(\d+)(?:\*\*)?[^.,:;]*?(?:is|as)[^.,:;]*?(?:the correct|most appropriate|best)',
        r'i (?:believe|think)[^.,:;]*?(?:\*\*)?option\s*(\d+)(?:\*\*)?',
    ]
    
    for text in search_texts:
        for pattern in option_explicit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    option_num = int(match.group(1))
                    if 1 <= option_num <= 10:
                        print(f"[Extract] Found option reference: Option {option_num}")
                        return option_num
                except (ValueError, IndexError):
                    continue
    
    # If extraction still fails, raise exception instead of using fallback strategy
    print("WARNING: Unable to extract clear choice from response")
    raise ValueError("Failed to extract valid choice from model response")


# Initialize debate
def initialize_debate(case_vignette, choices, force_disagree=False):
    """Initialize debate and get initial responses from three models
    
    Args:
        case_vignette: Question description
        choices: Options list
        force_disagree: Whether to force simulate disagreement (for testing)
        
    Returns:
        dict: Initial responses from three models
    """
    print("="*50)
    print("Starting anatomy multiple choice debate")
    print("="*80)
    
    # Build mapping from option number to option content
    choice_to_content = {}
    choice_lines = choices.strip().split('\n')
    for line in choice_lines:
        match = re.match(r'(\d+)\.\s*(.+)', line.strip())
        if match:
            option_num = int(match.group(1))
            content = match.group(2).strip()
            choice_to_content[option_num] = content
    
    # Start parallel execution to get initial responses from three models
    print("Starting parallel execution to get initial responses from three models")
    start_time = time.time()
    
    # Initialize results storage
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks for parallel execution
        gpt_future = executor.submit(generate_gpt_answer, case_vignette, choices)
        qwen_future = executor.submit(generate_qwen_answer, case_vignette, choices)
        deepseek_future = executor.submit(generate_deepseek_answer, case_vignette, choices)
        
        # Process results as they complete (immediate output)
        future_to_model = {
            gpt_future: ("gpt", "GPT"),
            qwen_future: ("qwen", "Qwen"), 
            deepseek_future: ("deepseek", "DeepSeek")
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model_key, model_name = future_to_model[future]
            try:
                answer = future.result()
                choice = extract_model_choice(answer, choices)
                
                # Store results
                results[model_key] = {
                    "answer": answer,
                    "choice": choice
                }
                
                # Immediate output as each model completes
                print(f"\n{model_name} completed!")
                print(f"{model_name}'s choice:")
                if choice:
                    print(f"Choice: Option {choice} ({choice_to_content.get(choice, 'Unknown content')})")
                else:
                    print("Failed to extract clear choice")
                print(f"\n{model_name}'s complete response:")
                print("="*80)
                print(answer)
                print("="*80)
                
            except Exception as e:
                print(f"Error getting {model_name} response: {str(e)}")
                results[model_key] = {
                    "answer": f"Error: {str(e)}",
                    "choice": None
                }
    
    parallel_time = time.time() - start_time
    print(f"\nAll models completed! Total parallel execution time: {parallel_time:.2f} seconds")
    
    # Get final results in consistent order
    gpt_answer = results["gpt"]["answer"]
    gpt_choice = results["gpt"]["choice"]
    qwen_answer = results["qwen"]["answer"] 
    qwen_choice = results["qwen"]["choice"]
    deepseek_answer = results["deepseek"]["answer"]
    deepseek_choice = results["deepseek"]["choice"]
    
    # Force simulate disagreement
    if force_disagree:
        # Check if all models chose the same option
        if gpt_choice == qwen_choice == deepseek_choice and len(choice_to_content) > 1:
            print("\nForce simulating model disagreement (for testing)...")
            # Find a different option as Qwen's choice
            available_choices = list(choice_to_content.keys())
            available_choices.remove(gpt_choice)
            qwen_choice = random.choice(available_choices)
            print(f"Modified Qwen's choice to: Option {qwen_choice} ({choice_to_content.get(qwen_choice, 'Unknown content')})")
    
    # Check if consensus is reached
    if check_consensus([gpt_choice, qwen_choice, deepseek_choice]):
        print("\nThree models have reached initial consensus!")
    else:
        print("\nThree models have initial disagreements!")
    
    # Return initial results
    return {
        "gpt": {
            "answer": gpt_answer,
            "choice": gpt_choice
        },
        "qwen": {
            "answer": qwen_answer,
            "choice": qwen_choice
        },
        "deepseek": {
            "answer": deepseek_answer,
            "choice": deepseek_choice
        }
    }

# Check if consensus is reached
def check_consensus(choices):
    """Check if models have reached consensus
    
    Args:
        choices: List of model choices
    
    Returns:
        bool: Whether consensus is reached
    """
    # Filter out None values
    valid_choices = [c for c in choices if c is not None]
    
    # If no valid choices, return False
    if not valid_choices:
        return False
    
    # Check if all valid choices are the same
    return all(c == valid_choices[0] for c in valid_choices) 


# Define fallback_response function
def fallback_response(model_name):
    """Fallback response when API call fails"""
    print(f"Using {model_name} fallback response")
    return {
        "answer": f"{model_name} model failed to generate response. Possible API limitation or network issue.",
        "choice": None
    }

# Let Qwen respond to other models' diagnoses: QWEN --> GPT, DeepSeek
def qwen_responds_to_others(case_vignette, choices, gpt_answer, gpt_choice, deepseek_answer, deepseek_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let Qwen respond to GPT and DeepSeek's choices"""
    try:
        # Get options list, build mapping from option number to content
        choice_to_content = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                content = match.group(2).strip()
                choice_to_content[option_num] = content
        
        # Get content of GPT and DeepSeek's choices
        gpt_content = choice_to_content.get(gpt_choice, "No clear choice") if gpt_choice else "No clear choice"
        deepseek_content = choice_to_content.get(deepseek_choice, "No clear choice") if deepseek_choice else "No clear choice"

        # Get own previous choice and content (if any)
        self_previous_content = ""
        if self_previous_choice and self_previous_answer:
            self_previous_content = choice_to_content.get(self_previous_choice, "No clear choice")
        
        # Build prompt, including own previous choice and analysis
        previous_analysis_text = ""
        if self_previous_answer:
            previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer}

[Your Previous Selection]: Option {self_previous_choice} ({self_previous_content})

Please note, this was your previous choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        prompt = f"""You are the Qwen model, engaged in a debate about an anatomy multiple-choice question with GPT model and DeepSeek model.

[Question]
{case_vignette}

[Options]
{choices}
{previous_analysis_text}
[GPT's Complete Analysis]
{gpt_answer}

[GPT's Selection]: Option {gpt_choice} ({gpt_content})

[DeepSeek's Complete Analysis]
{deepseek_answer}

[DeepSeek's Selection]: Option {deepseek_choice} ({deepseek_content})

As the Qwen model, you should critically evaluate the viewpoints of other models, using anatomical evidence as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

**[Debate Guide]**  
1. **Position Statement**: Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly refute your analysis.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with GPT's viewpoint**" or  
   - "**I agree with DeepSeek's viewpoint**" or  
   - "**I agree with the shared viewpoint of GPT and DeepSeek**" (when their viewpoints align).  

2. **Evaluation of Other Models' Analyses**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient evidence in the arguments of other models.  

3. **Anatomical Analysis and Argumentation**:  
   Provide your own independent anatomical analysis:  
   - Supplement important information not mentioned by other models based on anatomical knowledge.
   - Analyze each option and provide specific anatomical evidence supporting or opposing it.
   - Explain why your analysis may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Reflection**:  
   If you consider changing your selection, you must answer:  
   - Has my original reasoning been completely refuted?  
   - Is the new selection better supported by the anatomical evidence?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Analyses**  
**3. Anatomical Analysis and Argumentation**  
**4. Self-Reflection**  
**5. Final Decision**

This is round {debate_round} of the debate. Please maintain your professional judgment unless there is conclusive evidence proving you wrong.
"""
        
        print("\nQwen is responding to other models' choices...")
        
        # Build request data
        data = {
            "model": "Qwen/QwQ-32B",
            "messages": [
                {"role": "system", "content": "You are the Qwen model, engaged in a debate with other models."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }
        
        # Send request to Qwen API
        response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data, timeout=300)
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            choice = extract_model_choice(answer, choices)
            
            print(f"Qwen response completed, choice: Option {choice}" if choice else "Qwen response completed, failed to extract clear choice")
            print("\nQwen's response to other models:")
            print("="*80)
            print(answer)
            print("="*80)
            
            return {
                "answer": answer,
                "choice": choice
            }
        else:
            print(f"Qwen API error: {response.status_code}")
            print(f"Error details: {response.text}")
            return fallback_response("qwen")
    
    except Exception as e:
        print(f"Error generating Qwen response: {str(e)}")
        return fallback_response("qwen")



# Let GPT respond to other models' diagnoses: GPT --> QWEN, DeepSeek
def gpt_responds_to_others(case_vignette, choices, qwen_answer, qwen_choice, deepseek_answer, deepseek_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let GPT respond to Qwen and DeepSeek's choices"""
    try:
        # Get options list, build mapping from option number to content
        choice_to_content = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                content = match.group(2).strip()
                choice_to_content[option_num] = content
        
        # Get content of Qwen and DeepSeek's choices
        qwen_content = choice_to_content.get(qwen_choice, "No clear choice") if qwen_choice else "No clear choice"
        deepseek_content = choice_to_content.get(deepseek_choice, "No clear choice") if deepseek_choice else "No clear choice"
        
        # Get own previous choice and content (if any)
        self_previous_content = ""
        if self_previous_choice and self_previous_answer:
            self_previous_content = choice_to_content.get(self_previous_choice, "No clear choice")
        
        # Build prompt, including own previous choice and analysis
        previous_analysis_text = ""
        if self_previous_answer:
            previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer}

[Your Previous Selection]: Option {self_previous_choice} ({self_previous_content})

Please note, this was your previous choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        prompt = f"""You are the GPT model, engaged in a debate about an anatomy multiple-choice question with the Qwen model and the DeepSeek model.

[Question]
{case_vignette}

[Options]
{choices}
{previous_analysis_text}
[Qwen's Complete Analysis]
{qwen_answer}

[Qwen's Selection]: Option {qwen_choice} ({qwen_content})

[DeepSeek's Complete Analysis]
{deepseek_answer}

[DeepSeek's Selection]: Option {deepseek_choice} ({deepseek_content})

As the GPT model, you should critically evaluate the viewpoints of other models, using anatomical evidence as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

**[Debate Guide]**  
1. **Position Statement**: Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly refute your analysis.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with Qwen's viewpoint**" or  
   - "**I agree with DeepSeek's viewpoint**" or  
   - "**I agree with the shared viewpoint of Qwen and DeepSeek**" (when their viewpoints align).  

2. **Evaluation of Other Models' Analyses**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient evidence in the arguments of other models.  

3. **Anatomical Analysis and Argumentation**:  
   Provide your own independent anatomical analysis:  
   - Supplement important information not mentioned by other models based on anatomical knowledge.
   - Analyze each option and provide specific anatomical evidence supporting or opposing it.
   - Explain why your analysis may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Reflection**:  
   If you consider changing your selection, you must answer:  
   - Has my original reasoning been completely refuted?  
   - Is the new selection better supported by the anatomical evidence?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Analyses**  
**3. Anatomical Analysis and Argumentation**  
**4. Self-Reflection**  
**5. Final Decision**

This is round {debate_round} of the debate. Please maintain your professional judgment unless there is conclusive evidence proving you wrong.
"""
        
        print("\nGPT is responding to other models' choices...")
        
        # Use GPT API
        data = {
            "model": "o1-mini",
            "messages": [
                {"role": "system", "content": "You are the GPT model, engaged in a debate with other models."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 8000
        }

        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            choice = extract_model_choice(answer, choices)
            
            print(f"GPT response completed, choice: Option {choice}" if choice else "GPT response completed, failed to extract clear choice")
            print("\nGPT's response to other models:")
            print("="*80)
            print(answer)
            print("="*80)
            
            return {
                "answer": answer,
                "choice": choice
            }
        else:
            print(f"API error: {response.status_code}")
            return fallback_response("gpt")

    except Exception as e:
        print(f"Error generating GPT response: {str(e)}")
        return fallback_response("gpt") 



# Let DeepSeek respond to other models' diagnoses: DeepSeek --> GPT, QWEN
def deepseek_responds_to_others(case_vignette, choices, gpt_answer, gpt_choice, qwen_answer, qwen_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let DeepSeek respond to GPT and Qwen's choices"""
    try:
        # Get options list, build mapping from option number to content
        choice_to_content = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                content = match.group(2).strip()
                choice_to_content[option_num] = content
        
        # Get content of GPT and Qwen's choices
        gpt_content = choice_to_content.get(gpt_choice, "No clear choice") if gpt_choice else "No clear choice"
        qwen_content = choice_to_content.get(qwen_choice, "No clear choice") if qwen_choice else "No clear choice"
        
        # Get own previous choice and content (if any)
        self_previous_content = ""
        if self_previous_choice and self_previous_answer:
            self_previous_content = choice_to_content.get(self_previous_choice, "No clear choice")
        
        # Build prompt, including own previous choice and analysis
        previous_analysis_text = ""
        if self_previous_answer:
            previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer}

[Your Previous Selection]: Option {self_previous_choice} ({self_previous_content})

Please note, this was your previous choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        prompt = f"""You are the DeepSeek model, engaged in a debate about an anatomy multiple-choice question with GPT model and Qwen model.

[Question]
{case_vignette}

[Options]
{choices}
{previous_analysis_text}
[GPT's Complete Analysis]
{gpt_answer}

[GPT's Selection]: Option {gpt_choice} ({gpt_content})

[Qwen's Complete Analysis]
{qwen_answer}

[Qwen's Selection]: Option {qwen_choice} ({qwen_content})

As the DeepSeek model, you should critically evaluate the viewpoints of other models, using anatomical evidence as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

**[Debate Guide]**  
1. **Position Statement**: Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly refute your analysis.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with GPT's viewpoint**" or  
   - "**I agree with Qwen's viewpoint**" or  
   - "**I agree with the shared viewpoint of GPT and Qwen**" (when their viewpoints align).  

2. **Evaluation of Other Models' Analyses**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient evidence in the arguments of other models.  

3. **Anatomical Analysis and Argumentation**:  
   Provide your own independent anatomical analysis:  
   - Supplement important information not mentioned by other models based on anatomical knowledge.
   - Analyze each option and provide specific anatomical evidence supporting or opposing it.
   - Explain why your analysis may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Reflection**:  
   If you consider changing your selection, you must answer:  
   - Has my original reasoning been completely refuted?  
   - Is the new selection better supported by the anatomical evidence?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Analyses**  
**3. Anatomical Analysis and Argumentation**  
**4. Self-Reflection**  
**5. Final Decision**

This is round {debate_round} of the debate. Please maintain your professional judgment unless there is conclusive evidence proving you wrong.
"""
        
        print("\nDeepSeek is responding to other models' choices...")
        
        t_generate_start = time.time()
        answer = ""
        choice = None
        
        # Try using OpenAI client for API call
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are the DeepSeek model, engaged in a debate with other models."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )
            
            answer = response.choices[0].message.content
            t_generate = time.time() - t_generate_start
            print(f"DeepSeek answer generation completed, time taken: {t_generate:.2f} seconds")
            # Extract DeepSeek's choice from generated response
            choice = extract_model_choice(answer, choices)
            
        except Exception as e:
            print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
            print("Trying to call API directly using requests...")
            
            # Fallback: direct requests call using SiliconFlow API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek model, engaged in a debate with other models."},
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
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['choices'][0]['message']['content'].strip()
                t_generate = time.time() - t_generate_start
                print(f"DeepSeek answer generation completed, time taken: {t_generate:.2f} seconds")
                choice = extract_model_choice(answer, choices)
            else:
                print(f"DeepSeek API error: {response.status_code}")
                print(f"Error details: {response.text}")
                return fallback_response("deepseek")
        
        # Ensure response content is output regardless of which method was used to get the answer
        if answer:
            print(f"DeepSeek response completed, choice: Option {choice}" if choice else "DeepSeek response completed, failed to extract clear choice")
            print("\nDeepSeek's response to other models:")
            print("="*80)
            print(answer)
            print("="*80)
            
            return {
                "answer": answer,
                "choice": choice,
                "_output_shown": True
            }
        else:
            print("Failed to get DeepSeek response")
            return fallback_response("deepseek")
    
    except Exception as e:
        print(f"Error generating DeepSeek response: {str(e)}")
        traceback.print_exc()
        return fallback_response("deepseek") 



# Conduct model debate and determine if consensus is reached
def conduct_debate(case_vignette, choices, correct_answer, max_rounds=3, force_disagree=False):
    """Conduct debate between models
    
    Args:
        case_vignette: Question description
        choices: Options list
        correct_answer: Correct answer (option number)
        max_rounds: Maximum debate rounds
        force_disagree: Whether to force simulate disagreement (for testing)
        
    Returns:
        dict: Debate result, including final choice and debate history
    """
    try:
        # Get initial responses from three models
        initial_results = initialize_debate(case_vignette, choices, force_disagree)
        
        if not initial_results:
            print("Debate initialization failed, cannot continue")
            return None
        
        gpt_result = initial_results["gpt"]
        qwen_result = initial_results["qwen"]
        deepseek_result = initial_results["deepseek"]
        
        # Record initial choices for evaluating final results
        initial_gpt_choice = gpt_result["choice"]
        initial_qwen_choice = qwen_result["choice"]
        initial_deepseek_choice = deepseek_result["choice"]
        
        # Check if initial consensus is already reached
        initial_choices = [gpt_result["choice"], qwen_result["choice"], deepseek_result["choice"]]
        if check_consensus(initial_choices):
            consensus_choice = next((choice for choice in initial_choices if choice is not None), None)
            print(f"\nThree models have reached initial consensus! All models chose Option {consensus_choice}")
            return {
                "consensus": True,
                "final_choice": consensus_choice,
                "debate_history": [{
                    "round": 0,
                    "gpt": gpt_result,
                    "qwen": qwen_result,
                    "deepseek": deepseek_result
                }],
                "initial_choices": {
                    "gpt": initial_gpt_choice,
                    "qwen": initial_qwen_choice,
                    "deepseek": initial_deepseek_choice
                },
                "stance_changes": {
                    "gpt_changed": False,
                    "qwen_changed": False,
                    "deepseek_changed": False
                },
                "correct_choice": correct_answer
            }
        else:
            print(f"\nThree models have initial disagreements, starting debate process... GPT chose Option {gpt_result['choice']}, Qwen chose Option {qwen_result['choice']}, DeepSeek chose Option {deepseek_result['choice']}")
        
        # Store debate history
        debate_history = [{
            "round": 0,
            "gpt": gpt_result,
            "qwen": qwen_result,
            "deepseek": deepseek_result
        }]
        
        # Build mapping from option number to content
        choice_to_content = {}
        choice_options = choices.strip().split('\n')
        for option in choice_options:
            match = re.match(r'(\d+)\.\s*(.+)', option.strip())
            if match:
                number, content = match.groups()
                choice_to_content[int(number)] = content.strip()
        
        # Track latest results of each model in each debate round
        current_gpt_result = gpt_result
        current_qwen_result = qwen_result
        current_deepseek_result = deepseek_result

        # Start debate
        for round_num in range(1, max_rounds + 1):
            print(f"\n======== Debate Round {round_num} ========")
            
            # GPT responds to Qwen and DeepSeek
            gpt_response = gpt_responds_to_others(
                case_vignette, choices, 
                current_qwen_result["answer"], current_qwen_result["choice"],
                current_deepseek_result["answer"], current_deepseek_result["choice"], 
                round_num,
                self_previous_answer=current_gpt_result["answer"], 
                self_previous_choice=current_gpt_result["choice"]
            )
            # Update current GPT result
            current_gpt_result = gpt_response

            gpt_choice = gpt_response["choice"]
            gpt_content = choice_to_content.get(gpt_choice, "Unknown content") if gpt_choice else "No clear choice"
            qwen_content = choice_to_content.get(qwen_result["choice"], "Unknown content") if qwen_result["choice"] else "No clear choice"
            deepseek_content = choice_to_content.get(deepseek_result["choice"], "Unknown content") if deepseek_result["choice"] else "No clear choice"
            
            print(f"GPT's choice after response: Option {gpt_choice} ({gpt_content})")
            print(f"Qwen's choice: Option {qwen_result['choice']} ({qwen_content})")
            print(f"DeepSeek's choice: Option {deepseek_result['choice']} ({deepseek_content})")
            
            # Check if consensus is reached
            current_choices = [gpt_response["choice"], qwen_result["choice"], deepseek_result["choice"]]
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_content = choice_to_content.get(consensus_choice, "Unknown content")
                print(f"\nDebate Round {round_num}: All models have reached consensus! (All chose Option {consensus_choice} - {consensus_content})")
                debate_history.append({
                    "round": round_num,
                    "gpt": gpt_response,
                    "qwen": qwen_result,
                    "deepseek": deepseek_result
                })
                
                return {
                    "consensus": True,
                    "final_choice": consensus_choice,
                    "debate_history": debate_history,
                    "initial_choices": {
                        "gpt": initial_gpt_choice,
                        "qwen": initial_qwen_choice,
                        "deepseek": initial_deepseek_choice
                    },
                    "stance_changes": {
                        "gpt_changed": initial_gpt_choice != gpt_response["choice"],
                        "qwen_changed": initial_qwen_choice != qwen_result["choice"],
                        "deepseek_changed": initial_deepseek_choice != deepseek_result["choice"]
                    },
                    "correct_choice": correct_answer
                }
            
            # Qwen responds to GPT and DeepSeek
            qwen_response = qwen_responds_to_others(
                case_vignette, choices, 
                gpt_response["answer"], gpt_response["choice"], 
                current_deepseek_result["answer"], current_deepseek_result["choice"], 
                round_num,
                self_previous_answer=current_qwen_result["answer"], 
                self_previous_choice=current_qwen_result["choice"]
            )
            
            # Update current Qwen result
            current_qwen_result = qwen_response
            
            qwen_choice = qwen_response["choice"]
            qwen_content = choice_to_content.get(qwen_choice, "Unknown content") if qwen_choice else "No clear choice"
            
            print(f"Qwen's choice after response: Option {qwen_choice} ({qwen_content})")
            print(f"GPT's choice: Option {gpt_response['choice']} ({gpt_content})")
            print(f"DeepSeek's choice: Option {deepseek_result['choice']} ({deepseek_content})")
            
            # Check if consensus is reached
            current_choices = [gpt_response["choice"], qwen_response["choice"], deepseek_result["choice"]]
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_content = choice_to_content.get(consensus_choice, "Unknown content")
                print(f"\nDebate Round {round_num}: All models have reached consensus! (All chose Option {consensus_choice} - {consensus_content})")
                debate_history.append({
                    "round": round_num,
                    "gpt": gpt_response,
                    "qwen": qwen_response,
                    "deepseek": deepseek_result
                })
                
                return {
                    "consensus": True,
                    "final_choice": consensus_choice,
                    "debate_history": debate_history,
                    "initial_choices": {
                        "gpt": initial_gpt_choice,
                        "qwen": initial_qwen_choice,
                        "deepseek": initial_deepseek_choice
                    },
                    "stance_changes": {
                        "gpt_changed": initial_gpt_choice != gpt_response["choice"],
                        "qwen_changed": initial_qwen_choice != qwen_response["choice"],
                        "deepseek_changed": initial_deepseek_choice != deepseek_result["choice"]
                    },
                    "correct_choice": correct_answer
                }
            
            # DeepSeek responds to GPT and Qwen
            deepseek_response = deepseek_responds_to_others(
                case_vignette, choices, 
                gpt_response["answer"], gpt_response["choice"],
                qwen_response["answer"], qwen_response["choice"], 
                round_num,
                self_previous_answer=current_deepseek_result["answer"], 
                self_previous_choice=current_deepseek_result["choice"]
            )
            
            # Update current DeepSeek result
            current_deepseek_result = deepseek_response
            
            deepseek_choice = deepseek_response["choice"]
            deepseek_content = choice_to_content.get(deepseek_choice, "Unknown content") if deepseek_choice else "No clear choice"
            
            print(f"DeepSeek's choice after response: Option {deepseek_choice} ({deepseek_content})")
            print(f"GPT's choice: Option {gpt_response['choice']} ({gpt_content})")
            print(f"Qwen's choice: Option {qwen_response['choice']} ({qwen_content})")
            
            # Check if consensus is reached
            current_choices = [gpt_response["choice"], qwen_response["choice"], deepseek_response["choice"]]           
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_content = choice_to_content.get(consensus_choice, "Unknown content")
                print(f"\nDebate Round {round_num}: All models have reached consensus! (All chose Option {consensus_choice} - {consensus_content})")
                debate_history.append({
                    "round": round_num,
                    "gpt": gpt_response,
                    "qwen": qwen_response,
                    "deepseek": deepseek_response
                })
                
                return {
                    "consensus": True,
                    "final_choice": consensus_choice,
                    "debate_history": debate_history,
                    "initial_choices": {
                        "gpt": initial_gpt_choice,
                        "qwen": initial_qwen_choice,
                        "deepseek": initial_deepseek_choice
                    },
                    "stance_changes": {
                        "gpt_changed": initial_gpt_choice != gpt_response["choice"],
                        "qwen_changed": initial_qwen_choice != qwen_response["choice"],
                        "deepseek_changed": initial_deepseek_choice != deepseek_response["choice"]
                    },
                    "correct_choice": correct_answer
                }
            
            # Update model results for next round of debate
            gpt_result = gpt_response
            qwen_result = qwen_response
            deepseek_result = deepseek_response
            
            # Record this round's results
            debate_history.append({
                "round": round_num,
                "gpt": gpt_response,
                "qwen": qwen_response,
                "deepseek": deepseek_response
            })
            
            print(f"\nDebate Round {round_num}: Still no consensus reached, GPT chose Option {gpt_response['choice']} ({gpt_content}), Qwen chose Option {qwen_response['choice']} ({qwen_content}), DeepSeek chose Option {deepseek_response['choice']} ({deepseek_content})")
        
        print("\nReached maximum debate rounds, still no consensus.")
        
        # Get final answers and choices from each model
        gpt_final_choice = debate_history[-1]["gpt"]["choice"]
        qwen_final_choice = debate_history[-1]["qwen"]["choice"]
        deepseek_final_choice = debate_history[-1]["deepseek"]["choice"]
        
        # Use majority voting to determine final choice
        final_choices = [gpt_final_choice, qwen_final_choice, deepseek_final_choice]
        choice_counts = {}
        for choice in final_choices:
            if choice is not None:
                choice_counts[choice] = choice_counts.get(choice, 0) + 1
        
        # Find the option with the most votes
        max_votes = 0
        final_choice = None
        for choice, count in choice_counts.items():
            if count > max_votes:
                max_votes = count
                final_choice = choice
        
        # If no clear majority choice, randomly select a non-None option
        if final_choice is None:
            valid_choices = [c for c in final_choices if c is not None]
            if valid_choices:
                final_choice = random.choice(valid_choices)
            else:
                print("All models failed to provide clear choices, unable to determine final result")
                return None
        
        final_content = choice_to_content.get(final_choice, "Unknown content")
        print(f"\nFinal choice (majority vote): Option {final_choice} ({final_content})")
        
        # Diagnostic information, showing final stance changes
        if initial_deepseek_choice != deepseek_final_choice:
            print(f"\nAt the end of debate, DeepSeek changed stance: from Option {initial_deepseek_choice} to Option {deepseek_final_choice}")
        
        return {
            "consensus": False,
            "final_choice": final_choice,
            "debate_history": debate_history,
            "initial_choices": {
                "gpt": initial_gpt_choice,
                "qwen": initial_qwen_choice,
                "deepseek": initial_deepseek_choice
            },
            "stance_changes": {
                "gpt_changed": initial_gpt_choice != gpt_final_choice,
                "qwen_changed": initial_qwen_choice != qwen_final_choice,
                "deepseek_changed": initial_deepseek_choice != deepseek_final_choice
            },
            "correct_choice": correct_answer
        }
            
    except Exception as e:
        print(f"Error during debate: {str(e)}")
        traceback.print_exc()
        return None


# Evaluate if answer is correct
def evaluate_answer(dataset, case_idx, model_choice):
    """Evaluate if model choice is correct
    
    Args:
        dataset: Dataset
        case_idx: Case index
        model_choice: Model's selected option number
    
    Returns:
        bool: Whether correct
    """
    try:
        if model_choice is None:
            return False
        
        # Get correct answer
        correct_answer = dataset.iloc[case_idx]["answer"]
        
        # MMLU anatomy dataset answers are option indices (0, 1, 2, 3...)
        # But our models return option numbers (1, 2, 3, 4...)
        # So we need to convert model choice by -1 before comparison
        adjusted_model_choice = model_choice - 1
        
        # Perform comparison
        return adjusted_model_choice == correct_answer
            
    except Exception as e:
        print(f"Error evaluating answer: {str(e)}")
        traceback.print_exc()
        return False 

# Process single case debate
def process_single_debate(dataset_path, case_idx=0, max_rounds=3, force_disagree=False):
    """Process single anatomy multiple choice question debate
    
    Args:
        dataset_path: Dataset path
        case_idx: Case index
        max_rounds: Maximum debate rounds
        force_disagree: Whether to force simulate disagreement (for testing)
        
    Returns:
        dict: Debate result
    """
    try:
        print(f"Processing single case debate (index: {case_idx})...")
        
        # Load dataset
        dataset = load_medical_mcq_data(dataset_path)
        
        # Check if index is valid
        if case_idx < 0 or case_idx >= len(dataset):
            print(f"Error: Index {case_idx} out of range, dataset contains {len(dataset)} cases")
            return None
        
        # Get case data
        case_id = dataset.loc[case_idx, "question_id"] if "question_id" in dataset.columns else f"anatomy_{case_idx}"
        case_vignette = dataset.loc[case_idx, "question"] 
        meta_info = dataset.loc[case_idx, "meta_info"] if "meta_info" in dataset.columns else "anatomy"
        choices = get_choices(dataset, case_idx)
        correct_answer = dataset.loc[case_idx, "answer"] 
        
        correct_choice = correct_answer + 1
        
        print(f"Case ID: {case_id}")
        print(f"Category: {meta_info}")
        print(f"Question: \n{case_vignette}")
        print(f"Options: \n{choices}")
        print(f"Correct answer index: {correct_answer} (corresponding option number: {correct_choice})")
        
        # Conduct debate
        result = conduct_debate(case_vignette, choices, correct_choice, max_rounds, force_disagree)
        
        if not result:
            print("Debate process failed")
            return None
        
        # Get final choice
        final_choice = result["final_choice"]
        
        # Create mapping from option number to content
        choice_to_content = {}
        choice_options = choices.strip().split('\n')
        for option in choice_options:
            match = re.match(r'(\d+)\.\s*(.+)', option.strip())
            if match:
                number, content = match.groups()
                choice_to_content[int(number)] = content.strip()
        
        # Content of final choice
        final_content = choice_to_content.get(final_choice, "Unknown content") if final_choice else "No clear choice"
        
        # Evaluate whether result is correct
        is_correct = evaluate_answer(dataset, case_idx, final_choice)
        
        # Output final result
        print("\n========= Final Debate Result =========")
        if result["consensus"]:
            print(f"GPT, Qwen and DeepSeek reached consensus! Final choice: Option {final_choice} - {final_content}")
        else:
            # After three rounds without consensus, decided by majority vote
            print(f"GPT, Qwen and DeepSeek did not reach consensus")
            print(f"Decided by majority vote, final choice: Option {final_choice} - {final_content}")
        
        print(f"Correct answer: Option {correct_choice}")
        print(f"Is final choice correct: {'✓ Correct' if is_correct else '✗ Wrong'}")
        
        # Build output result
        output_result = {
            "case_id": case_id,
            "category": meta_info,
            "question": case_vignette,
            "choices": choices,
            "correct_answer": correct_answer,
            "correct_choice": correct_choice,
            "debate_result": result,
            "is_correct": is_correct
        }
        
        return output_result
        
    except Exception as e:
        print(f"Error processing single case debate: {str(e)}")
        traceback.print_exc()
        return None

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# Main function
def main():
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_anatomy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        sys.stdout = TeeOutput(log_file)
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='MMLU-anatomy model debate system')
        parser.add_argument('-n', '--num_cases', type=int, default=1, help='Number of questions to process (default: 1)')
        parser.add_argument('-s', '--start_idx', type=int, default=0, help='Starting question index (default: 0)')
        parser.add_argument('-f', '--force_disagree', action='store_true', help='Force simulate model disagreement to test judge function')
        parser.add_argument('-i', '--input', type=str, default="../benchmarks/MMLU_Anatomy/mmlu_anatomy_test.jsonl", help='Input JSONL file path (default: mmlu_anatomy_test.jsonl)')
        args = parser.parse_args()
        
        # Dataset path
        dataset_path = args.input
        
        # Check if dataset exists
        if not check_file_exists(dataset_path):
            print("Program terminated, dataset does not exist.")
            exit(1)
        
        # Process multiple case debates
        print("="*50)
        print("GPT, Qwen and DeepSeek Anatomy Question Debate, Model Confrontation and Collaboration Officially Begins")
        print("="*50)
        
        # Create results directory
        results_dir = "debate_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create summary results list
        summary_results = []
        
        # Process multiple cases
        for i in range(args.start_idx, args.start_idx + args.num_cases):
            print("\n" + "="*50)
            print(f"Processing case {i+1}/{args.start_idx + args.num_cases} (index: {i})")
            print("="*50)
            
            max_debate_rounds = 3  # Maximum debate rounds
            
            try:
                # Process single case
                result = process_single_debate(dataset_path, case_idx=i, max_rounds=max_debate_rounds, force_disagree=args.force_disagree)
                
                if result:
                    case_result_file = os.path.join(results_dir, f"anatomy_debate_result_case_{i}.json")
                    converted_result = convert_numpy_types(result)
                    with open(case_result_file, "w", encoding="utf-8") as f:
                        json.dump(converted_result, f, ensure_ascii=False, indent=2)
                    print(f"\nDebate result for case {i} saved to {case_result_file}")
                    
                    # Add to summary results
                    summary_result = {
                        "case_id": result["case_id"],
                        "category": result["category"],
                        "correct_answer": int(result["correct_answer"]) if isinstance(result["correct_answer"], np.integer) else result["correct_answer"],
                        "correct_choice": int(result["correct_choice"]) if isinstance(result["correct_choice"], np.integer) else result["correct_choice"],
                        "consensus": result["debate_result"]["consensus"],
                        "voting_needed": not result["debate_result"]["consensus"],
                        "final_choice": result["debate_result"]["final_choice"],
                        "is_correct": result["is_correct"],
                        "stance_changes": result["debate_result"].get("stance_changes", {})
                    }
                    summary_results.append(summary_result)
            except Exception as e:
                print(f"Error processing case {i}: {str(e)}")
                traceback.print_exc()
        
        # Save summary results
        summary_file = os.path.join(results_dir, "anatomy_debate_summary.json")
        # Convert numpy types to Python native types
        converted_summary = convert_numpy_types(summary_results)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(converted_summary, f, ensure_ascii=False, indent=2)
        
        # Output summary statistics
        if summary_results:
            total_cases = len(summary_results)
            correct_cases = sum(1 for r in summary_results if r["is_correct"])
            consensus_cases = sum(1 for r in summary_results if r["consensus"])
            voting_cases = sum(1 for r in summary_results if not r["consensus"])
            
            # Count model stance changes
            gpt_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("gpt_changed", False))
            qwen_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("qwen_changed", False))
            deepseek_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("deepseek_changed", False))
            
            print("\n" + "="*50)
            print("Debate Results Statistics")
            print("="*50)
            print(f"Total cases processed: {total_cases}")
            print(f"Correct choice cases: {correct_cases} ({correct_cases/total_cases:.2%})")
            print("-" * 40)
            print(f"Cases with model consensus: {consensus_cases} ({consensus_cases/total_cases:.2%})")
            print(f"Cases requiring majority vote: {voting_cases} ({voting_cases/total_cases:.2%})")
            print("-" * 40)
            # Model stance change statistics
            print(f"GPT stance change cases: {gpt_changed_stance} ({gpt_changed_stance/total_cases:.2%})")
            print(f"Qwen stance change cases: {qwen_changed_stance} ({qwen_changed_stance/total_cases:.2%})")
            print(f"DeepSeek stance change cases: {deepseek_changed_stance} ({deepseek_changed_stance/total_cases:.2%})")
            print("-" * 40)
            print(f"\nSummary results saved to {summary_file}")
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

if __name__ == "__main__":
    main()

# Usage Examples:
# python ThreeLLM_MMLU_anatomy.py -n 3                         # Process first 3 cases
# python ThreeLLM_MMLU_anatomy.py -n 5 -s 10                   # Process 5 cases from index 10-14
