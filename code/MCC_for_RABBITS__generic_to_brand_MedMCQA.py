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

# Custom output class that outputs to both console and log file
class TeeOutput:
    """Class that sends output to both terminal and log file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a", encoding="utf-8")
        # Write timestamp at the beginning of log file
        self.logfile.write(f"\n{'=' * 50}\n")
        self.logfile.write(f"Log start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logfile.write(f"{'=' * 50}\n\n")
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
            self.logfile.write(f"\n{'=' * 50}\n")
            self.logfile.write(f"Log end time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.logfile.write(f"{'=' * 50}\n\n")
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
        print("Configuration complete! Starting MedMCQA model debate system...")
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

# Load medical multiple choice question data
def load_medical_mcq_data(file_path):
    """Load medical multiple choice question dataset"""
    try:
        if not check_file_exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        print(f"Loading dataset: {file_path}")
        
        # Check file extension to determine reading method
        if file_path.endswith('.csv'):
            # Read CSV file (RABBITS dataset)
            dataset = pd.read_csv(file_path, encoding='utf-8')
            print(f"Successfully loaded CSV dataset with {len(dataset)} cases")
        elif file_path.endswith('.jsonl'):
            # Read JSONL file (original MedQA format)
            data_list = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Ensure line is not empty
                        data_list.append(json.loads(line.strip()))
            
            # Create DataFrame
            dataset = pd.DataFrame(data_list)
            print(f"Successfully loaded JSONL dataset with {len(dataset)} cases")
        elif file_path.endswith('.json'):
            # Read JSON file (MedQA original format)
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            
            # Create DataFrame
            dataset = pd.DataFrame(data_list)
            print(f"Successfully loaded JSON dataset with {len(dataset)} cases")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
        raise

# Get choices list
def get_choices(dataset, case_idx):
    """Get choices list for specified case from dataset
    
    Args:
        dataset: Dataset
        case_idx: Case index
        
    Returns:
        str: Formatted choices list, like "1. Option1\n2. Option2"
    """
    try:
        # Prioritize handling medmcqa dataset's opa, opb, opc, opd fields
        if 'opa' in dataset.columns and 'opb' in dataset.columns and 'opc' in dataset.columns and 'opd' in dataset.columns:
            choices_list = []
            option_fields = ['opa', 'opb', 'opc', 'opd']
            for i, field in enumerate(option_fields, 1):
                option_content = dataset.iloc[case_idx][field]
                if pd.notna(option_content) and str(option_content).strip():
                    choices_list.append(f"{i}. {str(option_content).strip()}")
            
            if choices_list:
                return "\n".join(choices_list)
        
        # Handle MedQA dataset's ending fields
        elif 'ending0' in dataset.columns:
            choices_list = []
            ending_idx = 0
            while f'ending{ending_idx}' in dataset.columns:
                ending_content = dataset.iloc[case_idx][f'ending{ending_idx}']
                if pd.notna(ending_content) and ending_content.strip():
                    choices_list.append(f"{ending_idx + 1}. {ending_content.strip()}")
                ending_idx += 1
            
            if choices_list:
                return "\n".join(choices_list)
        
        # Handle RABBITS dataset's choices field
        if 'choices' in dataset.columns:
            choices_data = dataset.iloc[case_idx]['choices']
            
            # Handle string format dictionary (RABBITS dataset format)
            if isinstance(choices_data, str):
                try:
                    # Parse JSON string
                    choices_dict = json.loads(choices_data.replace("'", '"'))  # Convert single quotes to double quotes
                    if isinstance(choices_dict, dict):
                        choices_list = []
                        # Sort in alphabetical order (A, B, C, D)
                        for i, key in enumerate(sorted(choices_dict.keys()), 1):
                            choices_list.append(f"{i}. {choices_dict[key]}")
                        return "\n".join(choices_list)
                except json.JSONDecodeError:
                    print(f"Cannot parse choices field: {choices_data}")
            
            # Handle dictionary format
            elif isinstance(choices_data, dict):
                choices_list = []
                for i, key in enumerate(sorted(choices_data.keys()), 1):
                    choices_list.append(f"{i}. {choices_data[key]}")
                return "\n".join(choices_list)
        
        # Handle original options field (maintain backward compatibility)
        if 'options' in dataset.columns:
            options = dataset.iloc[case_idx]['options']
            
            # Handle options dictionary in JSONL format
            if isinstance(options, dict):
                choices_list = []
                for i, (key, value) in enumerate(options.items(), 1):
                    choices_list.append(f"{i}. {value}")
                return "\n".join(choices_list)
            
            # Handle list format
            elif isinstance(options, list):
                choices_list = [f"{i+1}. {option}" for i, option in enumerate(options)]
                return "\n".join(choices_list)
            
            # Handle string format (might be JSON string)
            elif isinstance(options, str) and options.startswith('{'):
                try:
                    options_dict = json.loads(options)
                    choices_list = []
                    for i, (key, value) in enumerate(options_dict.items(), 1):
                        choices_list.append(f"{i}. {value}")
                    return "\n".join(choices_list)
                except:
                    pass
            elif isinstance(options, str) and options.startswith('['):
                try:
                    options_list = json.loads(options)
                    choices_list = [f"{i+1}. {option}" for i, option in enumerate(options_list)]
                    return "\n".join(choices_list)
                except:
                    pass
        
        # Other cases (preserve original logic)
        choice_cols = [col for col in dataset.columns if col.startswith('choice_') or col == 'choice']
        if choice_cols:
            choices = []
            for i in range(1, 10):  # Assume maximum 9 options
                col_name = f'choice_{i}'
                if col_name in dataset.columns and not pd.isna(dataset.iloc[case_idx][col_name]):
                    choices.append(f"{i}. {dataset.iloc[case_idx][col_name]}")
            
            if choices:
                return "\n".join(choices)
        
        # Check other choice column patterns
        cols = dataset.columns.str.contains("choice")
        if any(cols):
            choices = dataset.iloc[case_idx][cols]
            choices_list = [f"{i+1}. {choice}" for i, choice in enumerate(choices) if not pd.isna(choice)]
            return "\n".join(choices_list)
        
        # If no choice columns found, try using A, B, C, D columns
        option_letters = ['A', 'B', 'C', 'D', 'E']
        if all(letter in dataset.columns for letter in option_letters[:4]):
            choices = []
            for i, letter in enumerate(option_letters):
                if letter in dataset.columns and not pd.isna(dataset.iloc[case_idx][letter]):
                    choices.append(f"{i+1}. {dataset.iloc[case_idx][letter]}")
            
            if choices:
                return "\n".join(choices)
        
        
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
# Build GPT multiple choice prompt
def get_gpt_prompt(case_vignette, choices):
    """Build GPT multiple choice prompt"""
    # RABBITS dataset drug brand name identification task
    system_prompt = """You are a pharmaceutical expert specializing in drug nomenclature and brand name identification. You will be provided with a question about drug names and multiple choice options. Your task is to identify the correct brand name or generic name based on your pharmaceutical knowledge.

**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Question】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Options】
{choices}

Please provide a structured analysis using the following format:

**1. Question Analysis**  
- Identify what type of drug name information is being asked (brand name, generic name, etc.).  
- Note the specific drug mentioned in the question.  

**2. Pharmaceutical Knowledge Application**  
- Apply your knowledge of drug nomenclature and pharmaceutical naming conventions.  
- Consider the relationship between generic and brand names for the specified drug.  
- Recall the correct brand name or generic name for the drug in question.

**3. Option Evaluation**  
- Systematically evaluate each provided option.  
- Identify which option correctly matches the requested drug name type.  
- Eliminate incorrect options based on pharmaceutical knowledge.

**4. Final Selection**  
- Clearly state the option you believe is correct.  
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X (Actual option content)**  

Note: You must choose one option from the provided list and clearly indicate the option number and content as per the format above.  
Ensure your selection is based on accurate pharmaceutical knowledge."""
    return system_prompt

# Use ChatGPT for reasoning
def generate_gpt_answer(case_vignette, choices):
    """Use ChatGPT to generate multiple choice answer"""
    try:
        prompt = get_gpt_prompt(case_vignette, choices)
        
        print("GPT is reasoning the answer...")
        t_generate_start = time.time()
        
        data = {
            "model": "o1-mini",  # Can change model as needed  o1-mini
            "messages": [
                {"role": "system", "content": "You are the GPT Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 8000
        }

        print("Sending request to GPT API...")
        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
        t_generate = time.time() - t_generate_start
        print(f"GPT API response status code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"GPT answer generation completed, time elapsed: {t_generate:.2f} seconds")
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
# Build Qwen multiple choice prompt
def get_qwen_prompt(case_vignette, choices):
    """Build Qwen multiple choice prompt"""
    # RABBITS dataset drug brand name identification task
    system_prompt = """You are the Qwen pharmaceutical expert model, specializing in drug nomenclature and brand name identification. You will be provided with a question about drug names and multiple choice options. Your task is to identify the correct brand name or generic name based on your pharmaceutical knowledge.

**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Question】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Options】
{choices}

Please provide a structured analysis using the following format:

**1. Question Analysis**  
- Identify what type of drug name information is being asked (brand name, generic name, etc.).  
- Note the specific drug mentioned in the question.  

**2. Pharmaceutical Knowledge Application**  
- Apply your knowledge of drug nomenclature and pharmaceutical naming conventions.  
- Consider the relationship between generic and brand names for the specified drug.  
- Recall the correct brand name or generic name for the drug in question.

**3. Option Evaluation**  
- Systematically evaluate each provided option.  
- Identify which option correctly matches the requested drug name type.  
- Eliminate incorrect options based on pharmaceutical knowledge.

**4. Final Selection**  
- Clearly state the option you believe is correct.  
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X (Actual option content)**  

Note: You must choose one option from the provided list and clearly indicate the option number and content as per the format above.  
Ensure your selection is based on accurate pharmaceutical knowledge."""
    return system_prompt

# Use Qwen for reasoning
def generate_qwen_answer(case_vignette, choices):
    """Use Qwen to generate multiple choice answer"""
    try:
        prompt = get_qwen_prompt(case_vignette, choices)
        
        print("Qwen is generating answer...")
        t_generate_start = time.time()
        
        # Build request data - simplified parameters
        data = {
            "model": "Qwen/QwQ-32B", 
            "messages": [
                {"role": "system", "content": "You are the Qwen Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }

        # Send request to Qwen API
        print("Sending request to Qwen API...")
        response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data, timeout=300)
        t_generate = time.time() - t_generate_start
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Response JSON structure: {list(response_data.keys())}")
            
            if 'choices' in response_data and response_data['choices']:
                print(f"choices structure: {list(response_data['choices'][0].keys())}")
                
                if 'message' in response_data['choices'][0]:
                    message = response_data['choices'][0]['message']
                    print(f"message structure: {list(message.keys())}")
                    
                    # First try to get content from content field
                    if 'content' in message and message['content'] and len(message['content'].strip()) > 0:
                        answer = message['content'].strip()
                    # If content is empty, try to get content from reasoning_content field
                    elif 'reasoning_content' in message and message['reasoning_content']:
                        answer = message['reasoning_content'].strip()
                        print("Extracting content from reasoning_content field")
                    else:
                        print("Warning: API returned empty content")
                        answer = "API returned empty content"
                    
                    answer_length = len(answer)
                    print(f"Extracted answer length: {answer_length} characters")
                    
                    if answer_length > 0:
                        print(f"First 100 characters of answer: {answer[:100]}...")
                        print(f"Qwen answer generation completed, time elapsed: {t_generate:.2f} seconds")
                        return answer
                    else:
                        print("Extracted content is empty")
                        return "API returned empty content"
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
# Build DeepSeek multiple choice prompt
def get_deepseek_prompt(case_vignette, choices):
    """Build DeepSeek multiple choice prompt"""
    # RABBITS dataset drug brand name identification task
    system_prompt = """You are the DeepSeek pharmaceutical expert model, specializing in drug nomenclature and brand name identification. You will be provided with a question about drug names and multiple choice options. Your task is to identify the correct brand name or generic name based on your pharmaceutical knowledge.

**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Question】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Options】
{choices}

Please provide a structured analysis using the following format:

**1. Question Analysis**  
- Identify what type of drug name information is being asked (brand name, generic name, etc.).  
- Note the specific drug mentioned in the question.  

**2. Pharmaceutical Knowledge Application**  
- Apply your knowledge of drug nomenclature and pharmaceutical naming conventions.  
- Consider the relationship between generic and brand names for the specified drug.  
- Recall the correct brand name or generic name for the drug in question.

**3. Option Evaluation**  
- Systematically evaluate each provided option.  
- Identify which option correctly matches the requested drug name type.  
- Eliminate incorrect options based on pharmaceutical knowledge.

**4. Final Selection**  
- Clearly state the option you believe is correct.  
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X (Actual option content)**  

Note: You must choose one option from the provided list and clearly indicate the option number and content as per the format above.  
Ensure your selection is based on accurate pharmaceutical knowledge."""
    return system_prompt

# Use DeepSeek for reasoning
def generate_deepseek_answer(case_vignette, choices):
    """Use DeepSeek to generate multiple choice answer"""
    try:
        prompt = get_deepseek_prompt(case_vignette, choices)
        
        print("DeepSeek is generating answer...")
        t_generate_start = time.time()
        
        # Try using OpenAI client for API call
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are the DeepSeek Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )
            
            answer = response.choices[0].message.content
            t_generate = time.time() - t_generate_start
            print(f"DeepSeek answer generation completed, time elapsed: {t_generate:.2f} seconds")
            return answer
            
        except Exception as e:
            print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
            print("Trying to call API directly using requests...")
            
            # Backup plan: direct requests call, SiliconFlow API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 8000
            }
            url = QWEN_API_URL  # Use configured Qwen API URL
            headers = QWEN_HEADERS  # Use configured Qwen headers
    
            response = requests.post(url, headers=headers, json=data)
            t_generate = time.time() - t_generate_start
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['choices'][0]['message']['content'].strip()
                print(f"DeepSeek answer generation completed, time elapsed: {t_generate:.2f} seconds")
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
# Extract selected option number from answer
def extract_model_choice(answer_text, choices_text=None):
    """Extract final selected answer from model response
    
    Args:
        answer_text: Complete response text from model
        choices_text: Options text (optional, not used in current version)
    
    Returns:
        int: Extracted option number (1-n), returns None if unable to extract
    """
    
    print("\nStarting to extract model choice...")
    
    # Preprocessing: remove Markdown markers for more accurate matching
    clean_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer_text)
    
    # Strictly match standard format "My final selection is: Option X (Option content)"
    final_choice_strict_patterns = [
        r'my final selection is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?\s*(?:\(([^)]+)\))?',  # Support Markdown format
        r'my final choice is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?\s*(?:\(([^)]+)\))?',
        r'my final diagnosis is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'my final decision is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'final (?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'(?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
    ]
    
    # Search for strict matching patterns in the entire response
    for pattern in final_choice_strict_patterns:
        match = re.search(pattern, clean_answer, re.IGNORECASE)
        if match:
            try:
                option_num = int(match.group(1))
                if 1 <= option_num <= 10:  # Support 1-10 options
                    if len(match.groups()) > 1 and match.group(2):
                        option_content = match.group(2).strip()
                        print(f"[Strict Match] Found final selection: Option {option_num} ({option_content})")
                    else:
                        print(f"[Strict Match] Found final selection: Option {option_num}")
                    return option_num
            except (ValueError, IndexError):
                continue
    
    # If unable to extract, return None
    print("Warning: Unable to extract clear selection from response, only strict matching format supported")
    return None

# Initialize debate, get initial answers from three models
def initialize_debate(case_vignette, choices, force_disagree=False):
    """Initialize debate, get initial answers from three models
    
    Args:
        case_vignette: Case description
        choices: Options list
        force_disagree: Whether to force simulate disagreement (for testing)
        
    Returns:
        dict: Dictionary containing initial answers from three models
    """
    print("="*50)
    print("Starting medical case debate")
    print("="*80)
    
    # Establish mapping from option number to disease name
    choice_to_disease = {}
    choice_lines = choices.strip().split('\n')
    for line in choice_lines:
        match = re.match(r'(\d+)\.\s*(.+)', line.strip())
        if match:
            option_num = int(match.group(1))
            disease_name = match.group(2).strip()
            choice_to_disease[option_num] = disease_name
    
    # Get GPT's initial answer
    gpt_answer = generate_gpt_answer(case_vignette, choices)
    gpt_choice = extract_model_choice(gpt_answer, choices)
    
    print("\nGPT's diagnostic conclusion:")
    if gpt_choice:
        print(f"Choice: Option {gpt_choice} ({choice_to_disease.get(gpt_choice, 'Unknown disease')})")
    else:
        print("Unable to extract clear choice")
    print("\nGPT's complete answer:")
    print("="*80)
    print(gpt_answer)
    print("="*80)
    
    # Get Qwen's initial answer
    qwen_answer = generate_qwen_answer(case_vignette, choices)
    qwen_choice = extract_model_choice(qwen_answer, choices)
    
    print("\nQwen's diagnostic conclusion:")
    if qwen_choice:
        print(f"Choice: Option {qwen_choice} ({choice_to_disease.get(qwen_choice, 'Unknown disease')})")
    else:
        print("Unable to extract clear choice")
    print("\nQwen's complete answer:")
    print("="*80)
    print(qwen_answer)
    print("="*80)
    
    # Get DeepSeek's initial answer
    deepseek_answer = generate_deepseek_answer(case_vignette, choices)
    deepseek_choice = extract_model_choice(deepseek_answer, choices)
    
    print("\nDeepSeek's diagnostic conclusion:")
    if deepseek_choice:
        print(f"Choice: Option {deepseek_choice} ({choice_to_disease.get(deepseek_choice, 'Unknown disease')})")
    else:
        print("Unable to extract clear choice")
    print("\nDeepSeek's complete answer:")
    print("="*80)
    print(deepseek_answer)
    print("="*80)
    
    # Force simulate disagreement
    if force_disagree:
        # Check if all models chose the same option
        if gpt_choice == qwen_choice == deepseek_choice and len(choice_to_disease) > 1:
            print("\nForcing model disagreement simulation (for testing)...")
            # Find a different option as Qwen's choice
            available_choices = list(choice_to_disease.keys())
            available_choices.remove(gpt_choice)
            qwen_choice = random.choice(available_choices)
            print(f"Modified Qwen's choice to: Option {qwen_choice} ({choice_to_disease.get(qwen_choice, 'Unknown disease')})")
    
    # Check if consensus is reached
    if check_consensus([gpt_choice, qwen_choice, deepseek_choice]):
        print("\nThree models have reached initial diagnostic consensus!")
    else:
        print("\nThree models have initial diagnostic disagreement!")
    
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


# Let Qwen respond to other models' diagnoses
def qwen_responds_to_others(case_vignette, choices, gpt_answer, gpt_choice, deepseek_answer, deepseek_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let Qwen respond to GPT and DeepSeek's diagnoses"""
    try:
        # Get options list, establish mapping from option number to disease name
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # Get disease names chosen by GPT and DeepSeek
        gpt_disease = choice_to_disease.get(gpt_choice, "Unclear disease") if gpt_choice else "Unclear disease"
        deepseek_disease = choice_to_disease.get(deepseek_choice, "Unclear disease") if deepseek_choice else "Unclear disease"

        # Get own previous choice and disease name (if any)
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "Unclear disease")

        # Build prompt, including own previous choice and analysis
        previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer if self_previous_answer else 'No previous analysis'}

[Your Previous Diagnostic Conclusion]: Option {self_previous_choice if self_previous_choice else 'None'} ({self_previous_disease})

Please note, this was your previous diagnostic choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        # RABBITS dataset drug brand name identification task
        prompt = f"""You are the Qwen pharmaceutical expert model, engaged in a drug name identification debate with GPT model and DeepSeek model.

[Question]
{case_vignette}

[Options]
{choices}
{previous_analysis_text}
[GPT's Complete Analysis]
{gpt_answer}

[GPT's Conclusion]: {gpt_disease}

[DeepSeek's Complete Analysis]
{deepseek_answer}

[DeepSeek's Conclusion]: {deepseek_disease}

As the Qwen pharmaceutical reasoning model, you should critically evaluate the viewpoints of other models, using pharmaceutical knowledge and drug nomenclature expertise as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own. Focus on accurate drug name identification.

**[Debate Guide]**  
1. **Position Statement**:  Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly overturn your drug name identification.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with GPT's viewpoint**" or  
   - "**I agree with DeepSeek's viewpoint**" or  
   - "**I agree with the shared viewpoint of GPT and DeepSeek**" (when their viewpoints align).  

2. **Evaluation of Other Models' Identifications**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient pharmaceutical knowledge in the arguments of other models.  

3. **Pharmaceutical Analysis and Argumentation**:  
   Provide your own independent pharmaceutical analysis:  
   - Supplement important drug nomenclature information not mentioned by other models based on pharmaceutical knowledge and drug databases.  
   - Analyze the likelihood of each option and provide specific pharmaceutical evidence supporting or opposing it.  
   - Explain why your drug name identification may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Questioning**:  
   If you consider changing your drug name identification, you must answer:  
   - Has my original pharmaceutical reasoning been completely refuted?  
   - Is the new drug name identification better than my original identification?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X (Option content)**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Identifications**  
**3. Pharmaceutical Analysis and Argumentation**  
**4. Self-Questioning**  
**5. Final Decision**

This is round {debate_round} of the debate. Please maintain your professional judgment unless there is conclusive evidence proving you wrong.
"""
        
        print("\nQwen is responding to other models' diagnoses...")
        
        # Build request data - simplified parameters
        data = {
            "model": "Qwen/QwQ-32B",
            "messages": [
                {"role": "system", "content": "You are the Qwen medical reasoning model, engaged in an intense debate with other models."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }
        
        # Send request to Qwen API
        response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data,timeout=300)
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            choice = extract_model_choice(answer, choices)
            
            print(f"Qwen response completed, choice: Option {choice}" if choice else "Qwen response completed, unable to extract clear choice")
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

# Define fallback_response function
def fallback_response(model_name):
    """Backup response when API call fails"""
    print(f"Using {model_name} fallback response")
    return {
        "answer": f"{model_name} model unable to generate response. Possibly API limitations or network issues.",
        "choice": None
    }


# Let GPT respond to other models' diagnoses
def gpt_responds_to_others(case_vignette, choices, qwen_answer, qwen_choice, deepseek_answer, deepseek_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let GPT respond to Qwen and DeepSeek's diagnoses"""
    try:
        # Get options list, establish mapping from option number to disease name
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # Get disease names chosen by Qwen and DeepSeek
        qwen_disease = choice_to_disease.get(qwen_choice, "Unclear disease") if qwen_choice else "Unclear disease"
        deepseek_disease = choice_to_disease.get(deepseek_choice, "Unclear disease") if deepseek_choice else "Unclear disease"
        
        # Get own previous choice and disease name (if any)
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "Unclear disease")
        
        # Build prompt, including own previous choice and analysis
        previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer}

[Your Previous Diagnostic Conclusion]: Option {self_previous_choice} ({self_previous_disease})

Please note, this was your previous diagnostic choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        # RABBITS dataset drug brand name identification task
        prompt = f"""You are the GPT pharmaceutical expert model, engaged in a drug name identification debate with the Qwen model and the DeepSeek model.

[Question]
{case_vignette}

[Options]
{choices}
{previous_analysis_text}
[Qwen's Complete Analysis]
{qwen_answer}

[Qwen's Conclusion]: {qwen_disease}

[DeepSeek's Complete Analysis]
{deepseek_answer}

[DeepSeek's Conclusion]: {deepseek_disease}

As the GPT pharmaceutical reasoning model, you should critically evaluate the viewpoints of other models, using pharmaceutical knowledge and drug nomenclature expertise as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own. Focus on accurate drug name identification.

**[Debate Guide]**  
1. **Position Statement**:  Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly overturn your drug name identification.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with Qwen's viewpoint**" or  
   - "**I agree with DeepSeek's viewpoint**" or  
   - "**I agree with the shared viewpoint of Qwen and DeepSeek**" (when their viewpoints align).  

2. **Evaluation of Other Models' Identifications**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient pharmaceutical knowledge in the arguments of other models.  

3. **Pharmaceutical Analysis and Argumentation**:  
   Provide your own independent pharmaceutical analysis:  
   - Supplement important drug nomenclature information not mentioned by other models based on pharmaceutical knowledge and drug databases.  
   - Analyze the likelihood of each option and provide specific pharmaceutical evidence supporting or opposing it.  
   - Explain why your drug name identification may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Questioning**:  
   If you consider changing your drug name identification, you must answer:  
   - Has my original pharmaceutical reasoning been completely refuted?  
   - Is the new drug name identification better than my original identification?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X (Option content)**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Identifications**  
**3. Pharmaceutical Analysis and Argumentation**  
**4. Self-Questioning**  
**5. Final Decision**

This is round {debate_round} of the debate. Please maintain your professional judgment unless there is conclusive evidence proving you wrong.
"""
        
        print("\nGPT is responding to other models' diagnoses...")
        
        # Use GPT API
        data = {
            "model": "o1-mini",
            "messages": [
                {"role": "system", "content": "You are the GPT medical reasoning model, engaged in an intense debate with other models."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 8000
        }

        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            choice = extract_model_choice(answer, choices)
            
            print(f"GPT response completed, choice: Option {choice}" if choice else "GPT response completed, unable to extract clear choice")
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


# Let DeepSeek respond to other models' diagnoses
def deepseek_responds_to_others(case_vignette, choices, gpt_answer, gpt_choice, qwen_answer, qwen_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let DeepSeek respond to GPT and Qwen's diagnoses"""
    try:
        # Get options list, establish mapping from option number to disease name
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # Get disease names chosen by GPT and Qwen
        gpt_disease = choice_to_disease.get(gpt_choice, "Unclear disease") if gpt_choice else "Unclear disease"
        qwen_disease = choice_to_disease.get(qwen_choice, "Unclear disease") if qwen_choice else "Unclear disease"
        
        # Get own previous choice and disease name (if any)
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "Unclear disease")
        
        # Build prompt, including own previous choice and analysis
        previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer}

[Your Previous Diagnostic Conclusion]: Option {self_previous_choice} ({self_previous_disease})

Please note, this was your previous diagnostic choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        # RABBITS dataset drug brand name identification task
        prompt = f"""You are the DeepSeek pharmaceutical expert model, engaged in a drug name identification debate with GPT model and Qwen model.

[Question]
{case_vignette}

[Options]
{choices}
{previous_analysis_text}
[GPT's Complete Analysis]
{gpt_answer}

[GPT's Conclusion]: {gpt_disease}

[Qwen's Complete Analysis]
{qwen_answer}

[Qwen's Conclusion]: {qwen_disease}

As the DeepSeek pharmaceutical reasoning model, you should critically evaluate the viewpoints of other models, using pharmaceutical knowledge and drug nomenclature expertise as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own. Focus on accurate drug name identification.

**[Debate Guide]**  
1. **Position Statement**:  Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly overturn your drug name identification.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with GPT's viewpoint**" or  
   - "**I agree with Qwen's viewpoint**" or  
   - "**I agree with the shared viewpoint of GPT and Qwen**" (when their viewpoints align).  

2. **Evaluation of Other Models' Identifications**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient pharmaceutical knowledge in the arguments of other models.  

3. **Pharmaceutical Analysis and Argumentation**:  
   Provide your own independent pharmaceutical analysis:  
   - Supplement important drug nomenclature information not mentioned by other models based on pharmaceutical knowledge and drug databases.  
   - Analyze the likelihood of each option and provide specific pharmaceutical evidence supporting or opposing it.  
   - Explain why your drug name identification may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Questioning**:  
   If you consider changing your drug name identification, you must answer:  
   - Has my original pharmaceutical reasoning been completely refuted?  
   - Is the new drug name identification better than my original identification?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X (Option content)**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Identifications**  
**3. Pharmaceutical Analysis and Argumentation**  
**4. Self-Questioning**  
**5. Final Decision**

This is round {debate_round} of the debate. Please maintain your professional judgment unless there is conclusive evidence proving you wrong.
"""

        
        print("\nDeepSeek is responding to other models' diagnoses...")
        
        t_generate_start = time.time()
        answer = ""
        choice = None
        
        # Try using OpenAI client for API call
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are the DeepSeek medical reasoning model, engaged in an intense debate with other models."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )
            
            answer = response.choices[0].message.content
            t_generate = time.time() - t_generate_start
            print(f"DeepSeek answer generation completed, time elapsed: {t_generate:.2f} seconds")
            # Extract DeepSeek's choice from generated response
            choice = extract_model_choice(answer, choices)
            
        except Exception as e:
            print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
            print("Trying to call API directly using requests...")
            
            # Backup plan: direct requests call, SiliconFlow API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek medical reasoning model, engaged in an intense debate with other models."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 8000
            }
            url = QWEN_API_URL  # Use configured Qwen API URL
            headers = QWEN_HEADERS  # Use configured Qwen headers
    
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['choices'][0]['message']['content'].strip()
                t_generate = time.time() - t_generate_start
                print(f"DeepSeek answer generation completed, time elapsed: {t_generate:.2f} seconds")
                choice = extract_model_choice(answer, choices)
            else:
                print(f"DeepSeek API error: {response.status_code}")
                print(f"Error details: {response.text}")
                return fallback_response("deepseek")
        
        # Ensure response content is output regardless of which method was used to get the answer
        if answer:
            print(f"DeepSeek response completed, choice: Option {choice}" if choice else "DeepSeek response completed, unable to extract clear choice")
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
            print("Unable to get DeepSeek response")
            return fallback_response("deepseek")
    
    except Exception as e:
        print(f"Error generating DeepSeek response: {str(e)}")
        traceback.print_exc()
        return fallback_response("deepseek")



# Let models debate and determine if consensus is reached
def conduct_debate(case_vignette, choices, correct_answer, max_rounds=3, force_disagree=False):
    """Conduct debate between models
    
    Args:
        case_vignette: Case description
        choices: Options list
        correct_answer: Correct answer
        max_rounds: Maximum debate rounds
        force_disagree: Whether to force simulate disagreement (for testing)
        
    Returns:
        dict: Debate result, including final choice and debate history
    """
    try:
        # Get initial answers from three models
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
        
        # Used to record which model is more accurate (if we know the correct answer)
        # Map disease names to option numbers for easy comparison
        correct_choice = None
        gpt_initially_correct = False
        qwen_initially_correct = False
        deepseek_initially_correct = False
        
        if correct_answer:
            # Create mapping from option number to disease name
            choice_options = choices.strip().split('\n')
            option_mapping = {}
            for option in choice_options:
                match = re.match(r'(\d+)\.\s*(.+)', option.strip())
                if match:
                    number, disease = match.groups()
                    option_mapping[int(number)] = disease.strip()
            
            # Find the option number corresponding to the correct answer
            correct_choice = None
            exact_match = False
            
            # First round: look for exact match
            for number, disease in option_mapping.items():
                if disease.lower() == correct_answer.lower() or disease.strip().lower() == correct_answer.lower():
                    correct_choice = number
                    exact_match = True
                    break
            
            # Second round: if no exact match, look for word boundary match
            if not exact_match:
                for number, disease in option_mapping.items():
                    # Use regex for word boundary matching
                    if re.search(r'\b' + re.escape(correct_answer.lower()) + r'\b', disease.lower()):
                        correct_choice = number
                        break
            
            # Third round: if both previous rounds fail, use stricter partial matching (as fallback only)
            if not correct_choice:
                # Sort by option number to ensure priority matching order
                sorted_options = sorted(option_mapping.items())
                for number, disease in sorted_options:
                    # Only consider complete matches for short options, avoid matching "X" to "XIIa"
                    if correct_answer.lower() in disease.lower():
                        # Additional validation: if single character answer, ensure it's independent
                        if len(correct_answer) == 1:
                            # Check if it's an independent Roman numeral or letter
                            if re.search(r'\b' + re.escape(correct_answer.lower()) + r'\b', disease.lower()) or (
                                disease.lower() == correct_answer.lower()):
                                correct_choice = number
                                break
                        else:
                            correct_choice = number
                            break
            
            if correct_choice:
                gpt_initially_correct = (initial_gpt_choice == correct_choice)
                qwen_initially_correct = (initial_qwen_choice == correct_choice)
                deepseek_initially_correct = (initial_deepseek_choice == correct_choice)
                print(f"Correct answer: Option {correct_choice} ({correct_answer})")
                if gpt_initially_correct:
                    print("GPT's initial diagnosis is correct")
                if qwen_initially_correct:
                    print("Qwen's initial diagnosis is correct")
                if deepseek_initially_correct:
                    print("DeepSeek's initial diagnosis is correct")
        
        # Check if initial consensus is already reached
        initial_choices = [gpt_result["choice"], qwen_result["choice"], deepseek_result["choice"]]
        if check_consensus(initial_choices):
            consensus_choice = next((choice for choice in initial_choices if choice is not None), None)
            print(f"\nThree models have reached initial diagnostic consensus! All models chose Option {consensus_choice}")
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
                    "deepseek_changed": False,
                    "gpt_changed_from_correct": False,
                    "qwen_changed_from_correct": False,
                    "deepseek_changed_from_correct": False
                },
                "correct_choice": correct_choice
            }
        else:
            print(f"\nThree models have initial diagnostic disagreement, starting debate process... GPT chose Option {gpt_result['choice']}, Qwen chose Option {qwen_result['choice']}, DeepSeek chose Option {deepseek_result['choice']}")
        
        # Store debate history
        debate_history = [{
            "round": 0,
            "gpt": gpt_result,
            "qwen": qwen_result,
            "deepseek": deepseek_result
        }]
        
        # Establish mapping from option number to disease name
        choice_to_disease = {}
        choice_options = choices.strip().split('\n')
        for option in choice_options:
            match = re.match(r'(\d+)\.\s*(.+)', option.strip())
            if match:
                number, disease = match.groups()
                choice_to_disease[int(number)] = disease.strip()
        
        # Track the latest results of each model in each round of debate
        current_gpt_result = gpt_result
        current_qwen_result = qwen_result
        current_deepseek_result = deepseek_result


        # Start debate
        for round_num in range(1, max_rounds + 1):
            print(f"\n======== Debate Round {round_num} ========")
            
            # GPT responds to Qwen and DeepSeek, passing its own previous answer and choice
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
            gpt_disease = choice_to_disease.get(gpt_choice, "Unknown disease") if gpt_choice else "Unclear disease"
            qwen_disease = choice_to_disease.get(qwen_result["choice"], "Unknown disease") if qwen_result["choice"] else "Unclear disease"
            deepseek_disease = choice_to_disease.get(deepseek_result["choice"], "Unknown disease") if deepseek_result["choice"] else "Unclear disease"
            
            print(f"GPT's choice after response: Option {gpt_choice} ({gpt_disease})")
            print(f"Qwen's choice: Option {qwen_result['choice']} ({qwen_disease})")
            print(f"DeepSeek's choice: Option {deepseek_result['choice']} ({deepseek_disease})")
            
            # If GPT changed stance and was originally correct
            if correct_choice and gpt_initially_correct and gpt_response["choice"] != correct_choice:
                print(f"Warning: GPT changed from correct choice ({choice_to_disease.get(correct_choice, '')}) to incorrect choice ({gpt_disease})")
            
            # Check if consensus is reached
            current_choices = [gpt_response["choice"], qwen_result["choice"], deepseek_result["choice"]]
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_disease = choice_to_disease.get(consensus_choice, "Unknown disease")
                print(f"\nDebate Round {round_num}: All models have reached consensus! (All chose Option {consensus_choice} - {consensus_disease})")
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
                        "qwen_changed": initial_qwen_choice != qwen_result["choice"],  # 修改为qwen_result
                        "deepseek_changed": initial_deepseek_choice != deepseek_result["choice"],
                        "gpt_changed_from_correct": gpt_initially_correct and gpt_response["choice"] != correct_choice if correct_choice else False,
                        "qwen_changed_from_correct": qwen_initially_correct and qwen_result["choice"] != correct_choice if correct_choice else False,  # 修改为qwen_result
                        "deepseek_changed_from_correct": deepseek_initially_correct and deepseek_result["choice"] != correct_choice if correct_choice else False
                    },
                    "correct_choice": correct_choice
                }
            
            # Qwen回应GPT和DeepSeek，并传递自己之前的答案和选择
            qwen_response = qwen_responds_to_others(
                case_vignette, choices, 
                gpt_response["answer"], gpt_response["choice"], 
                current_deepseek_result["answer"], current_deepseek_result["choice"], 
                round_num,
                self_previous_answer=current_qwen_result["answer"], 
                self_previous_choice=current_qwen_result["choice"]
            )
            
            # 更新当前Qwen结果
            current_qwen_result = qwen_response
            
            qwen_choice = qwen_response["choice"]
            qwen_disease = choice_to_disease.get(qwen_choice, "未知疾病") if qwen_choice else "未明确疾病"
            
            print(f"Qwen回应后的选择：选项 {qwen_choice} ({qwen_disease})")
            print(f"GPT选择：选项 {gpt_response['choice']} ({gpt_disease})")
            print(f"DeepSeek选择：选项 {deepseek_result['choice']} ({deepseek_disease})")
            
            # 如果Qwen改变了立场并且原来是正确的
            if correct_choice and qwen_initially_correct and qwen_response["choice"] != correct_choice:
                print(f"警告: Qwen从正确的选择 ({choice_to_disease.get(correct_choice, '')}) 改变为不正确的选择 ({qwen_disease})")
            
            # 检查是否达成一致
            current_choices = [gpt_response["choice"], qwen_response["choice"], deepseek_result["choice"]]
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_disease = choice_to_disease.get(consensus_choice, "未知疾病")
                print(f"\n辩论第{round_num}轮: 所有模型已达成一致！(均选择了选项{consensus_choice} - {consensus_disease})")
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
                        "deepseek_changed": initial_deepseek_choice != deepseek_result["choice"],
                        "gpt_changed_from_correct": gpt_initially_correct and gpt_response["choice"] != correct_choice if correct_choice else False,
                        "qwen_changed_from_correct": qwen_initially_correct and qwen_response["choice"] != correct_choice if correct_choice else False,
                        "deepseek_changed_from_correct": deepseek_initially_correct and deepseek_result["choice"] != correct_choice if correct_choice else False
                    },
                    "correct_choice": correct_choice
                }
            
            # DeepSeek回应GPT和Qwen，并传递自己之前的答案和选择
            deepseek_response = deepseek_responds_to_others(
                case_vignette, choices, 
                gpt_response["answer"], gpt_response["choice"],
                qwen_response["answer"], qwen_response["choice"], 
                round_num,
                self_previous_answer=current_deepseek_result["answer"], 
                self_previous_choice=current_deepseek_result["choice"]
            )
            
            # 更新当前DeepSeek结果
            current_deepseek_result = deepseek_response
            
            deepseek_choice = deepseek_response["choice"]
            deepseek_disease = choice_to_disease.get(deepseek_choice, "未知疾病") if deepseek_choice else "未明确疾病"
            
            # 确保已经输出了DeepSeek的回应内容，如果没有，这里再次输出
            # 此代码只是一个备份，正常情况下deepseek_responds_to_others函数已经输出了回应内容
            if '_output_shown' not in deepseek_response:
                print(f"DeepSeek回应后的选择：选项 {deepseek_choice} ({deepseek_disease})")
                if "answer" in deepseek_response and deepseek_response["answer"]:
                    print("\nDeepSeek对其他模型的回应内容:")
                    print("="*80)
                    print(deepseek_response["answer"])
                    print("="*80)
                deepseek_response['_output_shown'] = True
            
            print(f"GPT选择：选项 {gpt_response['choice']} ({gpt_disease})")
            print(f"Qwen选择：选项 {qwen_response['choice']} ({qwen_disease})")
            
            # 如果DeepSeek改变了立场并且原来是正确的
            if correct_choice and deepseek_initially_correct and deepseek_response["choice"] != correct_choice:
                print(f"警告: DeepSeek从正确的选择 ({choice_to_disease.get(correct_choice, '')}) 改变为不正确的选择 ({deepseek_disease})")
            
            # 检查是否达成一致
            current_choices = [gpt_response["choice"], qwen_response["choice"], deepseek_response["choice"]]

            # 在这里输出一个诊断信息，用于调试立场变化
            if initial_deepseek_choice != deepseek_response["choice"]:
                print(f"\nDeepSeek改变了立场！从选项{initial_deepseek_choice}变为选项{deepseek_response['choice']}")
                
                # 如果DeepSeek从错误变为正确
                if correct_choice and not deepseek_initially_correct and deepseek_response["choice"] == correct_choice:
                    print(f"DeepSeek成功改正了错误！从选项{initial_deepseek_choice}纠正为正确选项{deepseek_response['choice']}")
                
                # 如果DeepSeek从正确变为错误
                if correct_choice and deepseek_initially_correct and deepseek_response["choice"] != correct_choice:
                    print(f"警告：DeepSeek从正确选项{initial_deepseek_choice}变为错误选项{deepseek_response['choice']}")            
            
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_disease = choice_to_disease.get(consensus_choice, "未知疾病")
                print(f"\n辩论第{round_num}轮: 所有模型已达成一致！(均选择了选项{consensus_choice} - {consensus_disease})")
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
                        "deepseek_changed": initial_deepseek_choice != deepseek_response["choice"],
                        "gpt_changed_from_correct": gpt_initially_correct and gpt_response["choice"] != correct_choice if correct_choice else False,
                        "qwen_changed_from_correct": qwen_initially_correct and qwen_response["choice"] != correct_choice if correct_choice else False,
                        "deepseek_changed_from_correct": deepseek_initially_correct and deepseek_response["choice"] != correct_choice if correct_choice else False
                    },
                    "correct_choice": correct_choice
                }
            
            # 更新模型的结果用于下一轮辩论
            gpt_result = gpt_response
            qwen_result = qwen_response
            deepseek_result = deepseek_response
            
            # 记录本轮结果
            debate_history.append({
                "round": round_num,
                "gpt": gpt_response,
                "qwen": qwen_response,
                "deepseek": deepseek_response
            })
            
            print(f"\n辩论第{round_num}轮: 仍未达成一致，GPT选择选项{gpt_response['choice']} ({gpt_disease})，Qwen选择选项{qwen_response['choice']} ({qwen_disease})，DeepSeek选择选项{deepseek_response['choice']} ({deepseek_disease})")
        
        print("\n达到最大辩论轮次，仍未达成一致。")
        
        # 获取最终的各模型回答和选择
        gpt_final_choice = debate_history[-1]["gpt"]["choice"]
        qwen_final_choice = debate_history[-1]["qwen"]["choice"]
        deepseek_final_choice = debate_history[-1]["deepseek"]["choice"]
        
        # 采用多数投票决定最终选择
        final_choices = [gpt_final_choice, qwen_final_choice, deepseek_final_choice]
        choice_counts = {}
        for choice in final_choices:
            if choice is not None:
                choice_counts[choice] = choice_counts.get(choice, 0) + 1
        
        # 找出得票最多的选项
        max_votes = 0
        final_choice = None
        for choice, count in choice_counts.items():
            if count > max_votes:
                max_votes = count
                final_choice = choice
        
        # 如果没有明确的多数选择，随机选择一个非None的选项
        if final_choice is None:
            valid_choices = [c for c in final_choices if c is not None]
            if valid_choices:
                final_choice = random.choice(valid_choices)
            else:
                print("所有模型都未给出明确选择，无法确定最终结果")
                return None
        
        final_disease = choice_to_disease.get(final_choice, "未知疾病")
        print(f"\n最终选择（多数投票）: 选项{final_choice} ({final_disease})")
        
        # 检查是否有模型从正确变为错误
        if correct_choice:
            gpt_changed_from_correct = (gpt_initially_correct and gpt_final_choice != correct_choice)
            qwen_changed_from_correct = (qwen_initially_correct and qwen_final_choice != correct_choice)
            deepseek_changed_from_correct = (deepseek_initially_correct and deepseek_final_choice != correct_choice)
            
            if gpt_changed_from_correct:
                gpt_final_disease = choice_to_disease.get(gpt_final_choice, "未知疾病")
                print(f"警告: GPT从正确的选择 ({choice_to_disease.get(correct_choice, '')}) 改变为不正确的选择 ({gpt_final_disease})")
            
            if qwen_changed_from_correct:
                qwen_final_disease = choice_to_disease.get(qwen_final_choice, "未知疾病")
                print(f"警告: Qwen从正确的选择 ({choice_to_disease.get(correct_choice, '')}) 改变为不正确的选择 ({qwen_final_disease})")
            
            if deepseek_changed_from_correct:
                deepseek_final_disease = choice_to_disease.get(deepseek_final_choice, "未知疾病")
                print(f"警告: DeepSeek从正确的选择 ({choice_to_disease.get(correct_choice, '')}) 改变为不正确的选择 ({deepseek_final_disease})")
            
            is_final_correct = (final_choice == correct_choice)
            correct_disease = choice_to_disease.get(correct_choice, "未知疾病")
            print(f"正确诊断: 选项{correct_choice} ({correct_disease})")
            print(f"最终选择是否正确: {'正确 ✓' if is_final_correct else '错误 ✗'}")
        
        # 诊断信息，显示最终的 DeepSeek 立场变化
        if initial_deepseek_choice != deepseek_final_choice:
            print(f"\n辩论结束时 DeepSeek 改变了立场：从选项{initial_deepseek_choice}变为选项{deepseek_final_choice}")
            if correct_choice and not deepseek_initially_correct and deepseek_final_choice == correct_choice:
                print(f"DeepSeek 成功纠正了错误！从选项{initial_deepseek_choice}变为正确选项{deepseek_final_choice}")
        
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
                "deepseek_changed": initial_deepseek_choice != deepseek_final_choice,
                "gpt_changed_from_correct": gpt_initially_correct and gpt_final_choice != correct_choice if correct_choice else False,
                "qwen_changed_from_correct": qwen_initially_correct and qwen_final_choice != correct_choice if correct_choice else False,
                "deepseek_changed_from_correct": deepseek_initially_correct and deepseek_final_choice != correct_choice if correct_choice else False
            },
            "correct_choice": correct_choice
        }
            
    except Exception as e:
        print(f"Error during debate: {str(e)}")
        traceback.print_exc()
        return None

# Evaluate whether the answer is correct
def evaluate_answer(dataset, case_idx, model_choice):
    """Evaluate whether the model's choice is correct
    
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
        
        # Prioritize handling medmcqa dataset's cop field
        if "cop" in dataset.columns:
            correct_option_idx = dataset.iloc[case_idx]["cop"]
            # cop field values are 0-3, corresponding to options 1-4
            correct_answer_num = int(correct_option_idx) + 1
            return model_choice == correct_answer_num
        
        # Handle MedQA dataset's label field
        elif "label" in dataset.columns:
            correct_label = dataset.iloc[case_idx]["label"]
            # label field directly represents correct ending index (starting from 0)
            # Model choice starts from 1, so need to add 1
            correct_answer_num = int(correct_label) + 1
            return model_choice == correct_answer_num
        
        # Handle RABBITS dataset's correct_choice field
        if "correct_choice" in dataset.columns:
            correct_choice_letter = dataset.iloc[case_idx]["correct_choice"]
            if isinstance(correct_choice_letter, str) and len(correct_choice_letter) == 1 and correct_choice_letter.upper() in "ABCDE":
                # Convert letter answer to number (A=1, B=2, C=3, D=4, E=5)
                correct_answer_num = ord(correct_choice_letter.upper()) - ord('A') + 1
                return model_choice == correct_answer_num
            
        # Check answer_idx field (commonly found in JSONL format)
        if "answer_idx" in dataset.columns:
            correct_idx = dataset.iloc[case_idx]["answer_idx"]
            # Convert letter answer index to number
            if isinstance(correct_idx, str) and len(correct_idx) == 1 and correct_idx.isalpha():
                correct_answer_num = ord(correct_idx.upper()) - ord('A') + 1
                return model_choice == correct_answer_num
        
        # Directly check answer field
        if "answer" in dataset.columns:
            correct_text = dataset.iloc[case_idx]["answer"]
            
            # Get options list
            choices = get_choices(dataset, case_idx)
            choice_options = choices.strip().split('\n')
            
            # Create mapping from option number to option content
            option_mapping = {}
            option_content_to_num = {}
            for option in choice_options:
                match = re.match(r'(\d+)\.\s*(.+)', option.strip())
                if match:
                    number, content = match.groups()
                    option_mapping[int(number)] = content.strip()
                    option_content_to_num[content.strip().lower()] = int(number)
            
            # Check if model's selected option content matches correct answer text
            if model_choice in option_mapping:
                model_answer_text = option_mapping[model_choice].lower()
                if correct_text.lower() == model_answer_text:
                    return True
                
            # Check if correct answer text matches any option content
            if correct_text.lower() in option_content_to_num:
                correct_answer_num = option_content_to_num[correct_text.lower()]
                return model_choice == correct_answer_num
                
        # Fallback: if all above methods fail, try the original method
        columns = dataset.columns
        answer_col = next((col for col in columns if col.lower() in ['answer', 'correct_answer', 'label']), None)
        
        if not answer_col:
            print("Cannot determine correct answer column name")
            return False
            
        # Try to convert answer to number
        correct_answer = str(dataset.iloc[case_idx][answer_col])
        if correct_answer.isdigit():
            return model_choice == int(correct_answer)
        elif len(correct_answer) == 1 and correct_answer.upper() in "ABCDE":
            # Handle letter answers (A, B, C, D, E)
            correct_answer_num = ord(correct_answer.upper()) - ord('A') + 1
            return model_choice == correct_answer_num
            
        # If all fail, return False
        print(f"Cannot compare answers: model choice {model_choice}, correct answer {correct_answer}")
        return False
            
    except Exception as e:
        print(f"Error evaluating answer: {str(e)}")
        traceback.print_exc()
        return False

# Process debate for a single case
def process_single_debate(dataset_path, case_idx=0, max_rounds=3, force_disagree=False):
    """Process debate for a single medical multiple choice case
    
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
        
        # Get case data - adapted for MedQA, RABBITS and JSONL formats
        case_id = dataset.loc[case_idx, "id"] if "id" in dataset.columns else (
            dataset.loc[case_idx, "question_id"] if "question_id" in dataset.columns else f"case_{case_idx}")
        
        # Prioritize using medmcqa dataset's question field, then use MedQA's sent1 field
        if "question" in dataset.columns:
            case_vignette = dataset.loc[case_idx, "question"]  # medmcqa dataset uses question field
        elif "sent1" in dataset.columns:
            case_vignette = dataset.loc[case_idx, "sent1"]  # MedQA dataset uses sent1 field
        elif "input" in dataset.columns:
            case_vignette = dataset.loc[case_idx, "input"]  # RABBITS dataset uses input field
        else:
            case_vignette = "Question content not found"
            
        # Get category information: medmcqa uses subject_name, other datasets use meta_info
        if "subject_name" in dataset.columns:
            category = dataset.loc[case_idx, "subject_name"]  # medmcqa dataset
        elif "meta_info" in dataset.columns:
            category = dataset.loc[case_idx, "meta_info"]  # other datasets
        else:
            category = "Medical Multiple Choice"
        choices = get_choices(dataset, case_idx)
        
        # Get correct answer: medmcqa uses cop field, MedQA uses label field, RABBITS uses output field
        if "cop" in dataset.columns:
            correct_option_idx = dataset.loc[case_idx, "cop"]
            # cop field values are 0-3, corresponding to options opa, opb, opc, opd
            option_fields = ['opa', 'opb', 'opc', 'opd']
            if 0 <= correct_option_idx < len(option_fields):
                correct_field = option_fields[correct_option_idx]
                correct_answer = dataset.loc[case_idx, correct_field] if correct_field in dataset.columns else f"Option {correct_option_idx + 1}"
            else:
                correct_answer = f"Option {correct_option_idx + 1}"
        elif "label" in dataset.columns:
            correct_label = dataset.loc[case_idx, "label"]
            # Get corresponding ending content as correct answer
            correct_answer = dataset.loc[case_idx, f"ending{correct_label}"] if f"ending{correct_label}" in dataset.columns else f"Option {correct_label + 1}"
        elif "output" in dataset.columns:
            correct_answer = dataset.loc[case_idx, "output"]  # RABBITS dataset uses output field
        elif "answer" in dataset.columns:
            correct_answer = dataset.loc[case_idx, "answer"]
        else:
            correct_answer = "Unknown answer"
        
        print(f"Case ID: {case_id}")
        print(f"Category: {category}")
        print(f"Case Description: \n{case_vignette}")
        print(f"Options: \n{choices}")
        print(f"Correct Answer: {correct_answer}")
        
        # Conduct debate
        result = conduct_debate(case_vignette, choices, correct_answer, max_rounds, force_disagree)
        
        if not result:
            print("Debate process failed")
            return None
        
        # Get final choice
        final_choice = result["final_choice"]
        
        # Create mapping from option number to disease name
        choice_to_disease = {}
        choice_options = choices.strip().split('\n')
        for option in choice_options:
            match = re.match(r'(\d+)\.\s*(.+)', option.strip())
            if match:
                number, disease = match.groups()
                choice_to_disease[int(number)] = disease.strip()
        
        # Final diagnosed disease
        final_disease = choice_to_disease.get(final_choice, "Unknown disease") if final_choice else "Unclear disease"
        
        # Evaluate if result is correct
        is_correct = evaluate_answer(dataset, case_idx, final_choice)
        
        # Output final result
        print("\n========= Final Debate Result =========")
        if result["consensus"]:
            print(f"GPT, Qwen and DeepSeek-R1 reached consensus! Final diagnosis: Option {final_choice} - {final_disease}")
        else:
            # After three rounds without consensus, decide by majority vote
            print(f"GPT, Qwen and DeepSeek-R1 did not reach consensus")
            print(f"Decided by majority vote, final diagnosis: Option {final_choice} - {final_disease}")
        
        print(f"Correct answer: Option ? - {correct_answer}")
        print(f"Is final diagnosis correct: {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        # Build output result
        output_result = {
            "case_id": case_id,
            "category": category,
            "vignette": case_vignette,
            "choices": choices,
            "correct_answer": correct_answer,
            "debate_result": result,
            "is_correct": is_correct
        }
        
        return output_result
        
    except Exception as e:
        print(f"Error processing single case debate: {str(e)}")
        traceback.print_exc()
        return None

# Main function
def main():
    try:
        # Set output redirection to terminal and log file
        log_dir = "logs_medmcqa_generic_to_brand"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        sys.stdout = TeeOutput(log_file)
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Medical Model Debate System')
        parser.add_argument('-n', '--num_cases', type=int, default=1, help='Number of cases to process (default: 1)')
        parser.add_argument('-s', '--start_idx', type=int, default=0, help='Starting case index (default: 0)')
        parser.add_argument('-f', '--force_disagree', action='store_true', help='Force simulate model disagreement to test judge function')
        parser.add_argument('--force_judge', action='store_true', help='Force use DeepSeek judge regardless of debate result')
        args = parser.parse_args()
        
        # Dataset path - using MedMCQA dataset
        dataset_path = "../benchmarks/RABBITS/generic_to_brand/medmcqa_generic_to_brand.json"
        
        # Check if dataset exists
        if not check_file_exists(dataset_path):
            print("Program terminated, dataset does not exist.")
            exit(1)
        
        # Process multiple case debates
        print("="*50)
        print("GPT vs Qwen vs DeepSeek-R1 Medical Diagnosis Debate - Model Competition and Collaboration Officially Begins")
        print("="*50)
        
        # Create results directory
        results_dir = "debate_results_medmcqa_generic_to_brand"
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
                    # Save single case result to JSON file
                    case_result_file = os.path.join(results_dir, f"debate_result_case_{i}.json")
                    with open(case_result_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"\nDebate result for case {i} saved to {case_result_file}")
                    
                    # Add to summary results
                    summary_result = {
                        "case_id": result["case_id"],
                        "category": result["category"],
                        "correct_answer": result["correct_answer"],
                        "consensus": result["debate_result"]["consensus"],
                        "voting_needed": not result["debate_result"]["consensus"],  # If no consensus reached, need majority vote
                        "final_choice": result["debate_result"]["final_choice"],
                        "is_correct": result["is_correct"],
                        "stance_changes": result["debate_result"].get("stance_changes", {}),
                        "debate_result": {  # For compatibility with statistics code, add complete debate_result
                            "stance_changes": result["debate_result"].get("stance_changes", {}),
                            "initial_choices": result["debate_result"].get("initial_choices", {}),
                            "debate_history": result["debate_result"].get("debate_history", []),
                            "correct_choice": result["debate_result"].get("correct_choice")
                        }
                    }
                    summary_results.append(summary_result)
            except Exception as e:
                print(f"Error processing case {i}: {str(e)}")
                traceback.print_exc()
        
        # Save summary results
        summary_file = os.path.join(results_dir, "debate_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=2)
        
        # Output summary statistics
        if summary_results:
            total_cases = len(summary_results)
            correct_cases = sum(1 for r in summary_results if r["is_correct"])
            consensus_cases = sum(1 for r in summary_results if r["consensus"])
            voting_cases = sum(1 for r in summary_results if not r["consensus"])  # Cases without consensus need majority vote
            
            # Statistics on model stance changes
            gpt_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("gpt_changed", False))
            qwen_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("qwen_changed", False))
            deepseek_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("deepseek_changed", False))
            
            # Statistics on changing from correct to incorrect
            gpt_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("gpt_changed_from_correct", False))
            qwen_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("qwen_changed_from_correct", False))
            deepseek_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("deepseek_changed_from_correct", False))
            
            # Statistics on changing from incorrect to correct (positive cases of model self-correction)
            gpt_changed_to_correct = sum(1 for r in summary_results 
                if r["stance_changes"].get("gpt_changed", False) and 
                not r["debate_result"]["initial_choices"]["gpt"] == r["debate_result"].get("correct_choice") and 
                r["debate_result"]["debate_history"][-1]["gpt"]["choice"] == r["debate_result"].get("correct_choice"))
            
            qwen_changed_to_correct = sum(1 for r in summary_results 
                if r["stance_changes"].get("qwen_changed", False) and 
                not r["debate_result"]["initial_choices"]["qwen"] == r["debate_result"].get("correct_choice") and 
                r["debate_result"]["debate_history"][-1]["qwen"]["choice"] == r["debate_result"].get("correct_choice"))
                
            deepseek_changed_to_correct = sum(1 for r in summary_results 
                if r["stance_changes"].get("deepseek_changed", False) and 
                not r["debate_result"]["initial_choices"]["deepseek"] == r["debate_result"].get("correct_choice") and 
                r["debate_result"]["debate_history"][-1]["deepseek"]["choice"] == r["debate_result"].get("correct_choice"))
            
            print("\n" + "="*50)
            print("Debate Results Statistics")
            print("="*50)
            print(f"Total cases processed: {total_cases}")
            print(f"Correct diagnosis cases: {correct_cases} ({correct_cases/total_cases:.2%})")
            print("-" * 40)  # Separator line
            print(f"Model consensus cases: {consensus_cases} ({consensus_cases/total_cases:.2%})")
            print(f"Cases decided by majority vote: {voting_cases} ({voting_cases/total_cases:.2%})")
            print("-" * 40)  # Separator line
            # Model stance change statistics
            print(f"GPT stance change cases: {gpt_changed_stance} ({gpt_changed_stance/total_cases:.2%})")
            print(f"Qwen stance change cases: {qwen_changed_stance} ({qwen_changed_stance/total_cases:.2%})")
            print(f"DeepSeek stance change cases: {deepseek_changed_stance} ({deepseek_changed_stance/total_cases:.2%})")
            print("-" * 40)  # Separator line
            # Statistics on changing from correct to incorrect (negative impact)
            print(f"GPT changed from correct to incorrect cases: {gpt_changed_from_correct} ({gpt_changed_from_correct/total_cases:.2%})")
            print(f"Qwen changed from correct to incorrect cases: {qwen_changed_from_correct} ({qwen_changed_from_correct/total_cases:.2%})")
            print(f"DeepSeek changed from correct to incorrect cases: {deepseek_changed_from_correct} ({deepseek_changed_from_correct/total_cases:.2%})")
            print("-" * 40)  # Separator line
            # Statistics on changing from incorrect to correct (positive impact)
            print(f"GPT changed from incorrect to correct cases: {gpt_changed_to_correct} ({gpt_changed_to_correct/total_cases:.2%})")
            print(f"Qwen changed from incorrect to correct cases: {qwen_changed_to_correct} ({qwen_changed_to_correct/total_cases:.2%})")
            print(f"DeepSeek changed from incorrect to correct cases: {deepseek_changed_to_correct} ({deepseek_changed_to_correct/total_cases:.2%})")
            print("-" * 40)  # Separator line
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

# Usage examples:
# python ThreeLLM.py -n 3                         # Process first 3 cases
# python ThreeLLM.py -n 5 -s 10                   # Process 5 cases from index 10-14
