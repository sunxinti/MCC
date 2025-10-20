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
        print("Configuration complete! Starting medical model debate system...")
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


class TeeOutput:
    """Class that sends output to both terminal and log file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a", encoding="utf-8")
        self.logfile.write(f"\n{'='*50}\n")
        self.logfile.write(f"Log start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logfile.write(f"{'='*50}\n\n")
        self.logfile.flush()

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush() 

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
        
    def close(self):
        """Close log file"""
        if self.logfile:
            self.logfile.write(f"\n{'='*50}\n")
            self.logfile.write(f"Log end time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.logfile.write(f"{'='*50}\n\n")
            self.logfile.close()
            self.logfile = None


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print("Error: File '{}' does not exist!".format(file_path))
        return False
    return True

# load MedQA dataset
def load_medical_mcq_data(file_path):
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


def get_choices(dataset, case_idx):
    """Extract choices list from the specified case in the dataset
    Args:
        dataset: Dataset
        case_idx: Case index
        
    Returns:
        str: Formatted choices list, e.g., "1. Option1\n2. Option2"
    """
    try:
        # Get options
        if 'options' in dataset.columns:
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
            elif isinstance(options, str) and options.startswith('['):
                try:
                    options_list = json.loads(options)
                    choices_list = [f"{i+1}. {option}" for i, option in enumerate(options_list)]
                    return "\n".join(choices_list)
                except:
                    pass
        
        choice_cols = [col for col in dataset.columns if col.startswith('choice_') or col == 'choice']
        if choice_cols:
            choices = []
            for i in range(1, 10): 
                col_name = f'choice_{i}'
                if col_name in dataset.columns and not pd.isna(dataset.iloc[case_idx][col_name]):
                    choices.append(f"{i}. {dataset.iloc[case_idx][col_name]}")
            
            if choices:
                return "\n".join(choices)
        
        cols = dataset.columns.str.contains("choice")
        if any(cols):
            choices = dataset.iloc[case_idx][cols]
            choices_list = [f"{i+1}. {choice}" for i, choice in enumerate(choices) if not pd.isna(choice)]
            return "\n".join(choices_list)
        
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
def get_gpt_prompt(case_vignette, choices):
    system_prompt = """You are analyzing a medical case that requires systematic clinical reasoning and precise diagnostic evaluation. You will be provided with a medical case and possible diagnostic options. Your task is to conduct a detailed medical analysis through careful reasoning, critically evaluate each option, and select the most appropriate diagnosis.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Medical case】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Diagnostic Options】
{choices}

Please provide a structured medical analysis using the following format:

**1. Case Summary and Key Symptoms**  
- Briefly summarize the patient's basic information, chief complaint, and key clinical manifestations.  
- Highlight symptoms and signs that are decisive for diagnosis.  

**2. Analysis of Clinical Features**  
- Analyze the key clinical features in the case in detail.  
- Discuss the timeline, characteristics, and interrelationships of the symptoms.  
- Consider both present symptoms and notably absent findings that would be expected in certain diagnoses.
- Explain which features support or oppose certain diagnoses.  

**3. Differential Diagnosis Analysis**  
- Systematically evaluate the likelihood of each option.  
- Analyze how well each option aligns with the patient's clinical presentation.  
- Support your reasoning with relevant medical knowledge and clinical guidelines.  
- Re-evaluate whether your diagnostic reasoning explains ALL of the patient's key symptoms and findings.

**4. Final Selection**  
- Clearly state the option you believe is the most accurate.  
- Summarize the key evidence supporting this choice.  
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X (Actual option in English)(Translation of the option in Chinese)**  

Note: You must choose one option from the provided list and clearly indicate the option number and content as per the format above.  
Ensure your analysis is logically rigorous, medically sound, and the selection must be one of the options provided in the list."""
    return system_prompt

def generate_gpt_answer(case_vignette, choices):
    try:
        prompt = get_gpt_prompt(case_vignette, choices)
        
        print("\nGPT is reasoning the answer...")
        t_generate_start = time.time()
        
        data = {
            "model": "o1-mini", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are the GPT Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
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
            return f"Sorry, an error occurred while processing your request. Error code: {response.status_code}"

    except Exception as e:
        print(f"Error occurred while generating GPT answer: {str(e)}")
        traceback.print_exc()
        return f"Sorry, an error occurred while processing your request. Error message: {str(e)}"


# ===================== Qwen Model Section =====================
def get_qwen_prompt(case_vignette, choices):
    system_prompt = """You are analyzing a medical case that requires systematic clinical reasoning and precise diagnostic evaluation. You will be provided with a medical case and possible diagnostic options. Your task is to conduct a detailed medical analysis through careful reasoning, critically evaluate each option, and select the most appropriate diagnosis.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Medical case】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Diagnostic Options】
{choices}

Please provide a structured medical analysis using the following format:

**1. Case Summary and Key Symptoms**  
- Briefly summarize the patient's basic information, chief complaint, and key clinical manifestations.  
- Highlight symptoms and signs that are decisive for diagnosis.  

**2. Analysis of Clinical Features**  
- Analyze the key clinical features in the case in detail.  
- Discuss the timeline, characteristics, and interrelationships of the symptoms.  
- Consider both present symptoms and notably absent findings that would be expected in certain diagnoses.
- Explain which features support or oppose certain diagnoses.  

**3. Differential Diagnosis Analysis**  
- Systematically evaluate the likelihood of each option.  
- Analyze how well each option aligns with the patient's clinical presentation.  
- Support your reasoning with relevant medical knowledge and clinical guidelines.  
- Re-evaluate whether your diagnostic reasoning explains ALL of the patient's key symptoms and findings.

**4. Final Selection**  
- Clearly state the option you believe is the most accurate.  
- Summarize the key evidence supporting this choice.  
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X (Actual option in English)(Translation of the option in Chinese)**  

Note: You must choose one option from the provided list and clearly indicate the option number and content as per the format above.  
Ensure your analysis is logically rigorous, medically sound, and the selection must be one of the options provided in the list."""
    return system_prompt


def generate_qwen_answer(case_vignette, choices):
    try:
        prompt = get_qwen_prompt(case_vignette, choices)
        
        print("\nQwen is reasoning the answer...")
        t_generate_start = time.time()
        
        data = {
            "model": "Qwen/QwQ-32B", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are the Qwen Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }

        response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data, timeout=300)
        t_generate = time.time() - t_generate_start
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Response JSON structure: {list(response_data.keys())}")
            
            if 'choices' in response_data and response_data['choices']:
                print(f"Choices structure: {list(response_data['choices'][0].keys())}")
                
                if 'message' in response_data['choices'][0]:
                    message = response_data['choices'][0]['message']
                    print(f"Message structure: {list(message.keys())}")
                    
                    if 'content' in message and message['content'] and len(message['content'].strip()) > 0:
                        answer = message['content'].strip()
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
                        print(f"Qwen answer generation completed, time taken: {t_generate:.2f} seconds")
                        return answer
                    else:
                        print("Extracted content is empty")
                        return "API returned empty content"
                else:
                    print("Error: message field does not exist")
                    return "API response structure error: missing message field"
            else:
                print("Error: choices field does not exist or is empty")
                return "API response structure error: missing choices field"
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
    system_prompt = """You are analyzing a medical case that requires systematic clinical reasoning and precise diagnostic evaluation. You will be provided with a medical case and possible diagnostic options. Your task is to conduct a detailed medical analysis through careful reasoning, critically evaluate each option, and select the most appropriate diagnosis.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

【Medical case】
"""
    system_prompt += f"{case_vignette}"
    system_prompt += f"""

【Diagnostic Options】
{choices}

Please provide a structured medical analysis using the following format:

**1. Case Summary and Key Symptoms**  
- Briefly summarize the patient's basic information, chief complaint, and key clinical manifestations.  
- Highlight symptoms and signs that are decisive for diagnosis.  

**2. Analysis of Clinical Features**  
- Analyze the key clinical features in the case in detail.  
- Discuss the timeline, characteristics, and interrelationships of the symptoms.  
- Consider both present symptoms and notably absent findings that would be expected in certain diagnoses.
- Explain which features support or oppose certain diagnoses.  

**3. Differential Diagnosis Analysis**  
- Systematically evaluate the likelihood of each option.  
- Analyze how well each option aligns with the patient's clinical presentation.  
- Support your reasoning with relevant medical knowledge and clinical guidelines.  
- Re-evaluate whether your diagnostic reasoning explains ALL of the patient's key symptoms and findings.

**4. Final Selection**  
- Clearly state the option you believe is the most accurate.  
- Summarize the key evidence supporting this choice.  
- **[Extremely Important]** Your final selection must use the exact format below; otherwise, it will not be correctly recognized by the system:  
**My final selection is: Option X (Actual option in English)(Translation of the option in Chinese)**  

Note: You must choose one option from the provided list and clearly indicate the option number and content as per the format above.  
Ensure your analysis is logically rigorous, medically sound, and the selection must be one of the options provided in the list."""
    return system_prompt


def generate_deepseek_answer(case_vignette, choices):
    try:
        prompt = get_deepseek_prompt(case_vignette, choices)
        
        print("\nDeepSeek is reasoning the answer...")
        t_generate_start = time.time()
        
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model="deepseek-reasoner", # Note, use the same as the original study; you can replace it with other LLMs. 
                messages=[
                    {"role": "system", "content": "You are the DeepSeek Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
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
            
            # Fallback: Direct API call using requests, SiliconFlow API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
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
        print(f"Error occurred while generating DeepSeek answer: {str(e)}")
        traceback.print_exc()
        print("Unable to generate DeepSeek answer, task terminated")
        sys.exit(1)



# ===================== Model Debate Section =====================
def extract_model_choice(answer_text, choices_text=None):
    """Extract the final choice and agreement status from model response
    
    Args:
        answer_text: Complete response text from the model
        choices_text: Option text for dynamic matching (optional)
    
    Returns:
        int: Extracted option number (1-n), returns None if extraction fails
    """
    
    # Debug information
    print("\nStarting model choice extraction...")
    
    clean_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer_text)
    
    # Extract whether the model agrees with other models' opinions
    first_paragraph = clean_answer.split("\n")[0] if "\n" in clean_answer else clean_answer[:200]
    if "I acknowledge" in first_paragraph or "I agree" in first_paragraph or "I accept" in first_paragraph:
        print("Model agrees with other models' opinions")
    elif "I do not acknowledge" in first_paragraph or "I disagree" in first_paragraph or "I do not accept" in first_paragraph:
        print("Model disagrees with other models' opinions")
    
    # Parse choices_text and build mapping from option content to option numbers
    option_content_to_num = {}
    medical_term_expansions = {}
    if choices_text:
        choice_lines = choices_text.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                option_content = match.group(2).strip()
                option_content_to_num[option_content.lower()] = option_num
                # Handle short name matching
                short_name = option_content.split()[0].lower() if ' ' in option_content else option_content.lower()
                option_content_to_num[short_name] = option_num
                
    
    # Prioritize extracting the final conclusion section
    # Try to identify the final conclusion paragraph
    # Common final conclusion markers
    conclusion_markers = [
        "final decision", "final selection", "final choice", "final diagnosis", 
        "in conclusion", "to conclude", "final answer", "my conclusion"
    ]
    
    # Find the final conclusion paragraph
    conclusion_text = ""
    lines = clean_answer.split("\n")
    for i, line in enumerate(lines):
        if any(marker.lower() in line.lower() for marker in conclusion_markers):
            # Get this line and subsequent lines as conclusion text
            conclusion_text = "\n".join(lines[i:min(i+5, len(lines))])
            break
    
    # If no clear conclusion paragraph is found, use the last few lines
    if not conclusion_text:
        conclusion_text = "\n".join(lines[-15:])
    
    # 1. First, strict matching of standard format "My final selection is: Option X (option content)"
    # Define strict matching patterns for final conclusions, supporting Markdown format
    final_choice_strict_patterns = [
        r'my final selection is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?\s*(?:\(([^)]+)\))?',  # Support Markdown format
        r'my final choice is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?\s*(?:\(([^)]+)\))?',
        r'my final diagnosis is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'my final decision is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'final (?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'(?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
    ]
    
    # Prioritize matching in conclusion text
    search_texts = [conclusion_text, clean_answer]
    
    # Prioritize matching the most explicit conclusion patterns
    for text in search_texts:
        for pattern in final_choice_strict_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    option_num = int(match.group(1))
                    if 1 <= option_num <= 10:  # Expand option range to 10
                        if len(match.groups()) > 1 and match.group(2):
                            option_content = match.group(2).strip()
                            print(f"[Strict Match] Found final choice: Option {option_num} ({option_content})")
                        else:
                            print(f"[Strict Match] Found final choice: Option {option_num}")
                        return option_num
                except (ValueError, IndexError):
                    continue
    
    # 2. Match direct references to disease names
    # Search for disease/option content references in conclusion section
    disease_reference_patterns = [
        r'my final (?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?([^*().,:;]+)(?:\*\*)?',
        r'(?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?([^*().,:;]+)(?:\*\*)?',
    ]
    
    for text in search_texts:
        for pattern in disease_reference_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                disease_name = match.group(1).strip().lower()
                # First check extended medical terms
                if disease_name in medical_term_expansions:
                    print(f"[Disease Reference] Matched medical term from conclusion: {disease_name}, corresponding option: {medical_term_expansions[disease_name]}")
                    return medical_term_expansions[disease_name]
                # Then check option content
                for content, num in option_content_to_num.items():
                    if content == disease_name or disease_name in content or content in disease_name:
                        print(f"[Disease Reference] Matched option content from conclusion: {content}, corresponding option: {num}")
                        return num
    
    # 3. Search for explicit option references
    option_explicit_patterns = [
        r'(?:choose|select)[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'(?:\*\*)?option\s*(\d+)(?:\*\*)?\s*is the (?:most|correct|appropriate|accurate)',
        r'i (?:think|choose|select|recommend) (?:\*\*)?option\s*(\d+)(?:\*\*)?',
    ]
    
    for text in search_texts:
        for pattern in option_explicit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    option_num = int(match.group(1))
                    if 1 <= option_num <= 10:
                        print(f"[Explicit Reference] Found option reference: Option {option_num}")
                        return option_num
                except (ValueError, IndexError):
                    continue
    
    # 4. Search for specific references to disease/option names in conclusion section
    conclusion_disease_patterns = [
        r'(?:determine|conclude)[^.,:;]*?(?:\*\*)?([^\s*().,:;]+)(?:\*\*)?\s*(?:is|as)[^.,:;]*?(?:diagnosis|selection)',
        r'(?:diagnosis|selection)[^.,:;]*?(?:is|as)[^.,:;]*?(?:\*\*)?([^\s*().,:;]+)(?:\*\*)?',
    ]
    
    for text in search_texts:
        for pattern in conclusion_disease_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                disease_name = match.group(1).strip().lower()
                # First check extended medical terms
                if disease_name in medical_term_expansions:
                    print(f"[Conclusion Reference] Matched medical term from conclusion: {disease_name}, corresponding option: {medical_term_expansions[disease_name]}")
                    return medical_term_expansions[disease_name]
                # Then check option content
                for content, num in option_content_to_num.items():
                    if content == disease_name or disease_name in content or content in disease_name:
                        print(f"[Conclusion Reference] Matched option content from conclusion: {content}, corresponding option: {num}")
                        return num
    
    
    # If extraction still fails, return None
    print("Warning: Unable to extract clear choice from response")
    return None



# Initialize debate
def initialize_debate(case_vignette, choices, force_disagree=False):
    """Initialize debate and get initial responses from three models
    
    Args:
        case_vignette: Case description
        choices: Option list
        force_disagree: Whether to force simulate disagreement (for testing)
        
    Returns:
        dict: Initial responses from three models
    """
    print("="*50)
    print("Starting Medical Case Debate")
    print("="*80)
    
    # Build mapping from option number to disease name
    choice_to_disease = {}
    choice_lines = choices.strip().split('\n')
    for line in choice_lines:
        match = re.match(r'(\d+)\.\s*(.+)', line.strip())
        if match:
            option_num = int(match.group(1))
            disease_name = match.group(2).strip()
            choice_to_disease[option_num] = disease_name
    
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
                print(f"{model_name}'s Diagnostic Conclusion:")
                if choice:
                    print(f"Choice: Option {choice} ({choice_to_disease.get(choice, 'Unknown Disease')})")
                else:
                    print("Unable to extract clear choice")
                print(f"\n{model_name}'s Complete Response:")
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
        if gpt_choice == qwen_choice == deepseek_choice and len(choice_to_disease) > 1:
            print("\nForcing model disagreement simulation (for testing)...")
            # Find a different option as Qwen's choice
            available_choices = list(choice_to_disease.keys())
            available_choices.remove(gpt_choice)
            qwen_choice = random.choice(available_choices)
            print(f"Modified Qwen's choice to: Option {qwen_choice} ({choice_to_disease.get(qwen_choice, 'Unknown Disease')})")
    
    # Check if consensus is reached
    if check_consensus([gpt_choice, qwen_choice, deepseek_choice]):
        print("\nThree models have reached consensus on initial diagnosis!")
    else:
        print("\nThree models have disagreement on initial diagnosis!")
    
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
    print(f"Using fallback response for {model_name}")
    return {
        "answer": f"{model_name} model unable to generate response. Possible API limitations or network issues.",
        "choice": None
    }


# Let Qwen respond to other models' diagnoses: QWEN --> GPT, DeepSeek
def qwen_responds_to_others(case_vignette, choices, gpt_answer, gpt_choice, deepseek_answer, deepseek_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let Qwen respond to GPT and DeepSeek's diagnoses"""
    try:
        # Get option list and build mapping from option number to disease name
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # Get disease names chosen by GPT and DeepSeek
        gpt_disease = choice_to_disease.get(gpt_choice, "Unclear Disease") if gpt_choice else "Unclear Disease"
        deepseek_disease = choice_to_disease.get(deepseek_choice, "Unclear Disease") if deepseek_choice else "Unclear Disease"

        # Get own previous choice and disease name (if any)
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "Unclear Disease")
                # Build prompt including own previous choice and analysis
        previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer}

[Your Previous Diagnostic Conclusion]: Option {self_previous_choice} ({self_previous_disease})

Please note, this was your previous diagnostic choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        prompt = f"""You are the Qwen medical model, engaged in a medical diagnostic debate with GPT model and DeepSeek model.

[Medical Case]
{case_vignette}

[Diagnostic Options]
{choices}
{previous_analysis_text}
[GPT's Complete Analysis]
{gpt_answer}

[GPT's Diagnostic Conclusion]: {gpt_disease}

[DeepSeek's Complete Analysis]
{deepseek_answer}

[DeepSeek's Diagnostic Conclusion]: {deepseek_disease}

As the Qwen medical reasoning model, you should critically evaluate the viewpoints of other models, using medical evidence and guidelines as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own. Avoid falling into the other party's cognitive traps, with the goal of improving diagnostic accuracy.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

**[Debate Guide]**  
1. **Position Statement**:  Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly overturn your diagnosis.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with GPT's viewpoint**" or  
   - "**I agree with DeepSeek's viewpoint**" or  
   - "**I agree with the shared viewpoint of GPT and DeepSeek**" (when their viewpoints align).  

2. **Evaluation of Other Models' Diagnoses**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient evidence in the arguments of other models.  

3. **Medical Analysis and Argumentation**:  
   Provide your own independent medical analysis:  
   - Supplement important information not mentioned by other models based on clinical evidence and medical literature.  
   - Analyze the likelihood of each diagnostic option and provide specific medical evidence supporting or opposing it.  
   - Explain why your analysis may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Questioning**:  
   If you consider changing your diagnosis, you must answer:  
   - Has my original diagnostic reasoning been completely refuted?  
   - Is the new diagnosis better than my original diagnosis?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X (Option content)**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Diagnoses**  
**3. Medical Analysis and Argumentation**  
**4. Self-Questioning**  
**5. Final Decision**

This is round {debate_round} of the debate. Please maintain your professional judgment unless there is conclusive evidence proving you wrong.
"""
        
        print("\nQwen is responding to other models' diagnoses...")
        
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



# Let GPT respond to other models' diagnoses: GPT --> QWEN, DeepSeek
def gpt_responds_to_others(case_vignette, choices, qwen_answer, qwen_choice, deepseek_answer, deepseek_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let GPT respond to Qwen and DeepSeek's diagnoses"""
    try:
        # Get option list and build mapping from option number to disease name
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # Get disease names chosen by Qwen and DeepSeek
        qwen_disease = choice_to_disease.get(qwen_choice, "Unclear Disease") if qwen_choice else "Unclear Disease"
        deepseek_disease = choice_to_disease.get(deepseek_choice, "Unclear Disease") if deepseek_choice else "Unclear Disease"
        
        # Get own previous choice and disease name (if any)
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "Unclear Disease")
        
        # Build prompt including own previous choice and analysis
        previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer}

[Your Previous Diagnostic Conclusion]: Option {self_previous_choice} ({self_previous_disease})

Please note, this was your previous diagnostic choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        prompt = f"""You are the GPT medical model, engaged in a medical diagnostic debate with the Qwen model and the DeepSeek model.

[Medical Case]
{case_vignette}

[Diagnostic Options]
{choices}
{previous_analysis_text}
[Qwen's Complete Analysis]
{qwen_answer}

[Qwen's Diagnostic Conclusion]: {qwen_disease}

[DeepSeek's Complete Analysis]
{deepseek_answer}

[DeepSeek's Diagnostic Conclusion]: {deepseek_disease}

As the GPT reasoning model, you should critically evaluate the viewpoints of other models, using medical evidence and guidelines as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own. Avoid falling into the other party's cognitive traps, with the goal of improving diagnostic accuracy.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

**[Debate Guide]**  
1. **Position Statement**:  Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly overturn your diagnosis.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with Qwen's viewpoint**" or  
   - "**I agree with DeepSeek's viewpoint**" or  
   - "**I agree with the shared viewpoint of Qwen and DeepSeek**" (when their viewpoints align).  

2. **Evaluation of Other Models' Diagnoses**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient evidence in the arguments of other models.  

3. **Medical Analysis and Argumentation**:  
   Provide your own independent medical analysis:  
   - Supplement important information not mentioned by other models based on clinical evidence and medical literature.  
   - Analyze the likelihood of each diagnostic option and provide specific medical evidence supporting or opposing it.  
   - Explain why your analysis may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Questioning**:  
   If you consider changing your diagnosis, you must answer:  
   - Has my original diagnostic reasoning been completely refuted?  
   - Is the new diagnosis better than my original diagnosis?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X (Option content)**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Diagnoses**  
**3. Medical Analysis and Argumentation**  
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



# Let DeepSeek respond to other models' diagnoses: DeepSeek --> GPT, QWEN
def deepseek_responds_to_others(case_vignette, choices, gpt_answer, gpt_choice, qwen_answer, qwen_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """Let DeepSeek respond to GPT and Qwen's diagnoses"""
    try:
        # Get option list and build mapping from option number to disease name
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # Get disease names chosen by GPT and Qwen
        gpt_disease = choice_to_disease.get(gpt_choice, "Unclear Disease") if gpt_choice else "Unclear Disease"
        qwen_disease = choice_to_disease.get(qwen_choice, "Unclear Disease") if qwen_choice else "Unclear Disease"
        
        # Get own previous choice and disease name (if any)
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "Unclear Disease")
        
        # Build prompt including own previous choice and analysis
        previous_analysis_text = f"""
[Your Previous Complete Analysis]
{self_previous_answer}

[Your Previous Diagnostic Conclusion]: Option {self_previous_choice} ({self_previous_disease})

Please note, this was your previous diagnostic choice. Carefully consider the basis of your previous analysis. Unless there is conclusive evidence proving you wrong, you should maintain your professional judgment.
"""
        
        prompt = f"""You are the DeepSeek medical model, engaged in a medical diagnostic debate with GPT model and Qwen model.

[Medical Case]
{case_vignette}

[Diagnostic Options]
{choices}
{previous_analysis_text}
[GPT's Complete Analysis]
{gpt_answer}

[GPT's Diagnostic Conclusion]: {gpt_disease}

[Qwen's Complete Analysis]
{qwen_answer}

[Qwen's Diagnostic Conclusion]: {qwen_disease}

As the DeepSeek reasoning model, you should critically evaluate the viewpoints of other models, using medical evidence and guidelines as the basis for decision-making. Trust your prior professional judgment and adjust your conclusions only when the opposing party presents conclusive evidence that is superior to your own. Avoid falling into the other party's cognitive traps, with the goal of improving diagnostic accuracy.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

**[Debate Guide]**  
1. **Position Statement**:  Be sure to uphold your professional stance: do not be easily persuaded. Assess whether the arguments of other models truly overturn your diagnosis.  
   Clearly state your position by beginning your response in the following format:  
   - "**I disagree with their viewpoint because:**" or  
   - "**I agree with GPT's viewpoint**" or  
   - "**I agree with Qwen's viewpoint**" or  
   - "**I agree with the shared viewpoint of GPT and Qwen**" (when their viewpoints align).  

2. **Evaluation of Other Models' Diagnoses**: Conduct a critical analysis, pointing out in detail the flaws, misinterpretations, or insufficient evidence in the arguments of other models.  

3. **Medical Analysis and Argumentation**:  
   Provide your own independent medical analysis:  
   - Supplement important information not mentioned by other models based on clinical evidence and medical literature.  
   - Analyze the likelihood of each diagnostic option and provide specific medical evidence supporting or opposing it.  
   - Explain why your analysis may be more accurate or comprehensive (if you disagree with the conclusions of other models).  

4. **Self-Questioning**:  
   If you consider changing your diagnosis, you must answer:  
   - Has my original diagnostic reasoning been completely refuted?  
   - Is the new diagnosis better than my original diagnosis?  

5. **Final Decision**: Must conclude with "**My final selection is: Option X (Option content)**".  

Please respond in the following format:  

**1. Position Statement**  
**2. Evaluation of Other Models' Diagnoses**  
**3. Medical Analysis and Argumentation**  
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
            print(f"DeepSeek answer generation completed, time: {t_generate:.2f}s")
            # Extract DeepSeek's choice from the generated response
            choice = extract_model_choice(answer, choices)
            
        except Exception as e:
            print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
            print("Trying to call API directly using requests...")
            
            # Backup plan: Direct requests call, SiliconFlow API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek medical reasoning model, engaged in an intense debate with other models."},
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
                print(f"DeepSeek answer generation completed, time: {t_generate:.2f}s")
                choice = extract_model_choice(answer, choices)
            else:
                print(f"DeepSeek API error: {response.status_code}")
                print(f"Error details: {response.text}")
                return fallback_response("deepseek")
        
        # Ensure output of response content regardless of which method was used to get the answer
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
            print("Failed to get DeepSeek response")
            return fallback_response("deepseek")
    
    except Exception as e:
        print(f"Error generating DeepSeek response: {str(e)}")
        traceback.print_exc()
        return fallback_response("deepseek")





# Let models conduct debate and determine if consensus is reached
def conduct_debate(case_vignette, choices, correct_answer, max_rounds=3, force_disagree=False):
    """Conduct debate between models
    
    Args:
        case_vignette: Case description
        choices: Option list
        correct_answer: Correct answer
        max_rounds: Maximum debate rounds
        force_disagree: Whether to force simulate disagreement (for testing)
        
    Returns:
        dict: Debate results, including final choice and debate history
    """
    try:
        # Get initial responses from three models
        initial_results = initialize_debate(case_vignette, choices, force_disagree)
        
        if not initial_results:
            print("Failed to initialize debate, cannot continue")
            return None
        
        gpt_result = initial_results["gpt"]
        qwen_result = initial_results["qwen"]
        deepseek_result = initial_results["deepseek"]
        
        # Record initial choices for evaluating final results
        initial_gpt_choice = gpt_result["choice"]
        initial_qwen_choice = qwen_result["choice"]
        initial_deepseek_choice = deepseek_result["choice"]
        
        # Record which model is more accurate (if we know the correct answer)
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
            
            # Find option number corresponding to correct answer
            correct_choice = None
            exact_match = False
            
            # First round: find exact match
            for number, disease in option_mapping.items():
                if disease.lower() == correct_answer.lower() or disease.strip().lower() == correct_answer.lower():
                    correct_choice = number
                    exact_match = True
                    break
            
            # Second round: if no exact match, find word boundary match
            if not exact_match:
                for number, disease in option_mapping.items():
                    # Use regex for word boundary matching
                    if re.search(r'\b' + re.escape(correct_answer.lower()) + r'\b', disease.lower()):
                        correct_choice = number
                        break
            
            # Third round: if first two rounds fail, use stricter partial matching (as backup only)
            if not correct_choice:
                # Sort by option number to ensure priority matching order
                sorted_options = sorted(option_mapping.items())
                for number, disease in sorted_options:
                    # Only consider complete match for short options, avoid matching "X" to "XIIa"
                    if correct_answer.lower() in disease.lower():
                        # Additional verification: if single character answer, ensure it's independent
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
            print(f"\nThree models have reached initial consensus! All models chose option {consensus_choice}")
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
            print(f"\nThree models have initial disagreement, starting debate process... GPT chose option {gpt_result['choice']}, Qwen chose option {qwen_result['choice']}, DeepSeek chose option {deepseek_result['choice']}")
        
        # Store debate history
        debate_history = [{
            "round": 0,
            "gpt": gpt_result,
            "qwen": qwen_result,
            "deepseek": deepseek_result
        }]
        
        # Build mapping from option number to disease name
        choice_to_disease = {}
        choice_options = choices.strip().split('\n')
        for option in choice_options:
            match = re.match(r'(\d+)\.\s*(.+)', option.strip())
            if match:
                number, disease = match.groups()
                choice_to_disease[int(number)] = disease.strip()
        
        # Track latest results of each model in each round of debate
        current_gpt_result = gpt_result
        current_qwen_result = qwen_result
        current_deepseek_result = deepseek_result


        # Start debate
        for round_num in range(1, max_rounds + 1):
            print(f"\n======== Debate Round {round_num} ========")
            
            # GPT responds to Qwen and DeepSeek, passing its previous answer and choice
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
            gpt_disease = choice_to_disease.get(gpt_choice, "Unknown Disease") if gpt_choice else "Unclear Disease"
            qwen_disease = choice_to_disease.get(qwen_result["choice"], "Unknown Disease") if qwen_result["choice"] else "Unclear Disease"
            deepseek_disease = choice_to_disease.get(deepseek_result["choice"], "Unknown Disease") if deepseek_result["choice"] else "Unclear Disease"
            
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
                consensus_disease = choice_to_disease.get(consensus_choice, "Unknown Disease")
                print(f"\nDebate Round {round_num}: All models have reached consensus! (All chose option {consensus_choice} - {consensus_disease})")
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
                        "qwen_changed": initial_qwen_choice != qwen_result["choice"],  # Modified to qwen_result
                        "deepseek_changed": initial_deepseek_choice != deepseek_result["choice"],
                        "gpt_changed_from_correct": gpt_initially_correct and gpt_response["choice"] != correct_choice if correct_choice else False,
                        "qwen_changed_from_correct": qwen_initially_correct and qwen_result["choice"] != correct_choice if correct_choice else False,  # Modified to qwen_result
                        "deepseek_changed_from_correct": deepseek_initially_correct and deepseek_result["choice"] != correct_choice if correct_choice else False
                    },
                    "correct_choice": correct_choice
                }
            
            # Qwen responds to GPT and DeepSeek, passing its previous answer and choice
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
            qwen_disease = choice_to_disease.get(qwen_choice, "Unknown Disease") if qwen_choice else "Unclear Disease"
            
            print(f"Qwen's choice after response: Option {qwen_choice} ({qwen_disease})")
            print(f"GPT's choice: Option {gpt_response['choice']} ({gpt_disease})")
            print(f"DeepSeek's choice: Option {deepseek_result['choice']} ({deepseek_disease})")
            
            # If Qwen changed stance and was originally correct
            if correct_choice and qwen_initially_correct and qwen_response["choice"] != correct_choice:
                print(f"Warning: Qwen changed from correct choice ({choice_to_disease.get(correct_choice, '')}) to incorrect choice ({qwen_disease})")
            
            # Check if consensus is reached
            current_choices = [gpt_response["choice"], qwen_response["choice"], deepseek_result["choice"]]
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_disease = choice_to_disease.get(consensus_choice, "Unknown Disease")
                print(f"\nDebate Round {round_num}: All models have reached consensus! (All chose option {consensus_choice} - {consensus_disease})")
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
            
            # DeepSeek responds to GPT and Qwen, passing its own previous answer and choice
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
            deepseek_disease = choice_to_disease.get(deepseek_choice, "Unknown Disease") if deepseek_choice else "Unclear Disease"
            
            # Ensure DeepSeek's response content has been output, if not, output it here again
            # This code is just a backup, normally deepseek_responds_to_others function has already output the response content
            if '_output_shown' not in deepseek_response:
                print(f"DeepSeek's choice after response: Option {deepseek_choice} ({deepseek_disease})")
                if "answer" in deepseek_response and deepseek_response["answer"]:
                    print("\nDeepSeek's response to other models:")
                    print("="*80)
                    print(deepseek_response["answer"])
                    print("="*80)
                deepseek_response['_output_shown'] = True
            
            print(f"GPT's choice: Option {gpt_response['choice']} ({gpt_disease})")
            print(f"Qwen's choice: Option {qwen_response['choice']} ({qwen_disease})")
            
            # If DeepSeek changed stance and was originally correct
            if correct_choice and deepseek_initially_correct and deepseek_response["choice"] != correct_choice:
                print(f"Warning: DeepSeek changed from correct choice ({choice_to_disease.get(correct_choice, '')}) to incorrect choice ({deepseek_disease})")
            
            # Check if consensus is reached
            current_choices = [gpt_response["choice"], qwen_response["choice"], deepseek_response["choice"]]

            # Output diagnostic information here for debugging stance changes
            if initial_deepseek_choice != deepseek_response["choice"]:
                print(f"\nDeepSeek changed its stance! From option {initial_deepseek_choice} to option {deepseek_response['choice']}")
                
                # If DeepSeek changed from wrong to correct
                if correct_choice and not deepseek_initially_correct and deepseek_response["choice"] == correct_choice:
                    print(f"DeepSeek successfully corrected its error! From option {initial_deepseek_choice} corrected to correct option {deepseek_response['choice']}")
                
                # If DeepSeek changed from correct to wrong
                if correct_choice and deepseek_initially_correct and deepseek_response["choice"] != correct_choice:
                    print(f"Warning: DeepSeek changed from correct option {initial_deepseek_choice} to wrong option {deepseek_response['choice']}")            
            
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_disease = choice_to_disease.get(consensus_choice, "Unknown Disease")
                print(f"\nDebate Round {round_num}: All models have reached consensus! (All chose option {consensus_choice} - {consensus_disease})")
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
            
            print(f"\nDebate Round {round_num}: Still no consensus, GPT chose option {gpt_response['choice']} ({gpt_disease}), Qwen chose option {qwen_response['choice']} ({qwen_disease}), DeepSeek chose option {deepseek_response['choice']} ({deepseek_disease})")
        
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
        
        # Find option(s) with most votes
        if not choice_counts:
            print("All models failed to provide clear choices, cannot determine final result")
            return None
            
        max_votes = max(choice_counts.values())
        max_choices = [choice for choice, count in choice_counts.items() if count == max_votes]
        
        # Check for tie situation
        if len(max_choices) > 1:
            # If votes are tied, mark as invalid debate
            print(f"Tied votes, invalid debate! Vote distribution: {choice_counts}")
            print("No consensus reached after maximum rounds and voting resulted in a tie.")
            print(f"Models chose: GPT-Option {gpt_final_choice}, Qwen-Option {qwen_final_choice}, DeepSeek-Option {deepseek_final_choice}")
            return None
        else:
            # Clear majority winner
            final_choice = max_choices[0]
        
        final_disease = choice_to_disease.get(final_choice, "Unknown Disease")
        print(f"\nFinal choice (majority vote): Option {final_choice} ({final_disease})")
        
        # Check if any model changed from correct to incorrect
        if correct_choice:
            gpt_changed_from_correct = (gpt_initially_correct and gpt_final_choice != correct_choice)
            qwen_changed_from_correct = (qwen_initially_correct and qwen_final_choice != correct_choice)
            deepseek_changed_from_correct = (deepseek_initially_correct and deepseek_final_choice != correct_choice)
            
            if gpt_changed_from_correct:
                gpt_final_disease = choice_to_disease.get(gpt_final_choice, "Unknown Disease")
                print(f"Warning: GPT changed from correct choice ({choice_to_disease.get(correct_choice, '')}) to incorrect choice ({gpt_final_disease})")
            
            if qwen_changed_from_correct:
                qwen_final_disease = choice_to_disease.get(qwen_final_choice, "Unknown Disease")
                print(f"Warning: Qwen changed from correct choice ({choice_to_disease.get(correct_choice, '')}) to incorrect choice ({qwen_final_disease})")
            
            if deepseek_changed_from_correct:
                deepseek_final_disease = choice_to_disease.get(deepseek_final_choice, "Unknown Disease")
                print(f"Warning: DeepSeek changed from correct choice ({choice_to_disease.get(correct_choice, '')}) to incorrect choice ({deepseek_final_disease})")
            
            is_final_correct = (final_choice == correct_choice)
            correct_disease = choice_to_disease.get(correct_choice, "Unknown Disease")
            print(f"Correct diagnosis: Option {correct_choice} ({correct_disease})")
            print(f"Is final choice correct: {'Correct ✓' if is_final_correct else 'Incorrect ✗'}")
        
        # Diagnostic information showing final DeepSeek stance change
        if initial_deepseek_choice != deepseek_final_choice:
            print(f"\nAt the end of debate, DeepSeek changed its stance: from option {initial_deepseek_choice} to option {deepseek_final_choice}")
            if correct_choice and not deepseek_initially_correct and deepseek_final_choice == correct_choice:
                print(f"DeepSeek successfully corrected its error! From option {initial_deepseek_choice} to correct option {deepseek_final_choice}")
        
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



# Evaluate if answer is correct
def evaluate_answer(dataset, case_idx, model_choice):
    """Evaluate if model choice is correct
    
    Args:
        dataset: Dataset
        case_idx: Case index
        model_choice: Option number chosen by model
    
    Returns:
        bool: Whether correct
    """
    try:
        if model_choice is None:
            return False
            
        # Get correct answer
        correct_answer = None
        
        # Check answer_idx field (usually present in JSONL format)
        if "answer_idx" in dataset.columns:
            correct_idx = dataset.iloc[case_idx]["answer_idx"]
            # Convert letter answer index to number
            if isinstance(correct_idx, str) and len(correct_idx) == 1 and correct_idx.isalpha():
                correct_answer_num = ord(correct_idx.upper()) - ord('A') + 1
                return model_choice == correct_answer_num
        
        # Directly check answer field
        if "answer" in dataset.columns:
            correct_text = dataset.iloc[case_idx]["answer"]
            
            # Get option list
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
            
            # Check if model's chosen option content matches correct answer text
            if model_choice in option_mapping:
                model_answer_text = option_mapping[model_choice].lower()
                if correct_text.lower() == model_answer_text:
                    return True
                
            # Check if correct answer text matches any option content
            if correct_text.lower() in option_content_to_num:
                correct_answer_num = option_content_to_num[correct_text.lower()]
                return model_choice == correct_answer_num
                
        # Fallback: if all above methods fail, try original method
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



# Process debate for single case
def process_single_debate(dataset_path, case_idx=0, max_rounds=3, force_disagree=False):
    """Process debate for a single medical MCQ case
    
    Args:
        dataset_path: Dataset path
        case_idx: Case index
        max_rounds: Maximum debate rounds
        force_disagree: Whether to force simulate disagreement (for testing)
        
    Returns:
        dict: Debate results
    """
    try:
        print(f"Processing single case debate (index: {case_idx})...")
        
        # Load dataset
        dataset = load_medical_mcq_data(dataset_path)
        
        # Check if index is valid
        if case_idx < 0 or case_idx >= len(dataset):
            print(f"Error: index {case_idx} out of range, dataset contains {len(dataset)} cases")
            return None
        
        # Get case data - adjust fields to adapt to JSONL format
        case_id = dataset.loc[case_idx, "question_id"] if "question_id" in dataset.columns else f"case_{case_idx}"
        case_vignette = dataset.loc[case_idx, "question"]  # JSONL uses question instead of case_vignette
        category = dataset.loc[case_idx, "meta_info"] if "meta_info" in dataset.columns else "Unknown Category"
        choices = get_choices(dataset, case_idx)
        correct_answer = dataset.loc[case_idx, "answer"]
        
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
        
        # Final diagnosis disease
        final_disease = choice_to_disease.get(final_choice, "Unknown Disease") if final_choice else "Unclear Disease"
        
        # Evaluate if result is correct
        is_correct = evaluate_answer(dataset, case_idx, final_choice)
        
        # Output final results
        print("\n========= Final Debate Results =========")
        if result["consensus"]:
            print(f"GPT, Qwen and DeepSeek-R1 reached consensus! Final diagnosis: Option {final_choice} - {final_disease}")
        else:
            # No consensus after three rounds, decided by majority vote
            print(f"GPT, Qwen and DeepSeek-R1 did not reach consensus")
            print(f"Decided by majority vote, final diagnosis: Option {final_choice} - {final_disease}")
        
        print(f"Correct answer: Option? - {correct_answer}")
        print(f"Is final diagnosis correct: {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        # Build output results
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
        # Set up output redirection to terminal and log file
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        sys.stdout = TeeOutput(log_file)
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='MCC Framework for MedQA Medical Question Answering')
        parser.add_argument('-n', '--num_cases', type=int, default=1273, help='Number of cases to process (default: 1273)')
        parser.add_argument('-s', '--start_idx', type=int, default=0, help='Starting case index (default: 0)')
        parser.add_argument('-f', '--force_disagree', action='store_true', help='Force simulate model disagreement to test judge function')
        parser.add_argument('--force_judge', action='store_true', help='Force use DeepSeek judge regardless of debate results')
        args = parser.parse_args()
        
        # Dataset path
        dataset_path = "../benchmarks/MedQA/MedQA_test.jsonl"
        
        # Check if dataset exists
        if not check_file_exists(dataset_path):
            print("Program terminated, dataset does not exist.")
            exit(1)
        
        # Process multiple case debates
        print("="*50)
        print("GPT vs Qwen vs DeepSeek-R1 Medical Diagnosis Debate, Model Confrontation and Collaboration Officially Begins")
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
                    # Save single case result to JSON file
                    case_result_file = os.path.join(results_dir, f"debate_result_case_{i}.json")
                    with open(case_result_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"\nDebate result for case {i} has been saved to {case_result_file}")
                    
                    # Add to summary results
                    summary_result = {
                        "case_id": result["case_id"],
                        "category": result["category"],
                        "correct_answer": result["correct_answer"],
                        "consensus": result["debate_result"]["consensus"],
                        "voting_needed": not result["debate_result"]["consensus"],  # If no consensus reached, majority voting needed
                        "final_choice": result["debate_result"]["final_choice"],
                        "is_correct": result["is_correct"],
                        "stance_changes": result["debate_result"].get("stance_changes", {}),
                        "debate_result": {  # To maintain compatibility with statistical code, add complete debate_result
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
            voting_cases = sum(1 for r in summary_results if not r["consensus"])  # Cases without consensus need majority voting
            
            # Count model stance changes
            gpt_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("gpt_changed", False))
            qwen_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("qwen_changed", False))
            deepseek_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("deepseek_changed", False))
            
            # Count cases of changing from correct to incorrect
            gpt_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("gpt_changed_from_correct", False))
            qwen_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("qwen_changed_from_correct", False))
            deepseek_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("deepseek_changed_from_correct", False))
            
            # Count cases of changing from incorrect to correct (positive cases of model self-correction)
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
            print(f"Models reached consensus cases: {consensus_cases} ({consensus_cases/total_cases:.2%})")
            print(f"Majority voting needed cases: {voting_cases} ({voting_cases/total_cases:.2%})")
            print("-" * 40)  # Separator line
            # Model stance change statistics
            print(f"GPT changed stance cases: {gpt_changed_stance} ({gpt_changed_stance/total_cases:.2%})")
            print(f"Qwen changed stance cases: {qwen_changed_stance} ({qwen_changed_stance/total_cases:.2%})")
            print(f"DeepSeek changed stance cases: {deepseek_changed_stance} ({deepseek_changed_stance/total_cases:.2%})")
            print("-" * 40)  # Separator line
            # Statistics of changing from correct to incorrect (negative impact)
            print(f"GPT changed from correct to incorrect cases: {gpt_changed_from_correct} ({gpt_changed_from_correct/total_cases:.2%})")
            print(f"Qwen changed from correct to incorrect cases: {qwen_changed_from_correct} ({qwen_changed_from_correct/total_cases:.2%})")
            print(f"DeepSeek changed from correct to incorrect cases: {deepseek_changed_from_correct} ({deepseek_changed_from_correct/total_cases:.2%})")
            print("-" * 40)  # Separator line
            # Statistics of changing from incorrect to correct (positive impact)
            print(f"GPT changed from incorrect to correct cases: {gpt_changed_to_correct} ({gpt_changed_to_correct/total_cases:.2%})")
            print(f"Qwen changed from incorrect to correct cases: {qwen_changed_to_correct} ({qwen_changed_to_correct/total_cases:.2%})")
            print(f"DeepSeek changed from incorrect to correct cases: {deepseek_changed_to_correct} ({deepseek_changed_to_correct/total_cases:.2%})")
            print("-" * 40)  # Separator line
            print(f"\nSummary results have been saved to {summary_file}")
            print(f"Log file has been saved to {log_file}")
        
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

# Running examples:
# python ThreeLLM.py -n 3                         # Process first 3 cases
# python ThreeLLM.py -n 5 -s 10                   # Process 5 cases with index 10-14
# python ThreeLLM.py --num_cases=3 --start_idx=5  # Process 3 cases with index 5-7 