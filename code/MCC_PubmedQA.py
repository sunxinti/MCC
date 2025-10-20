
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
from tqdm import tqdm
from openai import OpenAI
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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
        print("Configuration complete! Starting PubMedQA medical model debate system...")
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

def get_session_with_retry(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def send_api_request(url, headers, data, model_name, max_retries=3, retry_delay=2):
    session = get_session_with_retry(retries=max_retries)
    
    for attempt in range(max_retries):
        try:
            response = session.post(url, headers=headers, json=data, timeout=180)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limiting
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"{model_name} API rate limited, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"{model_name} API error: {response.status_code}")
                print(f"Error details: {response.text}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Maximum retry attempts reached, giving up request")
                    return None
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"{model_name} API request exception: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Maximum retry attempts reached, giving up request")
                return None
    
    return None

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

class CaseLogOutput:
    """Class for creating separate log files for individual cases"""
    def __init__(self, case_id):
        self.terminal = sys.stdout
        self.case_id = case_id
        log_filename = f"{case_id}_log.txt"
        self.logfile = open(log_filename, "w", encoding="utf-8")
        # Add header to case log
        self.logfile.write(f"{'='*60}\n")
        self.logfile.write(f"PubMedQA Case Log - PMID: {case_id}\n")
        self.logfile.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logfile.write(f"{'='*60}\n\n")
        self.logfile.flush()
        
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
        
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
        
    def close(self):
        """Close log file with footer"""
        if self.logfile:
            self.logfile.write(f"\n{'='*60}\n")
            self.logfile.write(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.logfile.write(f"{'='*60}\n")
            self.logfile.close()
            self.logfile = None
        
    def get_filename(self):
        return f"{self.case_id}_log.txt"

def check_file_exists(file_path):
    """Check if file exists and provide detailed error information"""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist!")
        return False
    return True


# Load PubMedQA dataset
def load_pubmedqa_data(file_path):
    if not check_file_exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

# Fixed options are yes/no/maybe
def get_choices():
    """
    Get predefined three answer options
    
    Returns:
        list: List of three standardized answer options ["yes", "no", "maybe"]
    """
    return ["yes", "no", "maybe"]


# ===================== GPT Model Section =====================
def get_gpt_prompt(pmid, question, context):
    prompt = f"""You have received a PubMed context as abstract. Your task is to analyze the following medical research question and provide a well-reasoned answer with "yes", "no", or "maybe" based solely on the information in the abstract.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

This will assess your ability to understand and reason about scientific biomedical literature. Use ONLY information in this abstract - do not add background knowledge or guesses. Please carefully reason through the following steps to provide your answer:

PMID: {pmid}
Question: {question}
Abstract: {context}

1. Question Analysis
   - **Carefully read and fully understand what the question is asking:**
     * What is the core question being asked?
     * What specific comparison, relationship, or outcome is being examined?
     * If multiple elements are mentioned, clarify the logical relationship between them
   - Identify key concepts and terms

2. Evidence Evaluation
   - Analyze the research results and data in the abstract
   - Extract the core conclusion from the abstract.
   - Determine the strength and reliability of this evidence
   - Assess the relevance of the evidence to the question
   - **Based on the evidence, which answer do you lean towards and why?**
   - **For your preferred choice, explicitly analyze:**
     * **Supporting evidence**: What findings support this choice?
   - **Use elimination reasoning:**
     * Can you definitively rule out the other two options?

3. Conclusion
   - Based on the above analysis, provide a "yes", "no", or "maybe" answer
   - Explain the reasoning for your choice
   - Remember, strong scientific reasoning requires considering evidence both for AND against a claim

4. Final Answer:
-[EXTREMELY IMPORTANT] Your final choice must use the exact format below, otherwise it will not be correctly recognized by the system:
**My final choice is: "yes", "no", or "maybe"**

Please respond in the following format:
**1. Question Analysis**
**2. Evidence Evaluation**
**3. Conclusion**
**4. Final Answer**
"""
    return prompt

# Generate GPT answer
def generate_gpt_answer(pmid, question, context):
    try:
        prompt = get_gpt_prompt(pmid, question, context)
        
        print("\nGPT is reasoning the answer...")
        t_generate_start = time.time()
        
        data = {
            "model": "o1-mini", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are a GPT medical model, a medical expert with strong clinical reasoning abilities. You make judgments based on medical evidence and professional knowledge."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 8000
        }

        response_data = send_api_request(GPT_API_URL, GPT_HEADERS, data, "GPT")
        t_generate = time.time() - t_generate_start
        
        if response_data:
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"GPT answer generation completed, time taken: {t_generate:.2f} seconds")
            return answer
        else:
            raise Exception("API returned empty, please check connection and API key")

    except Exception as e:
        print(f"Error generating GPT answer: {str(e)}")
        traceback.print_exc()
        raise Exception(f"GPT API call failed: {str(e)}")



# ===================== Qwen Model Section =====================
def get_qwen_prompt(pmid, question, context):
    prompt = f"""You have received a PubMed context as abstract. Your task is to analyze the following medical research question and provide a well-reasoned answer with "yes", "no", or "maybe" based solely on the information in the abstract.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

This will assess your ability to understand and reason about scientific biomedical literature. Use ONLY information in this abstract - do not add background knowledge or guesses. Please carefully reason through the following steps to provide your answer:

PMID: {pmid}
Question: {question}
Abstract: {context}

1. Question Analysis
   - **Carefully read and fully understand what the question is asking:**
     * What is the core question being asked?
     * What specific comparison, relationship, or outcome is being examined?
     * If multiple elements are mentioned, clarify the logical relationship between them
   - Identify key concepts and terms

2. Evidence Evaluation
   - Analyze the research results and data in the abstract
   - Extract the core conclusion from the abstract.
   - Determine the strength and reliability of this evidence
   - Assess the relevance of the evidence to the question
   - **Based on the evidence, which answer do you lean towards and why?**
   - **For your preferred choice, explicitly analyze:**
     * **Supporting evidence**: What findings support this choice?
   - **Use elimination reasoning:**
     * Can you definitively rule out the other two options?

3. Conclusion
   - Based on the above analysis, provide a "yes", "no", or "maybe" answer
   - Explain the reasoning for your choice
   - Remember, strong scientific reasoning requires considering evidence both for AND against a claim

4. Final Answer:
-[EXTREMELY IMPORTANT] Your final choice must use the exact format below, otherwise it will not be correctly recognized by the system:
**My final choice is: "yes", "no", or "maybe"**

Please respond in the following format:
**1. Question Analysis**
**2. Evidence Evaluation**
**3. Conclusion**
**4. Final Answer**
"""
    return prompt

# Generate Qwen answer
def generate_qwen_answer(pmid, question, context):
    try:
        prompt = get_qwen_prompt(pmid, question, context)
        
        print("\nQwen is generating answer...")
        t_generate_start = time.time()
        
        # Build request data
        data = {
            "model": "Qwen/QwQ-32B", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are the Qwen medical model, a medical expert with strong clinical reasoning abilities. You make independent judgments based on medical evidence and professional knowledge."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }

        response_data = send_api_request(QWEN_API_URL, QWEN_HEADERS, data, "Qwen")
        t_generate = time.time() - t_generate_start
        
        if response_data:
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"Qwen answer generation completed, time taken: {t_generate:.2f} seconds")
            return answer
        else:
            raise Exception("API returned empty, please check connection and API key")

    except Exception as e:
        print(f"Error generating Qwen answer: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Qwen API call failed: {str(e)}")


# ===================== DeepSeek Model Section =====================
def get_deepseek_prompt(pmid, question, context):
    prompt = f"""You have received a PubMed context as abstract. Your task is to analyze the following medical research question and provide a well-reasoned answer with "yes", "no", or "maybe" based solely on the information in the abstract.
**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.**

This will assess your ability to understand and reason about scientific biomedical literature. Use ONLY information in this abstract - do not add background knowledge or guesses. Please carefully reason through the following steps to provide your answer:

PMID: {pmid}
Question: {question}
Abstract: {context}

1. Question Analysis
   - **Carefully read and fully understand what the question is asking:**
     * What is the core question being asked?
     * What specific comparison, relationship, or outcome is being examined?
     * If multiple elements are mentioned, clarify the logical relationship between them
   - Identify key concepts and terms

2. Evidence Evaluation
   - Analyze the research results and data in the abstract
   - Extract the core conclusion from the abstract.
   - Determine the strength and reliability of this evidence
   - Assess the relevance of the evidence to the question
   - **Based on the evidence, which answer do you lean towards and why?**
   - **For your preferred choice, explicitly analyze:**
     * **Supporting evidence**: What findings support this choice?
   - **Use elimination reasoning:**
     * Can you definitively rule out the other two options?

3. Conclusion
   - Based on the above analysis, provide a "yes", "no", or "maybe" answer
   - Explain the reasoning for your choice
   - Remember, strong scientific reasoning requires considering evidence both for AND against a claim

4. Final Answer:
-[EXTREMELY IMPORTANT] Your final choice must use the exact format below, otherwise it will not be correctly recognized by the system:
**My final choice is: "yes", "no", or "maybe"**

Please respond in the following format:
**1. Question Analysis**
**2. Evidence Evaluation**
**3. Conclusion**
**4. Final Answer**
"""
    return prompt

# Generate DeepSeek answer
def generate_deepseek_answer(pmid, question, context):
    try:
        prompt = get_deepseek_prompt(pmid, question, context)
        
        print("\nDeepSeek is generating answer...")
        t_generate_start = time.time()
        
        # First try direct API call using requests
        data = {
            "model": "deepseek-reasoner", # Note, use the same as the original study; you can replace it with other LLMs. 
            "messages": [
                {"role": "system", "content": "You are the DeepSeek medical model, a medical expert with strong clinical reasoning capabilities. You make independent judgments based on medical evidence and professional knowledge."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }
        
        response_data = send_api_request(DEEPSEEK_API_URL + "/v1/chat/completions", DEEPSEEK_HEADERS, data, "DeepSeek")
        
        # If main API call fails, try backup API (SiliconFlow)
        if not response_data:
            print("Trying backup API...")
            backup_data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1", # Note, use the same as the original study; you can replace it with other LLMs. 
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek medical model, a medical expert with strong clinical reasoning capabilities. You make independent judgments based on medical evidence and professional knowledge."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 8000
            }
            backup_url = "https://api.siliconflow.cn/v1/chat/completions"
            backup_headers = {
                "Authorization": f"Bearer {QWEN_API_KEY}",  # Use Qwen API key to access SiliconFlow
                "Content-Type": "application/json"
            }
            response_data = send_api_request(backup_url, backup_headers, backup_data, "DeepSeek(Backup)")
        
        t_generate = time.time() - t_generate_start
        
        if response_data:
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"DeepSeek answer generation completed, time taken: {t_generate:.2f} seconds")
            return answer
        else:
            raise Exception("API returned empty, please check connection and API key")
        
    except Exception as e:
        print(f"Error generating DeepSeek answer: {str(e)}")
        traceback.print_exc()
        raise Exception(f"DeepSeek API call failed: {str(e)}")



# Extract model choice from answer
def extract_model_choice(answer_text):
    answer_text = answer_text.lower()
    
    patterns = [
        # Original patterns
        r'\*\*my final choice is: "([a-z]+)"\*\*',
        r'\*\*my final choice is: ([a-z]+)\*\*',
        r'\*\*my final choice is:"([a-z]+)"\*\*', 
        r'\*\*my final choice is:([a-z]+)\*\*',
        r'my final choice is: "([a-z]+)"',
        r'my final choice is: ([a-z]+)',
        r'final choice is: "([a-z]+)"',
        r'final choice is: ([a-z]+)',
        
        # Additional numbered format patterns
        r'\*\*4\.\s*final answer:?\*\*\s*\*\*([a-z]+)\*\*',
        r'4\.\s*final answer:?\s*\*\*([a-z]+)\*\*',
        r'\*\*4\.\s*final answer:?\*\*\s*([a-z]+)',
        r'4\.\s*final answer:?\s*([a-z]+)',
        
        # Standalone answer format
        r'\*\*([a-z]+)\*\*\s*$'  # Match standalone bold answer at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer_text)
        if match:
            choice = match.group(1)
            if choice in ["yes", "no", "maybe"]:
                return choice
    
    # If none of the above patterns match, look for standalone yes/no/maybe words
    for choice in ["yes", "no", "maybe"]:
        if re.search(r'\*\*' + choice + r'\*\*', answer_text):
            return choice
    
    # Print warning and raise exception
    print("WARNING: Unable to extract choice from answer. Please check if answer format meets requirements.")
    print("Answer excerpt:")
    # Print last 200 characters as reference
    print(answer_text[-200:] if len(answer_text) > 200 else answer_text)
    
    # Raise exception instead of returning a default value
    raise ValueError("Failed to extract valid choice from model response")




# Initialize debate
def initialize_debate(pmid, question, context, force_disagree=False):
    print("\nn======================= Initialize Debate =======================")
    print(f"PMID: {pmid}")
    print(f"Question: {question}")
    
    # Start parallel execution to get initial responses from three models
    print("Starting parallel execution to get initial responses from three models")
    start_time = time.time()
    
    # Initialize results storage
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks for parallel execution
        gpt_future = executor.submit(generate_gpt_answer, pmid, question, context)
        qwen_future = executor.submit(generate_qwen_answer, pmid, question, context)
        deepseek_future = executor.submit(generate_deepseek_answer, pmid, question, context)
        
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
                choice = extract_model_choice(answer)
                
                # Store results
                results[model_key] = {
                    "answer": answer,
                    "choice": choice
                }
                
                # Immediate output as each model completes
                print(f"\n----- {model_name} Initial Answer -----")
                print(answer)
                print(f"{model_name} Choice: {choice}")
                
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
    
    # If forced disagreement and all models give same answer, modify one model's answer
    if force_disagree and gpt_choice == qwen_choice == deepseek_choice:
        other_choices = [c for c in ["yes", "no", "maybe"] if c != gpt_choice]
        deepseek_choice = random.choice(other_choices)
        print(f"Forced disagreement: Deepseek choice modified to {deepseek_choice}")
    
    return {
        "gpt": {"answer": gpt_answer, "choice": gpt_choice},
        "qwen": {"answer": qwen_answer, "choice": qwen_choice},
        "deepseek": {"answer": deepseek_answer, "choice": deepseek_choice}
    }

# Check if consensus is reached
def check_consensus(choices):
    if choices["gpt"] == choices["qwen"] == choices["deepseek"]:
        return True, choices["gpt"]
    
    # If two models agree, use majority vote
    if choices["gpt"] == choices["qwen"]:
        return False, choices["gpt"]
    elif choices["gpt"] == choices["deepseek"]:
        return False, choices["gpt"]
    elif choices["qwen"] == choices["deepseek"]:
        return False, choices["qwen"]
    
    # All models disagree, return no consensus
    return False, None




# Conduct debate
def conduct_debate(pmid, question, context, ground_truth=None, max_rounds=3, force_disagree=False):
    print("\n======================= Start New Question Debate =======================")
    print(f"PMID: {pmid}")
    print(f"Question: {question}")
    if ground_truth:
        print(f"Correct Answer: {ground_truth}")
    
    # Initialize debate
    debate_state = initialize_debate(pmid, question, context, force_disagree)
    
    # Save initial choices
    initial_choices = {
        "gpt": debate_state["gpt"]["choice"],
        "qwen": debate_state["qwen"]["choice"],
        "deepseek": debate_state["deepseek"]["choice"]
    }
    
    # Create history records for each model
    debate_history = {
        "gpt": [{"answer": debate_state["gpt"]["answer"], "choice": debate_state["gpt"]["choice"]}],
        "qwen": [{"answer": debate_state["qwen"]["answer"], "choice": debate_state["qwen"]["choice"]}],
        "deepseek": [{"answer": debate_state["deepseek"]["answer"], "choice": debate_state["deepseek"]["choice"]}]
    }
    
    # Check initial consensus
    has_consensus, consensus_choice = check_consensus({
        "gpt": debate_state["gpt"]["choice"],
        "qwen": debate_state["qwen"]["choice"],
        "deepseek": debate_state["deepseek"]["choice"]
    })
    
    if has_consensus:
        print(f"\nInitial consensus: All models chose {consensus_choice}")
        return consensus_choice, 0, initial_choices  # Return initial choices
    
    # Conduct debate rounds
    for round_num in range(1, max_rounds + 1):
        print(f"\n-------- Debate Round {round_num} --------")
        
        # GPT response
        print("\n----------------------- GPT Response -----------------------")
        gpt_response = gpt_responds_to_others(
            pmid, question, context,
            debate_state["qwen"]["answer"], debate_state["qwen"]["choice"],
            debate_state["deepseek"]["answer"], debate_state["deepseek"]["choice"],
            round_num,
            debate_history["gpt"]  # Pass GPT's history
        )
        print(gpt_response)
        gpt_choice = extract_model_choice(gpt_response)
        print(f"GPT New Choice: {gpt_choice}")
        debate_state["gpt"]["answer"] = gpt_response
        debate_state["gpt"]["choice"] = gpt_choice
        # Update GPT history
        debate_history["gpt"].append({"answer": gpt_response, "choice": gpt_choice})
        
        time.sleep(1)  # Prevent API rate limiting
        
        # Qwen response
        print("\n----------------------- Qwen Response -----------------------")
        qwen_response = qwen_responds_to_others(
            pmid, question, context,
            debate_state["gpt"]["answer"], debate_state["gpt"]["choice"],
            debate_state["deepseek"]["answer"], debate_state["deepseek"]["choice"],
            round_num,
            debate_history["qwen"]  # Pass Qwen's history
        )
        print(qwen_response)
        qwen_choice = extract_model_choice(qwen_response)
        print(f"Qwen New Choice: {qwen_choice}")
        debate_state["qwen"]["answer"] = qwen_response
        debate_state["qwen"]["choice"] = qwen_choice
        # Update Qwen history
        debate_history["qwen"].append({"answer": qwen_response, "choice": qwen_choice})
        
        time.sleep(1)  # Prevent API rate limiting
        
        # Deepseek response
        print("\n----------------------- Deepseek Response -----------------------")
        deepseek_response = deepseek_responds_to_others(
            pmid, question, context,
            debate_state["gpt"]["answer"], debate_state["gpt"]["choice"],
            debate_state["qwen"]["answer"], debate_state["qwen"]["choice"],
            round_num,
            debate_history["deepseek"]  # Pass Deepseek's history
        )
        print(deepseek_response)
        deepseek_choice = extract_model_choice(deepseek_response)
        print(f"Deepseek New Choice: {deepseek_choice}")
        debate_state["deepseek"]["answer"] = deepseek_response
        debate_state["deepseek"]["choice"] = deepseek_choice
        # Update Deepseek history
        debate_history["deepseek"].append({"answer": deepseek_response, "choice": deepseek_choice})
        
        # Check if consensus is reached
        has_consensus, consensus_choice = check_consensus({
            "gpt": debate_state["gpt"]["choice"],
            "qwen": debate_state["qwen"]["choice"],
            "deepseek": debate_state["deepseek"]["choice"]
        })
        
        if has_consensus:
            print(f"\nConsensus reached: All models chose {consensus_choice}")
            return consensus_choice, round_num, initial_choices
    
    # After maximum rounds without consensus, use majority voting
    print("\nMaximum rounds completed without full consensus. Using majority voting.")
    votes = {
        "yes": 0,
        "no": 0,
        "maybe": 0
    }
    
    votes[debate_state["gpt"]["choice"]] += 1
    votes[debate_state["qwen"]["choice"]] += 1
    votes[debate_state["deepseek"]["choice"]] += 1
    
    max_votes = max(votes.values())
    max_choices = [choice for choice, vote in votes.items() if vote == max_votes]
    
    if len(max_choices) == 1:
        final_choice = max_choices[0]
        print(f"Majority vote result: {final_choice}")
        return final_choice, max_rounds, initial_choices
    else:
        # If votes are tied, mark as invalid debate
        print(f"Tied votes, invalid debate! Vote distribution: {votes}")
        print("No consensus reached after maximum rounds and voting resulted in a tie.")
        return None, max_rounds, initial_choices




# Let GPT respond to other models' diagnoses: GPT --> QWEN, DeepSeek
def gpt_responds_to_others(pmid, question, context, qwen_answer, qwen_choice, deepseek_answer, deepseek_choice, debate_round, history=None):
    # Get history records
    self_previous_answer = ""
    self_previous_choice = ""
    
    if history and len(history) > 0:
        # Get previous round's answer and choice
        last_record = history[-1]
        self_previous_answer = last_record["answer"]
        self_previous_choice = last_record["choice"]
    
    prompt = f"""You are the GPT medical model engaged in an intense debate on a medical research question. Your task is to evaluate other models' perspectives while providing your own answer based on the scientific evidence in the abstract.

**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.** This will assess your ability to understand biomedical literature and maintain professional judgment during debate.

Question: {question}
Medical Literature Abstract: {context}

Debate Round: {debate_round}

【Your Previous Complete Analysis】
{self_previous_answer}

【Your Previous Choice】: {self_previous_choice}

Other Participants' Viewpoints:
Qwen's Answer: {qwen_answer}
Qwen's Choice: {qwen_choice}

Deepseek's Answer: {deepseek_answer}
Deepseek's Choice: {deepseek_choice}

Please critically examine the perspectives of other models while maintaining your professional independent judgment. If other models present compelling arguments that lead to a better conclusion, you should be willing to change your position.

【Debate Guidelines】
1. Position Statement: Please maintain your professional stance based on evidence analysis: don't be easily persuaded by consensus alone. Evaluate whether the arguments from other models provide better evidence interpretation or reveal flaws in your reasoning.
    Clearly state your position: Begin your response with one of the following formats:
   - "**I disagree with their viewpoints because:**" or
   - "**I agree with Qwen's perspective**" or
   - "**I agree with DeepSeek's perspective**" or
   - "**I agree with the shared perspective of Qwen and DeepSeek**" (when their views align)

2. Evaluation of Other Models' Diagnoses: Critically analyze and identify gaps, misinterpretations, or insufficient evidence in the arguments of other models. Specifically assess:
   - Do they correctly apply elimination reasoning?
   - Are their evidence interpretations more accurate than yours?
   - Do they provide new insights about the evidence you missed?

3. Medical Analysis and Argumentation:
    Provide your own independent medical analysis:
   - Re-examine the key evidence using the same systematic approach (pros/cons for each option)
   - Supplement important information not mentioned by other models based on the provided medical literature
   - Explain why your analysis might be more accurate or complete (if you disagree with other models' conclusions)
   - Apply elimination reasoning: Can you still rule out the options you previously ruled out?

4. Self-Questioning:
    **CRITICAL REMINDER: Base your decision on evidence analysis, not peer agreement.**
    
    If you consider changing your choice, you must answer:
   - Has the rationale for my original choice truly been completely refuted by new evidence interpretation?
   - Is the new choice better based on evidence, or just because others agree with it?
   - Does my original elimination reasoning still hold after hearing other perspectives?
   - Explain why you persist with or change your choice

5. Final Decision: 
   -[EXTREMELY IMPORTANT] Your final choice must use the exact format below, otherwise it will not be correctly recognized by the system:
    **My final choice is: "yes", "no", or "maybe"**

Please respond in the following format:
**1. Position Statement**
**2. Evaluation of Other Models**
**3. Medical Analysis and Argumentation**
**4. Self-Questioning**
**5. Final Decision**

This is round {debate_round} of the debate. Please maintain professional judgment.
"""

    try:
        print("GPT is generating response...")
        t_generate_start = time.time()
        
        data = {
            "model": "o1-mini",
            "messages": [
                {"role": "system", "content": "You are the GPT medical reasoning model, engaged in an intense debate on medical research questions with the Qwen model and DeepSeek model"},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 8000
        }

        print("Sending request to GPT API...")
        response_data = send_api_request(GPT_API_URL, GPT_HEADERS, data, "GPT")
        t_generate = time.time() - t_generate_start
        
        if response_data:
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"GPT response generation completed, time taken: {t_generate:.2f} seconds")
            return answer
        else:
            raise Exception("API returned empty, please check connection and API key")

    except Exception as e:
        print(f"Error generating GPT response: {str(e)}")
        traceback.print_exc()
        raise Exception(f"GPT API call failed: {str(e)}")


# Let Qwen respond to other models' diagnoses: Qwen --> GPT, DeepSeek
def qwen_responds_to_others(pmid, question, context, gpt_answer, gpt_choice, deepseek_answer, deepseek_choice, debate_round, history=None):
    # Get history records
    self_previous_answer = ""
    self_previous_choice = ""
    
    if history and len(history) > 0:
        # Get previous round's answer and choice
        last_record = history[-1]
        self_previous_answer = last_record["answer"]
        self_previous_choice = last_record["choice"]

    prompt = f"""You are the Qwen medical model engaged in an intense debate on a medical research question. Your task is to evaluate other models' perspectives while providing your own answer based on the scientific evidence in the abstract.

**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.** This will assess your ability to understand biomedical literature and maintain professional judgment during debate.

Question: {question}
Medical Literature Abstract: {context}

Debate Round: {debate_round}

【Your Previous Complete Analysis】
{self_previous_answer}

【Your Previous Choice】: {self_previous_choice}

Other Participants' Viewpoints:
GPT's Answer: {gpt_answer}
GPT's Choice: {gpt_choice}

Deepseek's Answer: {deepseek_answer}
Deepseek's Choice: {deepseek_choice}

**DEBATE REASONING REMINDER:**
Before evaluating other models, briefly re-confirm your evidence analysis using the same systematic approach from your initial response. Focus on whether other models provide superior evidence interpretation, not just different opinions.

Please critically examine the viewpoints of other models while maintaining your professional independent judgment.

【Debate Guidelines】
1. Position Statement: Please maintain your professional stance based on evidence analysis: don't be easily persuaded by consensus alone. Evaluate whether the arguments from other models provide better evidence interpretation or reveal flaws in your reasoning.
    Clearly state your position by beginning your answer with one of the following formats:
   - "**I disagree with their viewpoints because:**" or
   - "**I agree with GPT's viewpoint**" or
   - "**I agree with DeepSeek's viewpoint**" or
   - "**I agree with the common viewpoint of GPT and DeepSeek**" (when their viewpoints align)

2. Evaluation of Other Models' Diagnoses: Critically analyze and identify gaps, misinterpretations, or insufficient evidence in the arguments of other models. Specifically assess:
   - Do they correctly apply elimination reasoning?
   - Are their evidence interpretations more accurate than yours?
   - Do they provide new insights about the evidence you missed?

3. Medical Analysis and Argumentation:
    Provide your own independent medical analysis:
   - Re-examine the key evidence using the same systematic approach (pros/cons for each option)
   - Supplement important information not mentioned by other models based on the provided medical literature
   - Explain why your analysis might be more accurate or complete (if you disagree with other models' conclusions)
   - Apply elimination reasoning: Can you still rule out the options you previously ruled out?

4. Self-Questioning:
    **CRITICAL REMINDER: Base your decision on evidence analysis, not peer agreement.**
    
    If you consider changing your choice, you must answer:
   - Has the rationale for my original choice truly been completely refuted by new evidence interpretation?
   - Is the new choice better based on evidence, or just because others agree with it?
   - Does my original elimination reasoning still hold after hearing other perspectives?
   - Explain why you persist with or change your choice

5. Final Decision: 
   -[EXTREMELY IMPORTANT] Your final choice must use the exact format below, otherwise it will not be correctly recognized by the system:
    **My final choice is: "yes", "no", or "maybe"**

Please respond in the following format:
**1. Position Statement**
**2. Evaluation of Other Models**
**3. Medical Analysis and Argumentation**
**4. Self-Questioning**
**5. Final Decision**


This is round {debate_round} of the debate. Please maintain professional judgment.
"""

    try:
        print("Qwen is generating response...")
        t_generate_start = time.time()
        
        data = {
            "model": "Qwen/QwQ-32B",
            "messages": [
                {"role": "system", "content": "You are the Qwen medical reasoning model, engaged in an intense debate on medical research questions with the GPT model and DeepSeek model"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }

        print("Sending request to Qwen API...")
        response_data = send_api_request(QWEN_API_URL, QWEN_HEADERS, data, "Qwen")
        t_generate = time.time() - t_generate_start
        
        if response_data:
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"Qwen response generation completed, time taken: {t_generate:.2f} seconds")
            return answer
        else:
            raise Exception("API returned empty, please check connection and API key")

    except Exception as e:
        print(f"Error generating Qwen response: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Qwen API call failed: {str(e)}")


# Let DeepSeek respond to other models' diagnoses: DeepSeek --> GPT, Qwen
def deepseek_responds_to_others(pmid, question, context, gpt_answer, gpt_choice, qwen_answer, qwen_choice, debate_round, history=None):
    # Get history records
    self_previous_answer = ""
    self_previous_choice = ""
    
    if history and len(history) > 0:
        # Get previous round's answer and choice
        last_record = history[-1]
        self_previous_answer = last_record["answer"]
        self_previous_choice = last_record["choice"]

    prompt = f"""You are the DeepSeek medical model engaged in an intense debate on a medical research question. Your task is to evaluate other models' perspectives while providing your own answer based on the scientific evidence in the abstract.

**CRITICAL: Use your full reasoning capabilities! Think thoroughly and carefully to avoid all potential traps and nuanced interpretations.** This will assess your ability to understand biomedical literature and maintain professional judgment during debate.

Question: {question}
Medical Literature Abstract: {context}

Debate Round: {debate_round}

【Your Previous Complete Analysis】
{self_previous_answer}

【Your Previous Choice】: {self_previous_choice}

Other Participants' Viewpoints:
GPT's Answer: {gpt_answer}
GPT's Choice: {gpt_choice}

Qwen's Answer: {qwen_answer}
Qwen's Choice: {qwen_choice}

**DEBATE REASONING REMINDER:**
Before evaluating other models, briefly re-confirm your evidence analysis using the same systematic approach from your initial response. Focus on whether other models provide superior evidence interpretation, not just different opinions.

Please critically examine the viewpoints of other models while maintaining your professional independent judgment.

【Debate Guidelines】
1. Position Statement: Please maintain your professional stance: don't be easily persuaded. Evaluate whether the arguments from other models truly refute your diagnosis.
    Clearly state your position: Begin your response with one of the following formats:
   - "**I disagree with their viewpoints because:**" or
   - "**I agree with GPT's viewpoint**" or
   - "**I agree with Qwen's viewpoint**" or
   - "**I agree with the shared viewpoint of GPT and Qwen**" (when their viewpoints align)

2. Evaluation of Other Models' Diagnoses: Critically analyze and identify gaps, misinterpretations, or insufficient evidence in the arguments of other models.

3. Medical Analysis and Argumentation:
    Provide your own independent medical analysis:
   - Re-examine the key evidence using the same systematic approach (pros/cons for each option)
   - Supplement important information not mentioned by other models based on the provided medical literature
   - Explain why your analysis might be more accurate or complete (if you disagree with other models' conclusions)
   - Apply elimination reasoning: Can you still rule out the options you previously ruled out?

4. Self-Questioning:
    **CRITICAL REMINDER: Base your decision on evidence analysis, not peer agreement.**
    
    If you consider changing your choice, you must answer:
   - Has the rationale for my original choice truly been completely refuted by new evidence interpretation?
   - Is the new choice better based on evidence, or just because others agree with it?
   - Does my original elimination reasoning still hold after hearing other perspectives?
   - Explain why you persist with or change your choice

5. Final Decision: 
   -[EXTREMELY IMPORTANT] Your final choice must use the exact format below, otherwise it will not be correctly recognized by the system:
    **My final choice is: "yes", "no", or "maybe"**

Please respond in the following format:
**1. Position Statement**
**2. Evaluation of Other Models**
**3. Medical Analysis and Argumentation**
**4. Self-Questioning**
**5. Final Decision**


This is round {debate_round} of the debate. Please maintain professional judgment.
"""

    try:
        print("Deepseek is generating response...")
        t_generate_start = time.time()
        
        data = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": "You are the DeepSeek medical reasoning model, engaged in an intense debate on medical research questions with the GPT model and Qwen model"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000
        }
        
        print("Sending request to DeepSeek API...")
        response_data = send_api_request(DEEPSEEK_API_URL + "/v1/chat/completions", DEEPSEEK_HEADERS, data, "DeepSeek")
        
        # If main API call fails, try backup API
        if not response_data:
            print("Trying backup API...")
            backup_data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek medical reasoning model, engaged in an intense debate on medical research questions with the GPT model and Qwen model"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 8000
            }
            backup_url = "https://api.siliconflow.cn/v1/chat/completions"
            backup_headers = {
                "Authorization": f"Bearer {QWEN_API_KEY}",
                "Content-Type": "application/json"
            }
            response_data = send_api_request(backup_url, backup_headers, backup_data, "DeepSeek(Backup)")
        
        t_generate = time.time() - t_generate_start
        
        if response_data:
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"Deepseek response generation completed, time taken: {t_generate:.2f} seconds")
            return answer
        else:
            raise Exception("API returned empty, please check connection and API key")
        
    except Exception as e:
        print(f"Error generating Deepseek response: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Deepseek API call failed: {str(e)}")


# Evaluate predictions
def evaluate_predictions(predictions, ground_truth):
    if not predictions or not ground_truth:
        print("Error: Predictions or ground truth is empty")
        return
    
    correct = 0
    total = 0
    
    for pmid, pred in predictions.items():
        if pmid in ground_truth:
            total += 1
            if pred == ground_truth[pmid]:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

# Run debate for single question
def process_single_debate(test_data, ground_truth, pmid, max_rounds=3, force_disagree=False):
    if pmid not in test_data:
        print(f"Error: PMID {pmid} not in test set")
        return None
    
    question_data = test_data[pmid]
    
    # Extract question information - note uppercase field names  
    question = question_data.get("QUESTION", "")  
    context = " ".join(question_data.get("CONTEXTS", []))  
    
    print(f"Extracted question: {question}")  # Add debug output
    print(f"Extracted context length: {len(context)}")
    
    # Get correct answer (if available)
    correct_answer = ground_truth.get(pmid) if ground_truth else None
    
    try:
        # Conduct debate
        final_choice, rounds_needed, initial_choices = conduct_debate(
            pmid, question, context, 
            correct_answer, max_rounds, force_disagree
        )
        
        # Results
        result = {
            "pmid": pmid,
            "prediction": final_choice,
            "rounds": rounds_needed,
            "initial_choices": initial_choices  # Add initial choices
        }
        
        if correct_answer:
            result["ground_truth"] = correct_answer
            result["correct"] = final_choice == correct_answer
        
        return result
    except Exception as e:
        print(f"Error processing PMID {pmid} debate: {str(e)}")
        traceback.print_exc()
        raise



# Run entire test set
def run_pubmedqa_debates(test_data_path, ground_truth_path, output_path, 
                          start_idx=0, max_debates=None, max_rounds=3, force_disagree=False, 
                          specific_pmids=None):
    # Load data
    test_data = load_pubmedqa_data(test_data_path)
    ground_truth = load_pubmedqa_data(ground_truth_path)
    
    if not test_data or not ground_truth:
        return
    
    # No longer use global logging, each case has its own log file
    print(f"Note: Each case will generate a separate log file, format: {{PMID}}_log.txt")
    
    print(f"Starting PubMedQA debate evaluation, time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test set: {test_data_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Start index: {start_idx}")
    print(f"Max debates: {max_debates if max_debates else 'All'}")
    print(f"Max debate rounds: {max_rounds}")
    
    # Determine PMID list to process
    if specific_pmids:
        # If specific PMIDs are specified, only process these
        pmids = [pmid for pmid in specific_pmids if pmid in ground_truth]
        invalid_pmids = [pmid for pmid in specific_pmids if pmid not in ground_truth]
        if invalid_pmids:
            print(f"Warning: Following PMIDs do not exist in ground truth: {invalid_pmids}")
        print(f"Specified to process {len(pmids)} specific PMIDs")
    else:
        # Normal processing flow
        pmids = list(ground_truth.keys())[start_idx:]
        if max_debates and max_debates < len(pmids):
            pmids = pmids[:max_debates]
            print(f"Limited to process {max_debates} debates, starting from index {start_idx}")
    
    # Initialize results
    predictions = {}
    debate_results = []
    
    # If output file exists, try to load to continue previous run
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            print(f"Loaded existing prediction results, total {len(predictions)} items")
            # Filter already processed PMIDs
            pmids = [pmid for pmid in pmids if pmid not in predictions]
            print(f"Remaining {len(pmids)} PMIDs need to be processed")
        except:
            print("Unable to load existing prediction results, will start over")
    
    # Process each question
    for pmid in tqdm(pmids, desc="Processing questions"):
        # Create separate log file for current case
        case_logger = CaseLogOutput(pmid)
        original_stdout = sys.stdout
        sys.stdout = case_logger
        
        try:
            print(f"\n======================= Start Processing PMID: {pmid} =======================")
            
            # Get question data for saving
            question_data = test_data.get(pmid, {})
            
            result = process_single_debate(
                test_data, ground_truth, pmid, 
                max_rounds, force_disagree
            )
            
            if result:
                predictions[pmid] = result["prediction"]
                debate_results.append(result)
                
                # Save separate JSON file for each sample, using PMID as filename
                case_filename = f"{pmid}.json"  # e.g.: 12345678.json
                
                print(f"\nSaving results... PMID: {pmid} -> {case_filename}")
                try:
                    # Save detailed information for single case
                    case_data = {
                        "pmid": pmid,
                        "question": question_data.get("QUESTION", ""),
                        "contexts": question_data.get("CONTEXTS", []),
                        "ground_truth": ground_truth.get(pmid) if ground_truth else None,
                        "prediction": result["prediction"],
                        "rounds_needed": result["rounds"],
                        "initial_choices": result.get("initial_choices", {}),  # Add initial choices
                        "correct": result.get("correct", None),
                        "debate_process": result,  # Include complete debate process
                        "log_file": case_logger.get_filename()  # Add log file name
                    }
                    
                    with open(case_filename, 'w', encoding='utf-8') as f:
                        json.dump(case_data, f, indent=2, ensure_ascii=False)
                    print(f"✅ Individual case file saved: {case_filename}")
                    
                    # Also save summary results
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(predictions, f, indent=2)
                    print(f"✅ Summary results file saved: {output_path}")
                    
                    print(f"Current progress: {len(predictions)}/{len(pmids) + len(predictions)} samples completed")
                    
                except Exception as save_error:
                    print(f"❌ Error saving file: {save_error}")
                
                print(f"\n======================= Completed Processing PMID: {pmid} =======================")
                
            else:
                print(f"❌ PMID {pmid} processing failed, no valid results obtained")
                    
        except Exception as e:
            print(f"Error processing PMID {pmid}: {str(e)}")
            traceback.print_exc()
        finally:
            # Restore standard output and close case log
            sys.stdout = original_stdout
            case_logger.close()
            
            # Display progress on main console
            if len(debate_results) % 10 == 0 and len(debate_results) > 0:
                print(f"\n🎯 Milestone progress: {len(predictions)} samples completed and saved!")
            
            # Save current results (in case of error)
            if 'e' in locals():
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2)
    
    # Save final prediction results
    #with open(output_path, 'w', encoding='utf-8') as f:
        #json.dump(predictions, f, indent=2)
    
    # Save detailed debate results
    #detailed_output = output_path.replace('.json', '_detailed.json')
    #with open(detailed_output, 'w', encoding='utf-8') as f:
        #json.dump(debate_results, f, indent=2)
    
    # Evaluate results
    print("\n=== Final Evaluation Results ===")
    accuracy = evaluate_predictions(predictions, ground_truth)
    
    # Analyze debate statistics
    print("\n=== Debate Statistics Analysis ===")
    stats_report = analyze_debate_statistics(debate_results, ground_truth)
    print(stats_report)
    
    # All case logs have been closed in their respective loops
    
    print(f"Evaluation completed! Accuracy: {accuracy:.4f}")
    print(f"Prediction results saved to: {output_path}")
    print(f"Detailed logs for each case saved separately as: {{PMID}}_log.txt files")
    #print(f"Detailed debate results saved to: {detailed_output}")
    
    return accuracy, predictions

# Analyze debate statistics
def analyze_debate_statistics(detailed_results, ground_truth):
    total_cases = len(detailed_results)
    if total_cases == 0:
        return "Not enough cases for analysis"
    
    # Initialize statistics
    stats = {
        "consensus_count": 0,
        "majority_vote_count": 0,
        "model_stance_changes": {"gpt": 0, "qwen": 0, "deepseek": 0},
        "correct_to_wrong": {"gpt": 0, "qwen": 0, "deepseek": 0},
        "wrong_to_correct": {"gpt": 0, "qwen": 0, "deepseek": 0}
    }
    
    for result in detailed_results:
        pmid = result["pmid"]
        truth = ground_truth.get(pmid)
        if not truth:
            continue
            
        # Extract debate history
        debate_history = result.get("debate_history", {})
        
        # Check if consensus or majority vote
        if result.get("consensus", False):
            stats["consensus_count"] += 1
        else:
            stats["majority_vote_count"] += 1
        
        # Analyze stance changes for each model
        for model in ["gpt", "qwen", "deepseek"]:
            history = debate_history.get(model, [])
            if len(history) < 2:  # Need at least initial and final stance
                continue
                
            initial_choice = history[0]["choice"]
            final_choice = history[-1]["choice"]
            
            # Check if stance changed
            if initial_choice != final_choice:
                stats["model_stance_changes"][model] += 1
                
                # Check correctness changes
                initial_correct = (initial_choice == truth)
                final_correct = (final_choice == truth)
                
                if initial_correct and not final_correct:
                    stats["correct_to_wrong"][model] += 1
                elif not initial_correct and final_correct:
                    stats["wrong_to_correct"][model] += 1
    
    # Generate report
    report = "----------------------------------------\n"
    report += f"Cases with model consensus: {stats['consensus_count']} ({stats['consensus_count']/total_cases*100:.2f}%)\n"
    report += f"Cases requiring majority vote: {stats['majority_vote_count']} ({stats['majority_vote_count']/total_cases*100:.2f}%)\n"
    report += "----------------------------------------\n"
    
    for model in ["gpt", "qwen", "deepseek"]:
        model_name = model.capitalize()
        report += f"{model_name} stance change cases: {stats['model_stance_changes'][model]} ({stats['model_stance_changes'][model]/total_cases*100:.2f}%)\n"
    
    report += "----------------------------------------\n"
    for model in ["gpt", "qwen", "deepseek"]:
        model_name = model.capitalize()
        report += f"{model_name} correct to wrong cases: {stats['correct_to_wrong'][model]} ({stats['correct_to_wrong'][model]/total_cases*100:.2f}%)\n"
    
    report += "----------------------------------------\n"
    for model in ["gpt", "qwen", "deepseek"]:
        model_name = model.capitalize()
        report += f"{model_name} wrong to correct cases: {stats['wrong_to_correct'][model]} ({stats['wrong_to_correct'][model]/total_cases*100:.2f}%)\n"
    
    return report

# Main function
def main():
    try:
        parser = argparse.ArgumentParser(description='MCC Framework for PubMedQA Medical Question Answering')
        parser.add_argument('--test_data', type=str, default='../benchmarks/PubMedQA/test_set.json',
                            help='Test dataset path')
        parser.add_argument('--ground_truth', type=str, default='../benchmarks/PubMedQA/test_ground_truth.json',
                            help='Ground truth path')
        parser.add_argument('--output', type=str, default='debate_predictions.json',
                            help='Output prediction results path')
        parser.add_argument('-n', '--num_debates', type=int, default=None,
                            help='Maximum number of questions to process')
        parser.add_argument('-r', '--max_rounds', type=int, default=3,
                            help='Maximum rounds per debate')
        parser.add_argument('-f', '--force_disagree', action='store_true',
                            help='Force models to have different initial opinions')
        parser.add_argument('-s', '--start_idx', type=int, default=0,
                            help='Starting question index')
        parser.add_argument('--pmids', type=str, nargs='+', default=None,
                            help='Specify PMID list to process, e.g.: --pmids 12345678 87654321')
        parser.add_argument('--pmid_file', type=str, default=None,
                            help='File path containing PMID list, one PMID per line')
        parser.add_argument('--gpt_api_key', type=str, default=GPT_API_KEY,
                            help='GPT API key')
        parser.add_argument('--qwen_api_key', type=str, default=QWEN_API_KEY,
                            help='Qwen API key')
        parser.add_argument('--deepseek_api_key', type=str, default=DEEPSEEK_API_KEY,
                            help='DeepSeek API key')
        
        args = parser.parse_args()
        
        print(f"Note: Each case will generate a separate log file, format: {{PMID}}_log.txt")
        
        print(f"Starting PubMedQA three-model debate evaluation, time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPT API URL: {GPT_API_URL}")
        print(f"Qwen API URL: {QWEN_API_URL}")
        print(f"DeepSeek API URL: {DEEPSEEK_API_URL}")
        
        # Process PMID list parameters
        specific_pmids = None
        if args.pmids:
            specific_pmids = args.pmids
            print(f"Specified PMIDs to process: {specific_pmids}")
        elif args.pmid_file:
            # Read PMID list from file
            try:
                with open(args.pmid_file, 'r', encoding='utf-8') as f:
                    specific_pmids = [line.strip() for line in f if line.strip()]
                print(f"Read {len(specific_pmids)} PMIDs from file {args.pmid_file}")
            except Exception as e:
                print(f"Failed to read PMID file: {e}")
                return
        
        # Run debate evaluation
        accuracy, predictions = run_pubmedqa_debates(
            args.test_data,
            args.ground_truth,
            args.output,
            args.start_idx,
            args.num_debates,
            args.max_rounds,
            args.force_disagree,
            specific_pmids
        )
        
        print(f"Final accuracy: {accuracy:.4f}")
        
        # All case logs have been closed in their respective debates
    
    except Exception as e:
        print(f"Error during program execution: {str(e)}")
        traceback.print_exc()
        
        # Ensure all case logs are closed
        pass

if __name__ == "__main__":
    main()

# Usage Examples:
# python MCC_PubmedQA.py                                              # Process all questions
# python MCC_PubmedQA.py -n 10                                        # Process first 10 questions
# python MCC_PubmedQA.py -r 5                                         # Maximum 5 rounds per debate
# python MCC_PubmedQA.py --pmids 12345678 87654321 11111111           # Process specified PMIDs
# python MCC_PubmedQA.py --pmid_file selected_pmids.txt               # Read PMID list from file





