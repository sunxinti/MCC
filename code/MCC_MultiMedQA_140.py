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
        print("Configuration complete! Starting MultiMedQA model debate system...")
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
    """Class for simultaneously sending output to terminal and log file"""
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


# Load medical Q&A data
def load_medical_qa_data(file_path):
    """Load medical Q&A dataset
    
    Args:
        file_path: CSV file path, should contain question_text column
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        if not check_file_exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        print(f"Loading dataset: {file_path}")
        
        # Read CSV file
        dataset = pd.read_csv(file_path, encoding='latin1')
        
        # Validate required columns exist
        required_columns = ["question_text"]
        for col in required_columns:
            if col not in dataset.columns:
                raise ValueError(f"Dataset missing required column: {col}")
        
        print(f"Successfully loaded dataset with {len(dataset)} questions")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
        raise


# ===================== GPT Model Section =====================
def get_gpt_prompt(question):
    system_prompt = """You are a medical expert tasked with providing a comprehensive and accurate answer to a medical question. 
Please approach the question methodically, ensuring your answer is:

1. Accurate and based on current medical knowledge
2. Comprehensive, covering the necessary aspects of the question
3. Well-structured and easy to understand
4. Appropriately cautious when dealing with diagnostic or treatment information

IMPORTANT: Your response MUST be in English only.

Question:
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
            "max_completion_tokens": 16000,
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
            print("\nGPT error message:")
            print("-" * 80)
            print(error_message)
            print("-" * 80)
            return error_message

    except Exception as e:
        print(f"Error occurred while GPT generating answer: {str(e)}")
        traceback.print_exc()
        error_message = f"Sorry, an error occurred while processing your question. Error message: {str(e)}"
        print("\nGPT error message:")
        print("-" * 80)
        print(error_message)
        print("-" * 80)
        return error_message



# ===================== Qwen Model Section =====================
def get_qwen_prompt(question):
    system_prompt = """You are a medical expert tasked with providing a comprehensive and accurate answer to a medical question. 
Please approach the question methodically, ensuring your answer is:

1. Accurate and based on current medical knowledge
2. Comprehensive, covering the necessary aspects of the question
3. Well-structured and easy to understand
4. Appropriately cautious when dealing with diagnostic or treatment information

IMPORTANT: Your response MUST be in English only.

Question:
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
            "max_tokens": 16000
        }

        # Send request to Qwen API
        #print("Sending request to Qwen API...")
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
                    error_message = "API response structure abnormal: missing message field"
                    print("\nQwen error message:")
                    print("-" * 80)
                    print(error_message)
                    print("-" * 80)
                    return error_message
            else:
                print("Error: choices field does not exist or is empty")
                error_message = "API response structure abnormal: missing choices field"
                print("\nQwen error message:")
                print("-" * 80)
                print(error_message)
                print("-" * 80)
                return error_message
        else:
            print(f"Qwen API error: {response.status_code}")
            print(f"Error details: {response.text}")
            error_message = f"Sorry, an error occurred while processing your question. Error code: {response.status_code}"
            print("\nQwen error message:")
            print("-" * 80)
            print(error_message)
            print("-" * 80)
            return error_message

    except Exception as e:
        print(f"Error occurred while Qwen generating answer: {str(e)}")
        traceback.print_exc()
        error_message = f"Sorry, an error occurred while processing your question. Error message: {str(e)}"
        print("\nQwen error message:")
        print("-" * 80)
        print(error_message)
        print("-" * 80)
        return error_message


# ===================== DeepSeek Model Section =====================
def get_deepseek_prompt(question):
    system_prompt = """You are a medical expert tasked with providing a comprehensive and accurate answer to a medical question. 
Please approach the question methodically, ensuring your answer is:

1. Accurate and based on current medical knowledge
2. Comprehensive, covering the necessary aspects of the question
3. Well-structured and easy to understand
4. Appropriately cautious when dealing with diagnostic or treatment information

IMPORTANT: Your response MUST be in English only.

Question:
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
                max_tokens=16000
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
            
            # Backup plan: direct requests call, SiliconFlow API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1", # Note, use the same as the original study; you can replace it with other LLMs. 
                "messages": [
                    {"role": "system", "content": "You are DeepSeek-R1, a medical expert with extensive knowledge in healthcare and medicine."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 16000
            }
            url = "https://api.siliconflow.cn/v1/chat/completions" # SiliconFlow URL
            headers = {
                "Authorization": f"Bearer {QWEN_API_KEY}",
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
                print("\nDeepSeek error message:")
                print("-" * 80)
                print(error_message)
                print("-" * 80)
                return error_message
        
    except Exception as e:
        print(f"Error occurred while DeepSeek generating answer: {str(e)}")
        traceback.print_exc()
        error_message = f"Sorry, an error occurred while processing your question. Error message: {str(e)}"
        print("\nDeepSeek error message:")
        print("-" * 80)
        print(error_message)
        print("-" * 80)
        return error_message


# ===================== Consistency Evaluation Section =====================
def structured_model_evaluation(question, model_name, answer_to_evaluate, evaluator_model_name, evaluator_previous_answer):
    """Perform structured multi-dimensional evaluation of model answers
    
    Args:
        question: Original question
        model_name: Name of the model being evaluated
        answer_to_evaluate: Answer content to be evaluated
        evaluator_model_name: Name of the evaluator model
        evaluator_previous_answer: Evaluator's own answer
        
    Returns:
        Dict: Contains scores and evaluation content for three dimensions
    """
    try:       
        # Three core dimensions from the MCC study
        evaluation_dimensions = [
            {"name": "factual_accuracy", "description": "Evaluate the objective correctness of medical information. Check for incorrect medical information, inaccurate dosages/numbers, outdated medical practices, misused medical terminology, and ensure alignment with current medical consensus and guidelines. Focus on factual accuracy without considering completeness."},
            {"name": "completeness", "description": "Evaluate the comprehensiveness and communication effectiveness of the response. Check whether all necessary information points of the question are covered without key omissions; whether the language is clear, empathetic, well-structured, and easy to understand; whether it accurately understands the dialogue context and user intent, and appropriately seeks additional background information when needed."},
            {"name": "safety", "description": "Evaluate the safety of medical advice and compliance with guidelines. Check for harmful medical advice, assess whether urgency level is appropriate for the described situation, ensure recommendations are proportionate to the described symptoms, avoid unnecessary alarmism while maintaining appropriate caution; also evaluate whether the response strictly follows the user's specific instructions or role-playing requirements."}
        ]

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

IMPORTANT: Your response MUST be in English only.

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
        
        # Select appropriate API based on evaluator model
        if evaluator_model_name == "GPT":
            data = {
                "model": "o1-mini",
                "messages": [
                    {"role": "system", "content": "You are a medical evaluation expert. Provide concise, accurate evaluations focusing only on the most important aspects. Always respond in English."},
                    {"role": "user", "content": prompt}
                ],
                "max_completion_tokens": 8000,
            }
            response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                evaluation_text = response_data['choices'][0]['message']['content'].strip()
            else:
                print(f"Evaluation API error: {response.status_code}")
                return create_default_evaluation()
                
        elif evaluator_model_name == "Qwen":
            data = {
                "model": "Qwen/QwQ-32B", 
                "messages": [
                    {"role": "system", "content": "You are a medical evaluation expert. Provide concise, accurate evaluations focusing only on the most important aspects. Always respond in English."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 8000 
            }
            response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data, timeout=300)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    if 'message' in response_data['choices'][0]:
                        message = response_data['choices'][0]['message']
                        evaluation_text = message['content'].strip() if 'content' in message else "Evaluation failed"
                    else:
                        return create_default_evaluation()
                else:
                    return create_default_evaluation()
            else:
                print(f"Evaluation API error: {response.status_code}")
                return create_default_evaluation()
                
        elif evaluator_model_name == "DeepSeek":
            try:
                client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
                
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are a medical evaluation expert. Provide concise, accurate evaluations focusing only on the most important aspects. Always respond in English."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8000
                )
                
                evaluation_text = response.choices[0].message.content
                
            except Exception as e:
                print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
                # Try backup API
                data = {
                    "model": "Pro/deepseek-ai/DeepSeek-R1",
                    "messages": [
                        {"role": "system", "content": "You are a medical evaluation expert. Provide concise, accurate evaluations focusing only on the most important aspects. Always respond in English."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 3000
                }
                url = "https://api.siliconflow.cn/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {QWEN_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    evaluation_text = response_data['choices'][0]['message']['content'].strip()
                else:
                    print(f"Evaluation API error: {response.status_code}")
                    return create_default_evaluation()
        else:
            print(f"Unknown evaluator model: {evaluator_model_name}")
            return create_default_evaluation()
            
        # Print original evaluation text
        print(f"\n{evaluator_model_name}'s evaluation results for {model_name}:")
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
        
        print(f"{evaluator_model_name} has completed evaluation of {model_name}'s results")
        print(f"{'-'*80}\n")
        return evaluation_result
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()
        return create_default_evaluation()



def create_default_evaluation():
    """Create default evaluation result for error cases"""
    return {
        "factual_accuracy": {"score": 5, "reason": "Unable to evaluate", "suggestion": "No suggestions available"},
        "completeness": {"score": 5, "reason": "Unable to evaluate", "suggestion": "No suggestions available"},
        "safety": {"score": 5, "reason": "Unable to evaluate", "suggestion": "No suggestions available"},
        "comparative_analysis": "Unable to perform comparative analysis"
    }

def parse_evaluation_text(evaluation_text):
    """Parse evaluation text and extract scores and evaluations for each dimension"""
    try:
        evaluation_result = {}
        
        # Extract scores and evaluations for each dimension
        factual_accuracy_score = re.search(r'FACTUAL_ACCURACY_SCORE:\s*(\d+)', evaluation_text)
        factual_accuracy_reason = re.search(r'FACTUAL_ACCURACY_REASON:\s*(.*?)(?=FACTUAL_ACCURACY_SUGGESTION:|COMPLETENESS_SCORE:|$)', evaluation_text, re.DOTALL)
        factual_accuracy_suggestion = re.search(r'FACTUAL_ACCURACY_SUGGESTION:\s*(.*?)(?=COMPLETENESS_SCORE:|$)', evaluation_text, re.DOTALL)
        
        completeness_score = re.search(r'COMPLETENESS_SCORE:\s*(\d+)', evaluation_text)
        completeness_reason = re.search(r'COMPLETENESS_REASON:\s*(.*?)(?=COMPLETENESS_SUGGESTION:|SAFETY_SCORE:|$)', evaluation_text, re.DOTALL)
        completeness_suggestion = re.search(r'COMPLETENESS_SUGGESTION:\s*(.*?)(?=SAFETY_SCORE:|$)', evaluation_text, re.DOTALL)
        
        safety_score = re.search(r'SAFETY_SCORE:\s*(\d+)', evaluation_text)
        safety_reason = re.search(r'SAFETY_REASON:\s*(.*?)(?=SAFETY_SUGGESTION:|COMPARATIVE_ANALYSIS:|$)', evaluation_text, re.DOTALL)
        safety_suggestion = re.search(r'SAFETY_SUGGESTION:\s*(.*?)(?=COMPARATIVE_ANALYSIS:|$)', evaluation_text, re.DOTALL)
        
        comparative_analysis = re.search(r'COMPARATIVE_ANALYSIS:\s*(.*?)($)', evaluation_text, re.DOTALL)
        
        # Build evaluation result dictionary
        evaluation_result["factual_accuracy"] = {
            "score": int(factual_accuracy_score.group(1)) if factual_accuracy_score else 5,
            "reason": factual_accuracy_reason.group(1).strip() if factual_accuracy_reason else "No reason provided",
            "suggestion": factual_accuracy_suggestion.group(1).strip() if factual_accuracy_suggestion else "No suggestion provided"
        }
        
        evaluation_result["completeness"] = {
            "score": int(completeness_score.group(1)) if completeness_score else 5,
            "reason": completeness_reason.group(1).strip() if completeness_reason else "No reason provided",
            "suggestion": completeness_suggestion.group(1).strip() if completeness_suggestion else "No suggestion provided"
        }
        
        evaluation_result["safety"] = {
            "score": int(safety_score.group(1)) if safety_score else 5,
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
        return create_default_evaluation()



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
               
        # Build improvement prompt
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

IMPORTANT: Your response MUST be in English only.

First provide a critique summary, then provide your complete improved answer. Use this format:

CRITIQUE_SUMMARY: [Brief summary of evaluation feedback, including points you agree and disagree with]

IMPROVED_ANSWER: [Your complete improved answer]
"""
        
        # Select appropriate API based on model
        if model_name == "GPT":
            data = {
                "model": "o1-mini",
                "messages": [
                    {"role": "system", "content": "You are GPT-o1, a medical expert who is improving your answer based on feedback."},
                    {"role": "user", "content": prompt}
                ],
                "max_completion_tokens": 16000,
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
                "max_tokens": 16000
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
                    max_tokens=16000
                )
                
                improvement_text = response.choices[0].message.content
                
            except Exception as e:
                print(f"Failed to call DeepSeek API using OpenAI client: {str(e)}")
                # Try backup API
                data = {
                    "model": "Pro/deepseek-ai/DeepSeek-R1",
                    "messages": [
                        {"role": "system", "content": "You are DeepSeek-R1, a medical expert who is improving your answer based on feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 16000
                }
                url = "https://api.siliconflow.cn/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {QWEN_API_KEY}",
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
        print(f"\n{model_name}'s optimized output:")
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
    
    # Get the latest round of evaluations
    latest_round = evaluations_history[-1]
    
    # Calculate score differences for each dimension
    dimension_agreement = {}
    dimensions = ["factual_accuracy", "completeness", "safety"] 
    
    for dimension in dimensions:
        scores = []
        for model, evals in latest_round.items():
            for target, eval_data in evals.items():
                if dimension in eval_data and "score" in eval_data[dimension]:
                    scores.append(eval_data[dimension]["score"])
        
        # Calculate score standard deviation as consistency indicator
        if scores:
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            std_dev = variance ** 0.5
            
            # Smaller standard deviation means higher consistency
            agreement_score = max(0, 1 - (std_dev / 4)) 
            dimension_agreement[dimension] = agreement_score
    
    # Calculate overall consensus score
    if dimension_agreement:
        consensus_score = sum(dimension_agreement.values()) / len(dimension_agreement)
    else:
        consensus_score = 0.0
    
    # Determine if further debate is needed 
    requires_further_debate = consensus_score < 0.8 # Consistent with other task standards
    
    return {
        "consensus_score": consensus_score,
        "requires_further_debate": requires_further_debate,
        "dimension_agreement": dimension_agreement
    }



# ===================== Model Debate Section =====================
def circular_criticism_improvement(question, initial_answers, max_rounds=3):
    """Conduct model debate using circular criticism-improvement mode
    
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
        print(f"\nRound {round_num} - Phase 1: Model confrontation phase initiated")
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
                    print(f"{evaluator}'s evaluation of {target} completed")
                except Exception as e:
                    print(f"Error in {evaluator}'s evaluation of {target}: {str(e)}")
                    round_evaluations[evaluator][target] = create_default_evaluation()
        
        # Store current round evaluations
        evaluations_history.append(round_evaluations)
        
        # Calculate current consensus metrics
        consensus_metrics = calculate_consensus_metrics(evaluations_history)
        print(f"\nCurrent consensus status:")
        print(f"- Overall consensus score: {consensus_metrics['consensus_score']:.2f}")
        print("- Dimension consensus scores:")
        for dim, score in consensus_metrics.get("dimension_agreement", {}).items():
            print(f"  * {dim}: {score:.2f}")
        
        # If sufficient consensus is reached, debate can end early
        if not consensus_metrics["requires_further_debate"] and round_num > 1:
            print("\nModels have reached sufficient consensus, ending debate early")
            break
        
        # Step 2: Each model improves its own answer based on evaluation results (parallel)
        print(f"\nRound {round_num} - Phase 2: Models improve their own answers based on evaluation results")
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
                    print(f"{model}'s answer improvement completed")
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
    
    # Final consistency evaluation
    final_consensus_metrics = calculate_consensus_metrics(evaluations_history)
    print("\nDebate finished, final consensus status:")
    print(f"- Overall consensus score: {final_consensus_metrics['consensus_score']:.2f}")
    print("- Dimension consensus scores:")
    for dim, score in final_consensus_metrics.get("dimension_agreement", {}).items():
        print(f"  * {dim}: {score:.2f}")
    
    # Use expert integration model to generate final answer
    print("\nGenerating final integrated answer...")
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
        print("\nGenerating final integrated answer...")
        
        # Extract key information from debate history
        debate_summary = ""
        for round_data in debate_history:
            round_num = round_data["round"]
            if round_num > 0:  # Skip initial answers
                debate_summary += f"Round {round_num} summary:\n"
                
                # Add improvement and evaluation information for each model
                if "improvements" in round_data:
                    for model, improvement in round_data["improvements"].items():
                        if "critique" in improvement:
                            debate_summary += f"{model} critique summary: {improvement['critique'][:200]}...\n"
                
                debate_summary += "\n"
        
        # Print debate summary
        print("\nDebate history summary:")
        print(f"{'-'*40}")
        print(debate_summary)
        print(f"{'-'*40}")
        
        # Build dimension score information
        dimension_scores = ""
        if "dimension_agreement" in consensus_metrics:
            dimension_scores += "Dimension consensus scores:\n"
            for dimension, score in consensus_metrics["dimension_agreement"].items():
                dimension_scores += f"- {dimension}: {score:.2f}\n"
        
        prompt = f"""As a medical expert integrator, you need to create a final consolidated answer based on the responses and debate process of three AI models.

Original question:
{question}

GPT final answer:
{final_answers["GPT"]}

Qwen final answer:
{final_answers["Qwen"]}

DeepSeek final answer:
{final_answers["DeepSeek"]}

Debate process summary:
{debate_summary}

Consensus evaluation:
- Overall consensus score: {consensus_metrics["consensus_score"]:.2f}
{dimension_scores}

Please create a final integrated answer that:
1. Create a comprehensive and accurate answer by integrating information from all three models. Leverage each model's strengths while strictly limiting content to what was mentioned by at least one model - DO NOT add any new medical information, facts, or insights not present in the model answers.
2. Labels key points with contributing models (e.g., [GPT, Qwen, DeepSeek])
3. Has clear structure with focus on clinical relevance
4. Handles medical advice with appropriate caution

IMPORTANT: Your response MUST be in English only.
"""
        
        data = {
            "model": "o1-mini",
            "messages": [
                {"role": "system", "content": "You are a medical integration specialist responsible for carefully combining information from multiple AI models' answers. Your task is to organize and synthesize existing information without adding new medical facts."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 16000,
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


# ===================== Main Function and Runtime Logic =====================
def process_single_question(question, reference_answer=None, max_rounds=3, force_debate=False):
    """Process a single medical question
    
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
        
        print("Starting to obtain initial answers from three models in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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
        
        # Build results
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
        
        # Build error results
        return {
            "question": question,
            "error": str(e),
            "error_traceback": traceback.format_exc()
        }

# Compatibility functions
def gpt_responds_to_others(question, qwen_answer, deepseek_answer, debate_round, self_previous_answer=None):
    """Compatibility function for legacy code calls, will be deprecated in new version
    
    This function is no longer used in the current circular criticism-improvement mode, kept only for compatibility
    """
    print("Warning: gpt_responds_to_others function is deprecated, please use the new circular criticism-improvement mode")
    return {
        "critique": "This function is deprecated",
        "improved_answer": self_previous_answer or "GPT answer"
    }

def qwen_responds_to_others(question, gpt_answer, deepseek_answer, debate_round, self_previous_answer=None):
    """Compatibility function for legacy code calls, will be deprecated in new version
    
    This function is no longer used in the current circular criticism-improvement mode, kept only for compatibility
    """
    print("Warning: qwen_responds_to_others function is deprecated, please use the new circular criticism-improvement mode")
    return {
        "critique": "This function is deprecated",
        "improved_answer": self_previous_answer or "Qwen answer"
    }

def deepseek_responds_to_others(question, gpt_answer, qwen_answer, debate_round, self_previous_answer=None):
    """Compatibility function for legacy code calls, will be deprecated in new version
    
    This function is no longer used in the current circular criticism-improvement mode, kept only for compatibility
    """
    print("Warning: deepseek_responds_to_others function is deprecated, please use the new circular criticism-improvement mode")
    return {
        "critique": "This function is deprecated",
        "improved_answer": self_previous_answer or "DeepSeek answer"
    }


# Main function
def main():
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        sys.stdout = TeeOutput(log_file)
        
        parser = argparse.ArgumentParser(description='MCC Medical Q&A Framework')
        parser.add_argument('-d', '--dataset', type=str, default="../benchmarks/MultiMedQA_140/MultiMedQA140.csv", help='Dataset file path (default: MultiMedQA140.csv)')
        parser.add_argument('-n', '--num_questions', type=int, default=1, help='Number of questions to process (default: 1)')
        parser.add_argument('-s', '--start_idx', type=int, default=0, help='Starting question index (default: 0)')
        parser.add_argument('-r', '--max_rounds', type=int, default=3, help='Maximum debate rounds (default: 3)')
        parser.add_argument('-f', '--force_debate', action='store_true', help='Force debate even if initial answers are consistent')
        parser.add_argument('-q', '--question', type=str, help='Directly specify the question to process, if provided, dataset will not be used')
        args = parser.parse_args()
        
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        if args.question:
            print("\n" + "="*50)
            print(f"Processing specified question: {args.question}")
            print("="*50)
            
            # Process single question
            result = process_single_question(
                args.question,
                reference_answer=None,
                max_rounds=args.max_rounds,
                force_debate=args.force_debate
            )
            
            # Save results to JSON file
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = os.path.join(results_dir, f"result_direct_question_{timestamp}.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\nProcessing results saved to {result_file}")
            print(f"Log file saved to {log_file}")
            
        # Process questions from dataset
        else:
            # Check if dataset exists
            if not check_file_exists(args.dataset):
                print("Program terminated, dataset does not exist.")
                exit(1)
            
            # Load dataset
            dataset = load_medical_qa_data(args.dataset)
            
            # Create summary results list
            summary_results = []
            
            # Process multiple questions
            for i in range(args.start_idx, min(args.start_idx + args.num_questions, len(dataset))):
                print("\n" + "="*50)
                print(f"Processing question {i+1}/{min(args.start_idx + args.num_questions, len(dataset))} (index: {i})")
                print("="*50)
                
                # Get question
                question = dataset.iloc[i]["question_text"]
                
                # Process single question
                result = process_single_question(
                    question,
                    reference_answer=None,
                    max_rounds=args.max_rounds,
                    force_debate=args.force_debate
                )
                
                # Save individual question results to JSON file
                question_result_file = os.path.join(results_dir, f"result_question_{i}.json")
                with open(question_result_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # Add to summary results
                summary_result = {
                    "question_idx": i,
                    "question": question,
                    "consensus_score": result.get("consensus_score", 0.0)
                }
                summary_results.append(summary_result)
                
            # Save summary results
            summary_file = os.path.join(results_dir, "summary_results.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_results, f, ensure_ascii=False, indent=2)
            
            # Output summary statistics
            if summary_results:
                total_questions = len(summary_results)
                consensus_questions = sum(1 for r in summary_results if r.get("consensus_score", 0.0) > 0.0)
                
                print("\n" + "="*50)
                print("Processing Results Statistics")
                print("="*50)
                print(f"Total questions processed: {total_questions}")
                print(f"Questions reaching consensus: {consensus_questions} ({consensus_questions/total_questions:.2%})")
                print("-" * 40)
                print(f"Average consensus score: {sum(r.get('consensus_score', 0.0) for r in summary_results) / total_questions:.2f}")
                print("-" * 40)
                print(f"\nSummary results saved to {summary_file}")
                print(f"Log file saved to {log_file}")
        
        # Close log file
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            # Restore standard output
            sys.stdout = sys.__stdout__
    
    except Exception as e:
        print(f"Error occurred during program execution: {str(e)}")
        traceback.print_exc()
        
        # Ensure log file is closed
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main() 