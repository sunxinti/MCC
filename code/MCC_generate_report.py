#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Core Reports from Original MCC Debate Logs

Features:
1. Support both MCQ (Multiple Choice Questions) and LFQ (Long Form Questions) formats
2. Use GPT-4.1 to generate structured core reports in required format
3. Ensure faithful information compression without adding new content

Usage:
  python generate_MCC_report.py --type MCQ log_file.txt
  python generate_MCC_report.py --type LFQ log_file.txt
  python generate_MCC_report.py --type MCQ log_file.txt -o output.txt
"""

import requests
import os
import sys
import argparse
import re
import configparser

# ===== API Configuration =====
CONFIG_FILE = "api_config.ini"

def setup_api_config():
    """Setup API configuration from file or user input"""
    config = configparser.ConfigParser()
    
    if os.path.exists(CONFIG_FILE):
        print("Found API configuration file, loading...")
        config.read(CONFIG_FILE, encoding='utf-8')
        
        # Check if GPT configuration is complete
        if config.has_section('GPT') and \
           config.has_option('GPT', 'api_key') and \
           config.has_option('GPT', 'api_url') and \
           config.get('GPT', 'api_key').strip() and \
           config.get('GPT', 'api_url').strip():
            print("API configuration loaded successfully!")
            return config
        else:
            print("Configuration file incomplete, reconfiguration needed...")
    
    print("="*60)
    print("Welcome to MCC Report Generator!")
    print("First run requires API key configuration, please enter your GPT API information.")
    print("Configuration will be saved locally, no need to re-enter for subsequent runs.")
    print("="*60)
    
    # GPT API configuration
    print("\n【GPT API Configuration】")
    print("Please enter your GPT API configuration information:")
    gpt_api_url = input("GPT API URL (Press ENTER for default: https://api.chatanywhere.tech/v1/chat/completions): ").strip()
    if not gpt_api_url:
        gpt_api_url = "https://api.chatanywhere.tech/v1/chat/completions"
    gpt_api_key = input("GPT API Key: ").strip()
    
    if not gpt_api_key:
        print("\nError: API Key cannot be empty! Please restart the program and enter complete API configuration.")
        sys.exit(1)
    
    # Store configuration
    config['GPT'] = {
        'api_key': gpt_api_key,
        'api_url': gpt_api_url
    }
    
    # Write configuration file
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            config.write(f)
        print(f"\n✓ API configuration saved to {CONFIG_FILE}")
        print("Configuration complete! Starting report generation...")
    except Exception as e:
        print(f"\nWarning: Failed to save configuration file: {e}")
        print("Program will continue running, but configuration will need to be re-entered on next startup.")
    
    return config

def get_api_config():
    """Get GPT API configuration"""
    config = setup_api_config()
    
    gpt_config = {
        'api_key': config.get('GPT', 'api_key'),
        'api_url': config.get('GPT', 'api_url'),
        'headers': {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.get('GPT', 'api_key')}"
        }
    }
    
    return gpt_config

# Initialize API configuration
GPT_CONFIG = get_api_config()
GPT_API_KEY = GPT_CONFIG['api_key']
GPT_API_URL = GPT_CONFIG['api_url']
GPT_HEADERS = GPT_CONFIG['headers']

def get_mcq_prompt(log_content: str) -> str:
    """
    Get prompt for MCQ (Multiple Choice Questions) format
    
    Args:
        log_content: Original debate log content
        
    Returns:
        str: Formatted prompt for MCQ
    """
    prompt = f"""
Please generate a concise core report based on the following MCC (Model Confrontation and Collaboration) original logs to facilitate quick review and understanding by clinicians.

Requirements:
1. Strictly organize report content according to the following format:
   - Case Description: [Extract original case description]
   - Options: [List all options, maintain original numbering]
   - Correct Answer: [Correct answer option and content]
   
   - Initial Responses from Three Models:
     * GPT chose XX, Main reasoning: [Core viewpoints and key arguments]
     * Qwen chose XX, Main reasoning: [Core viewpoints and key arguments]  
     * DeepSeek chose XX, Main reasoning: [Core viewpoints and key arguments]

   - Consensus Status: [Consensus reached/No consensus reached]
   
   [If no consensus reached, display debate process by actual rounds in the following format:]
   - Round X Debate:
     * GPT chose XX, Main reasoning: [Viewpoint summary and rebuttal basis]
     * Qwen chose XX, Main reasoning: [Viewpoint summary and rebuttal basis]  
     * DeepSeek chose XX, Main reasoning: [Viewpoint summary and rebuttal basis]
     * Consensus Status: [Consensus reached/No consensus reached]
   
   [Repeat above format until debate ends or consensus is reached]

   - Voting Results (if no consensus reached):
     * Final choices of each model: GPT chose XX, Qwen chose XX, DeepSeek chose XX
     * Vote statistics: Option X received X votes, Option Y received Y votes
     * Majority choice: [Majority choice determined by voting results]
   
   - Final Results:
     * Consensus Status: [Consensus reached/No consensus reached, decided by voting]
     * Final Choice: [Consensus choice or majority voting choice]
     * Correctness Analysis: [Comparison and analysis with standard answer]

2. Language Requirements:
   - Use English
   - Maintain original professional terminology and key medical expressions
   - Be concise and clear, highlighting core viewpoint differences
   - Suitable for quick reading by clinicians

3. Core Principles:
   - Fidelity: Only perform structural organization and length compression of original debate content, absolutely do not add any information, judgments, or suggestions not present in the original logs
   - Objectivity: Do not make any additional medical judgments, interpretations, or reasoning, strictly maintain original reasoning logic and expressions
   - Completeness: Ensure all key diagnostic evidence, controversial points, and medical evidence are retained without omitting important information

4. Content Requirements:
   - Strictly based on original log content, serving as a pure information compression tool
   - Retain important medical terminology, diagnostic criteria, and differential points
   - Highlight key differences and controversial points among model viewpoints
   - Extract the most convincing diagnostic evidence

5. Special Case Handling:
   - If consensus is reached early, only show rounds up to consensus achievement, skip voting process
   - If any model changes choice, clearly mark the choice change
   - If no final consensus is reached, determine final choice through majority voting
   - If any model does not participate in a round, mark "Did not participate in this round"
   - Maintain medical professionalism, avoid colloquial expressions

Here is the complete debate log:

{log_content}

Please strictly generate the core report according to the above format:
"""
    return prompt

def get_lfq_prompt(log_content: str) -> str:
    """
    Get prompt for LFQ (Long Form Questions) format
    
    Args:
        log_content: Original debate log content
        
    Returns:
        str: Formatted prompt for LFQ
    """
    prompt = f"""
Please generate a concise core report based on the following MCC (Model Confrontation and Collaboration) original logs to facilitate quick review and understanding by clinicians.

Requirements:
1. Strictly organize report content according to the following format:
   - Patient Question: [Extract original patient question/query]
   - Initial Responses from Three Models:
     * GPT Response: [Core viewpoints and key medical advice]
     * Qwen Response: [Core viewpoints and key medical advice]  
     * DeepSeek Response: [Core viewpoints and key medical advice]

   - Consensus Status: [Consensus reached/No consensus reached]
   
   [If no consensus reached, display debate process by actual rounds in the following format:]
   - Round X Debate:
   
     Model Cross-Evaluations:
     * GPT evaluated DeepSeek: [Factual accuracy score, completeness score, safety score, key critique points]
     * GPT evaluated Qwen: [Factual accuracy score, completeness score, safety score, key critique points]
     * DeepSeek evaluated GPT: [Factual accuracy score, completeness score, safety score, key critique points]
     * DeepSeek evaluated Qwen: [Factual accuracy score, completeness score, safety score, key critique points]
     * Qwen evaluated GPT: [Factual accuracy score, completeness score, safety score, key critique points]
     * Qwen evaluated DeepSeek: [Factual accuracy score, completeness score, safety score, key critique points]
     
     Model Improvements:
     * GPT Improvement: 
       - Critique Summary: [Summary of feedback received from other models]
       - Specific Changes: [Diagnoses added/removed, new recommendations, modified urgency levels]
     * Qwen Improvement:
       - Critique Summary: [Summary of feedback received from other models]
       - Specific Changes: [Diagnoses added/removed, new recommendations, modified urgency levels]
     * DeepSeek Improvement:
       - Critique Summary: [Summary of feedback received from other models]
       - Specific Changes: [Diagnoses added/removed, new recommendations, modified urgency levels]
     
     * Consensus Status: [Consensus reached/No consensus reached]
   
   [Repeat above format until debate ends or consensus is reached]
   
   - Final Results:
     * Consensus Status: [Consensus reached/No consensus reached]
     * Final Integrated Answer: [CRITICAL: Extract and preserve the complete "Final reply" section from the logs WITHOUT any compression or summarization - this should be the full medical advice as originally written]
     
   - HealthBench Evaluation Results:
     * Individual Model Scores:
       - GPT Initial Response: [Extract HealthBench score percentage and explanation for why points were deducted]
       - Qwen Initial Response: [Extract HealthBench score percentage and explanation for why points were deducted]  
       - DeepSeek Initial Response: [Extract HealthBench score percentage and explanation for why points were deducted]
     * MCC Final Response: [Extract final MCC HealthBench score and evaluation summary]
     * Key Evaluation Criteria: [Extract the rubric criteria that were used to judge responses - look for criteria about emergency referral, relevance, etc.]

2. Language Requirements:
   - Use English
   - Maintain original professional terminology and key medical expressions
   - Be concise and clear, highlighting core viewpoint differences
   - Suitable for quick reading by clinicians

3. Core Principles:
   - Fidelity: Only perform structural organization and length compression of original debate content, absolutely do not add any information, judgments, or suggestions not present in the original logs
   - Objectivity: Do not make any additional medical judgments, interpretations, or reasoning, strictly maintain original reasoning logic and expressions
   - Completeness: Ensure all key diagnostic evidence, controversial points, and medical evidence are retained without omitting important information

4. Content Requirements:
   - Strictly based on original log content, serving as a pure information compression tool
   - Retain important medical terminology, diagnostic criteria, and differential points
   - Highlight key differences and controversial points among model viewpoints
   - Extract the most convincing diagnostic evidence
   - For each debate round, capture both the cross-evaluation scores/critiques AND the specific medical improvements made by each model
   - Show the logical flow: initial responses → cross-evaluations → improvements → next round
   - Include numerical scores (e.g., "factual_accuracy: 9.0/10") from the evaluation logs

5. Special Case Handling:
   - If consensus is reached early, only show rounds up to consensus achievement
   - If any model significantly changes their medical advice, clearly mark the change with phrases like "Added [diagnosis]", "Removed [diagnosis]", "Changed priority from X to Y"
   - Focus on the evolution of medical reasoning and safety considerations
   - If any model does not participate in a round, mark "Did not participate in this round"
   - Maintain medical professionalism, avoid colloquial expressions
   - Pay special attention to:
     * Models' cross-evaluation scores and reasoning (look for "FACTUAL_ACCURACY_SCORE", "COMPLETENESS_SCORE", "SAFETY_SCORE" sections)
     * Models' "CRITIQUE_SUMMARY" and "IMPROVED_ANSWER" sections - these contain the actual refined medical advice
     * The logical sequence: evaluation → critique summary → specific improvements
     * The "Final reply" section (usually starts with "====================================================================================================") - preserve this completely without compression as it represents the final MCC integrated medical advice
     * HealthBench evaluation sections (look for "GPT initial response HealthBench score", "MCC final response HealthBench score", "criteria_met", "explanation") - capture the rubric criteria, individual model scores, reasons for point deductions, and final MCC score

Here is the complete debate log:

{log_content}

Please strictly generate the core report according to the above format:
"""
    return prompt

def generate_core_report_from_log(log_content: str, case_id: str, report_type: str) -> str:
    """
    Generate core report based on original debate logs
    
    Args:
        log_content: Original debate log content
        case_id: Case ID
        report_type: Type of report ('MCQ' or 'LFQ')
        
    Returns:
        str: Generated core report
    """
    
    # Select appropriate prompt based on report type
    if report_type.upper() == 'MCQ':
        prompt = get_mcq_prompt(log_content)
    elif report_type.upper() == 'LFQ':
        prompt = get_lfq_prompt(log_content)
    else:
        return f"Error: Invalid report type '{report_type}'. Must be 'MCQ' or 'LFQ'."
    
    try:
        # Prepare API request
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an experienced clinician and medical education expert, skilled at extracting key information from complex medical discussions to generate concise and practical clinical reports. Please strictly summarize according to user requirements."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 10000,
            "temperature": 0.1
        }
        
        # Send request
        print(f"Generating {report_type} core report for case {case_id}...")
        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                report = result['choices'][0]['message']['content']
                return report
            else:
                return f"Report generation failed: API response format error"
        else:
            return f"Report generation failed: HTTP {response.status_code}, {response.text}"
            
    except Exception as e:
        return f"Error occurred during report generation: {str(e)}"

def read_log_file(file_path: str) -> str:
    """
    Read log content from txt file
    
    Args:
        file_path: Log file path
        
    Returns:
        str: Log file content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        raise Exception(f"Failed to read file: {str(e)}")

def extract_case_id(log_content: str) -> str:
    """
    Extract case ID from log content
    
    Args:
        log_content: Log content   
        
    Returns:
        str: Case ID
    """  
    # Try to extract case ID from different formats
    # Note: Patterns include both English and Chinese keywords to support different log formats
    patterns = [
        r'Case_id\s*:\s*([^\s]+)',                          # MCQ format: Case_id: case_961
        r'HealthBench Sample Log\s*[–-]\s*([^\s\n]+)',      # LFQ format: HealthBench Sample Log – consensus_sample_1155
        r'consensus_sample_(\d+)',                           # Extract from consensus_sample_XXXX
        r'案例ID\s*:\s*([^\s]+)',                            # Chinese MCQ format (if exists)
        r'HealthBench样本日志\s*-\s*([^\s\n]+)',            # Chinese LFQ format (if exists)
        r'样本日志.*?([a-zA-Z0-9_]+)'                        # Generic Chinese format (if exists)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, log_content)
        if match:
            return match.group(1)
    
    return "unknown_case"

def main():
    """
    Main function: Process command line arguments and generate report
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Generate core reports from MCC debate logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_MCC_report.py --type MCQ case_961.txt
  python generate_MCC_report.py --type LFQ consensus_sample_1155.txt
  python generate_MCC_report.py --type MCQ case_961.txt -o my_report.txt
        """
    )
    parser.add_argument('--type', required=True, choices=['MCQ', 'LFQ', 'mcq', 'lfq'],
                        help='Type of report to generate: MCQ (Multiple Choice Questions) or LFQ (Long Form Questions)')
    parser.add_argument('log_file', help='Input log txt file path')
    parser.add_argument('-o', '--output', help='Output report filename (optional)', default=None)
    
    args = parser.parse_args()
    
    # Normalize report type to uppercase
    report_type = args.type.upper()
    
    # Check if input file exists
    if not os.path.exists(args.log_file):
        print(f"Error: File {args.log_file} does not exist")
        return
    
    print(f"Starting to process file: {args.log_file}")
    print(f"Report type: {report_type}")
    
    # Read log file
    try:
        log_content = read_log_file(args.log_file)
        case_id = extract_case_id(log_content)
        print(f"Detected case ID: {case_id}")
    except Exception as e:
        print(f"Failed to read log file: {e}")
        return
    
    # Generate core report
    print("Successfully read log file, starting to generate core report...")
    core_report = generate_core_report_from_log(log_content, case_id, report_type)
    
    # Determine output filename
    if args.output:
        report_file = args.output
    else:
        report_file = f"core_report_{case_id}_en.txt"
    
    # Save core report
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(core_report)
        print(f"✓ Core report saved to: {report_file}")
    except Exception as e:
        print(f"✗ Failed to save core report: {e}")
        return
    
    print("\nProcessing completed!")
    print(f"Report type: {report_type}")
    print(f"Input file: {args.log_file}")
    print(f"Output file: {report_file}")
    print(f"Case ID: {case_id}")

if __name__ == "__main__":
    main()

