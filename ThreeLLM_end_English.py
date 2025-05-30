#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===================== 配置部分 =====================
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
from typing import Tuple, List, Dict, Any
from openai import OpenAI  # 用于GPT API调用

# 自定义输出类，同时输出到控制台和日志文件
class TeeOutput:
    """将输出同时发送到终端和日志文件的类"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a", encoding="utf-8")
        # 在日志文件开头写入时间戳
        self.logfile.write(f"\n{'='*50}\n")
        self.logfile.write(f"日志开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logfile.write(f"{'='*50}\n\n")
        self.logfile.flush()

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()  # 确保立即写入文件

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
        
    def close(self):
        """关闭日志文件"""
        if self.logfile:
            # 在日志文件结尾写入时间戳
            self.logfile.write(f"\n{'='*50}\n")
            self.logfile.write(f"日志结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.logfile.write(f"{'='*50}\n\n")
            self.logfile.close()
            self.logfile = None

# ===== 配置API密钥和URL =====
# GPT API配置  
GPT_API_KEY = "sk-jPgtADBXAQGoy4JCr7MrpkGDVBDz7v9u9FyC1tJ4Fc0b2F1u"  # 修改为原始API密钥
GPT_API_URL = "https://api.chatanywhere.tech/v1/chat/completions"  # 修改为原始API URL
GPT_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GPT_API_KEY}"
}

# Qwen API配置
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "sk-egbetwgfnaopvplrtpenocsbhsmferlbiyggubouibdpwulm")  # 从环境变量获取或使用默认值
QWEN_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
QWEN_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {QWEN_API_KEY}"
}

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-05607ff476624677a92ef34fe0b08c71"  # DeepSeek API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

# 检查文件是否存在
def check_file_exists(file_path):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        print("错误: 文件 '{}' 不存在!".format(file_path))
        return False
    return True

# 读取医学多选题数据
def load_medical_mcq_data(file_path):
    """加载医学多选题数据集"""
    try:
        if not check_file_exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        print(f"正在加载数据集: {file_path}")
        
        # 读取JSONL文件而非CSV文件
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 确保行不为空
                    data_list.append(json.loads(line.strip()))
        
        # 创建DataFrame
        dataset = pd.DataFrame(data_list)
        print(f"成功加载数据集，共 {len(dataset)} 个案例")
        return dataset
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        traceback.print_exc()
        raise

# 获取选项列表
def get_choices(dataset, case_idx):
    """从数据集中获取指定案例的选项列表
    
    Args:
        dataset: 数据集
        case_idx: 案例索引
        
    Returns:
        str: 格式化的选项列表，如"1. 选项1\n2. 选项2"
    """
    try:
        # 获取选项
        if 'options' in dataset.columns:
            options = dataset.iloc[case_idx]['options']
            
            # 处理JSONL格式中的options字典
            if isinstance(options, dict):
                choices_list = []
                for i, (key, value) in enumerate(options.items(), 1):
                    choices_list.append(f"{i}. {value}")
                return "\n".join(choices_list)
            
            # 处理列表格式
            elif isinstance(options, list):
                choices_list = [f"{i+1}. {option}" for i, option in enumerate(options)]
                return "\n".join(choices_list)
            
            # 处理字符串格式（可能是JSON字符串）
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
        
        # 其他情况（保留原有逻辑）
        choice_cols = [col for col in dataset.columns if col.startswith('choice_') or col == 'choice']
        if choice_cols:
            choices = []
            for i in range(1, 10):  # 假设最多有9个选项
                col_name = f'choice_{i}'
                if col_name in dataset.columns and not pd.isna(dataset.iloc[case_idx][col_name]):
                    choices.append(f"{i}. {dataset.iloc[case_idx][col_name]}")
            
            if choices:
                return "\n".join(choices)
        
        # 检查其他选项列模式
        cols = dataset.columns.str.contains("choice")
        if any(cols):
            choices = dataset.iloc[case_idx][cols]
            choices_list = [f"{i+1}. {choice}" for i, choice in enumerate(choices) if not pd.isna(choice)]
            return "\n".join(choices_list)
        
        # 如果没有找到选项列，尝试使用A, B, C, D列
        option_letters = ['A', 'B', 'C', 'D', 'E']
        if all(letter in dataset.columns for letter in option_letters[:4]):
            choices = []
            for i, letter in enumerate(option_letters):
                if letter in dataset.columns and not pd.isna(dataset.iloc[case_idx][letter]):
                    choices.append(f"{i+1}. {dataset.iloc[case_idx][letter]}")
            
            if choices:
                return "\n".join(choices)
        
        # 最后的后备方案
        return "1. Choice A\n2. Choice B\n3. Choice C\n4. Choice D"
        
    except Exception as e:
        print(f"获取选项列表时出错: {str(e)}")
        traceback.print_exc()
        return "1. Error Choice 1\n2. Error Choice 2\n3. Error Choice 3\n4. Error Choice 4"


# ===================== GPT模型部分 =====================
# 构建GPT多选题提示词
def get_gpt_prompt(case_vignette, choices):
    """构建GPT多选题的提示词"""
    system_prompt = """You are analyzing a medical case that requires systematic clinical reasoning and precise diagnostic evaluation. You will be provided with a medical case and possible diagnostic options. Your task is to conduct a detailed medical analysis through careful reasoning, critically evaluate each option, and select the most appropriate diagnosis.

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

# 使用ChatGPT进行推理
def generate_gpt_answer(case_vignette, choices):
    """使用ChatGPT生成多选题答案"""
    try:
        prompt = get_gpt_prompt(case_vignette, choices)
        
        print("GPT正在推理答案...")
        t_generate_start = time.time()
        
        data = {
            "model": "o1-mini",  # 可以根据需要更改模型  o1-mini
            "messages": [
                {"role": "system", "content": "You are the GPT Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 5000,
            #"temperature": 0.5 # 降低温度以获得更确定的答案
        }

        print("正在发送请求到GPT API...")
        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
        t_generate = time.time() - t_generate_start
        print(f"GPT API响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            print(f"GPT生成答案完成，耗时 {t_generate:.2f}秒")
            return answer
        else:
            print(f"GPT API错误: {response.status_code}")
            print(f"错误详情: {response.text}")
            return f"抱歉，在处理您的问题时出现了错误。错误代码：{response.status_code}"

    except Exception as e:
        print(f"GPT生成答案时出错: {str(e)}")
        traceback.print_exc()
        return f"抱歉，在处理您的问题时出现了错误。错误信息：{str(e)}"



# ===================== Qwen模型部分 =====================
# 构建Qwen多选题提示词
def get_qwen_prompt(case_vignette, choices):
    """构建Qwen多选题的提示词"""
    system_prompt = """You are analyzing a medical case that requires systematic clinical reasoning and precise diagnostic evaluation. You will be provided with a medical case and possible diagnostic options. Your task is to conduct a detailed medical analysis through careful reasoning, critically evaluate each option, and select the most appropriate diagnosis.

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

# 使用Qwen进行推理
def generate_qwen_answer(case_vignette, choices):
    """使用Qwen生成多选题答案"""
    try:
        prompt = get_qwen_prompt(case_vignette, choices)
        
        print("Qwen正在生成答案...")
        t_generate_start = time.time()
        
        # 构建请求数据 - 简化参数
        data = {
            "model": "Qwen/QwQ-32B", 
            "messages": [
                {"role": "system", "content": "You are the Qwen Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,  # 降低温度以获得更确定的答案
            "max_tokens": 5000
        }

        # 发送请求到Qwen API
        print("正在发送请求到Qwen API...")
        response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data, timeout=300)
        t_generate = time.time() - t_generate_start
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"响应JSON结构: {list(response_data.keys())}")
            
            if 'choices' in response_data and response_data['choices']:
                print(f"choices结构: {list(response_data['choices'][0].keys())}")
                
                if 'message' in response_data['choices'][0]:
                    message = response_data['choices'][0]['message']
                    print(f"message结构: {list(message.keys())}")
                    
                    # 首先尝试从content获取内容
                    if 'content' in message and message['content'] and len(message['content'].strip()) > 0:
                        answer = message['content'].strip()
                    # 如果content为空，尝试从reasoning_content获取内容
                    elif 'reasoning_content' in message and message['reasoning_content']:
                        answer = message['reasoning_content'].strip()
                        print("从reasoning_content字段提取内容")
                    else:
                        print("警告：API返回的内容为空")
                        answer = "API返回内容为空"
                    
                    answer_length = len(answer)
                    print(f"提取的答案长度: {answer_length} 字符")
                    
                    if answer_length > 0:
                        print(f"答案前100字符: {answer[:100]}...")
                        print(f"Qwen生成答案完成，耗时 {t_generate:.2f}秒")
                        return answer
                    else:
                        print("提取的内容为空")
                        return "API返回内容为空"
                else:
                    print("错误：message字段不存在")
                    return "API响应结构异常：缺少message字段"
            else:
                print("错误：choices字段不存在或为空")
                return "API响应结构异常：缺少choices字段"
        else:
            print(f"Qwen API错误: {response.status_code}")
            print(f"错误详情: {response.text}")
            print("无法调用Qwen API，任务终止")
            sys.exit(1)

    except Exception as e:
        print(f"Qwen生成答案时出错: {str(e)}")
        traceback.print_exc()
        print("无法生成Qwen答案，任务终止")
        sys.exit(1)



# ===================== DeepSeek模型部分 =====================
# 构建DeepSeek多选题提示词
def get_deepseek_prompt(case_vignette, choices):
    """构建DeepSeek多选题的提示词"""
    system_prompt = """You are analyzing a medical case that requires systematic clinical reasoning and precise diagnostic evaluation. You will be provided with a medical case and possible diagnostic options. Your task is to conduct a detailed medical analysis through careful reasoning, critically evaluate each option, and select the most appropriate diagnosis.

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

# 使用DeepSeek进行推理
def generate_deepseek_answer(case_vignette, choices):
    """使用DeepSeek生成多选题答案"""
    try:
        prompt = get_deepseek_prompt(case_vignette, choices)
        
        print("DeepSeek正在生成答案...")
        t_generate_start = time.time()
        
        # 尝试使用OpenAI客户端进行调用
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are the DeepSeek Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5000,
                temperature=0.5  # 较低的temperature以获得更确定的答案
            )
            
            answer = response.choices[0].message.content
            t_generate = time.time() - t_generate_start
            print(f"DeepSeek生成答案完成，耗时 {t_generate:.2f}秒")
            return answer
            
        except Exception as e:
            print(f"使用OpenAI客户端调用DeepSeek API失败: {str(e)}")
            print("尝试使用requests直接调用API...")
            
            # 备用方案：直接使用requests调用，硅基流动的API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek Medical Model, a top-tier medical expert with exceptional clinical reasoning capabilities. Your primary task is to maximize diagnostic accuracy in medical MCQs. Your thorough reasoning analysis process is critical for achieving the highest possible diagnostic precision."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 5000,
                "temperature": 0.5 # 降低温度以获得更确定的答案
            }
            url = "https://api.siliconflow.cn/v1/chat/completions" # 硅基流动的URL
            headers = {
                "Authorization": "Bearer sk-egbetwgfnaopvplrtpenocsbhsmferlbiyggubouibdpwulm",
                "Content-Type": "application/json"
            }
    
            response = requests.post(url, headers=headers, json=data)
            t_generate = time.time() - t_generate_start
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['choices'][0]['message']['content'].strip()
                print(f"DeepSeek生成答案完成，耗时 {t_generate:.2f}秒")
                return answer
            else:
                print(f"DeepSeek API错误: {response.status_code}")
                print(f"错误详情: {response.text}")
                print("无法调用DeepSeek API，任务终止")
                sys.exit(1)
        
    except Exception as e:
        print(f"DeepSeek生成答案时出错: {str(e)}")
        traceback.print_exc()
        print("无法生成DeepSeek答案，任务终止")
        sys.exit(1)



# ===================== 模型辩论部分 =====================
# 从回答中提取选择的选项编号
def extract_model_choice(answer_text, choices_text=None):
    """从模型回答中提取最终选择的答案和是否同意对方观点
    
    Args:
        answer_text: 模型的完整回答文本
        choices_text: 选项文本，用于动态匹配选项和选择（可选）
    
    Returns:
        int: 提取的选项编号（1-n），如果无法提取则返回None
    """
    
    # 调试信息
    print("\n开始提取模型选择...")
    
    # 预处理：移除Markdown标记以便更准确地匹配
    # 将**text**格式转换为text
    clean_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer_text)
    
    # 提取模型是否同意对方观点
    first_paragraph = clean_answer.split("\n")[0] if "\n" in clean_answer else clean_answer[:200]
    if "I acknowledge" in first_paragraph or "I agree" in first_paragraph or "I accept" in first_paragraph:
        print("Model agrees with other models' opinions")
    elif "I do not acknowledge" in first_paragraph or "I disagree" in first_paragraph or "I do not accept" in first_paragraph:
        print("Model disagrees with other models' opinions")
    
    # 解析choices_text，建立选项内容到选项编号的映射和医学术语扩展映射
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
                # 处理简短名称匹配
                short_name = option_content.split()[0].lower() if ' ' in option_content else option_content.lower()
                option_content_to_num[short_name] = option_num
                
                # 为特定疾病添加常见扩展术语映射
                if "lung cancer" in option_content.lower():
                    medical_term_expansions["lung cancer"] = option_num
                    medical_term_expansions["lung cancer with brain metastasis"] = option_num
                    medical_term_expansions["brain metastasis from lung cancer"] = option_num
                elif "multiple sclerosis" in option_content.lower():
                    medical_term_expansions["multiple sclerosis"] = option_num
                    medical_term_expansions["ms"] = option_num
                elif "meningioma" in option_content.lower():
                    medical_term_expansions["meningioma"] = option_num
                elif "glioblastoma" in option_content.lower():
                    medical_term_expansions["glioblastoma"] = option_num
                    medical_term_expansions["glioblastoma multiforme"] = option_num
                    medical_term_expansions["gbm"] = option_num
    
    # 优先提取最终结论部分
    # 尝试识别最终结论段落
    # 常见的最终结论标记
    conclusion_markers = [
        "final decision", "final selection", "final choice", "final diagnosis", 
        "in conclusion", "to conclude", "final answer", "my conclusion"
    ]
    
    # 查找最终结论段落
    conclusion_text = ""
    lines = clean_answer.split("\n")
    for i, line in enumerate(lines):
        if any(marker.lower() in line.lower() for marker in conclusion_markers):
            # 获取该行及后续几行作为结论文本
            conclusion_text = "\n".join(lines[i:min(i+5, len(lines))])
            break
    
    # 如果没有找到明确的结论段落，使用最后几行
    if not conclusion_text:
        conclusion_text = "\n".join(lines[-15:])
    
    # 1. 首先严格匹配标准格式的"我的最终选择是：选项X (选项内容)"
    # 定义最终结论的严格匹配模式，支持Markdown格式
    final_choice_strict_patterns = [
        r'my final selection is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?\s*(?:\(([^)]+)\))?',  # 支持Markdown格式
        r'my final choice is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?\s*(?:\(([^)]+)\))?',
        r'my final diagnosis is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'my final decision is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'final (?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
        r'(?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?option\s*(\d+)(?:\*\*)?',
    ]
    
    # 优先在结论文本中匹配
    search_texts = [conclusion_text, clean_answer]
    
    # 优先匹配最明确的结论模式
    for text in search_texts:
        for pattern in final_choice_strict_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    option_num = int(match.group(1))
                    if 1 <= option_num <= 10:  # 放宽选项范围到10个
                        if len(match.groups()) > 1 and match.group(2):
                            option_content = match.group(2).strip()
                            print(f"【严格匹配】找到最终选择: 选项{option_num} ({option_content})")
                        else:
                            print(f"【严格匹配】找到最终选择: 选项{option_num}")
                        return option_num
                except (ValueError, IndexError):
                    continue
    
    # 2. 匹配疾病名称的直接引用
    # 在结论部分查找疾病/选项内容引用
    disease_reference_patterns = [
        r'my final (?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?([^*().,:;]+)(?:\*\*)?',
        r'(?:selection|choice|diagnosis) is[：:\s]?\s*(?:\*\*)?([^*().,:;]+)(?:\*\*)?',
    ]
    
    for text in search_texts:
        for pattern in disease_reference_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                disease_name = match.group(1).strip().lower()
                # 先检查扩展的医学术语
                if disease_name in medical_term_expansions:
                    print(f"【疾病引用】从结论中匹配到医学术语: {disease_name}，对应选项: {medical_term_expansions[disease_name]}")
                    return medical_term_expansions[disease_name]
                # 然后检查选项内容
                for content, num in option_content_to_num.items():
                    if content == disease_name or disease_name in content or content in disease_name:
                        print(f"【疾病引用】从结论中匹配到选项内容: {content}，对应选项: {num}")
                        return num
    
    # 3. 查找明确的选项引用
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
                        print(f"【明确引用】找到选项引用: 选项{option_num}")
                        return option_num
                except (ValueError, IndexError):
                    continue
    
    # 4. 在结论部分查找疾病/选项名称的具体引用
    conclusion_disease_patterns = [
        r'(?:determine|conclude)[^.,:;]*?(?:\*\*)?([^\s*().,:;]+)(?:\*\*)?\s*(?:is|as)[^.,:;]*?(?:diagnosis|selection)',
        r'(?:diagnosis|selection)[^.,:;]*?(?:is|as)[^.,:;]*?(?:\*\*)?([^\s*().,:;]+)(?:\*\*)?',
    ]
    
    for text in search_texts:
        for pattern in conclusion_disease_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                disease_name = match.group(1).strip().lower()
                # 先检查扩展的医学术语
                if disease_name in medical_term_expansions:
                    print(f"【结论引用】从结论中匹配到医学术语: {disease_name}，对应选项: {medical_term_expansions[disease_name]}")
                    return medical_term_expansions[disease_name]
                # 然后检查选项内容
                for content, num in option_content_to_num.items():
                    if content == disease_name or disease_name in content or content in disease_name:
                        print(f"【结论引用】从结论中匹配到选项内容: {content}，对应选项: {num}")
                        return num
    
    # 5. 退化策略：在结论部分查找简单选项引用
    # 但要确保这是最后的选择，避免误匹配
    simple_option_patterns = [
        r'(?:\*\*)?option\s*(\d+)(?:\*\*)?',
    ]
    
    # 仅在结论文本中搜索，避免在讨论过程中的误匹配
    for pattern in simple_option_patterns:
        match = re.search(pattern, conclusion_text, re.IGNORECASE)
        if match:
            try:
                option_num = int(match.group(1))
                if 1 <= option_num <= 10:
                    print(f"【简单匹配】从结论中找到选项编号: {option_num}")
                    print(f"注意：这是基于简单匹配的结果，可能不如严格匹配准确")
                    return option_num
            except (ValueError, IndexError):
                continue
    
    # 6. 最后的退化策略：查找医学术语或选项内容
    conclusion_lower = conclusion_text.lower()
    
    # 首先检查扩展的医学术语
    for term, num in medical_term_expansions.items():
        if term in conclusion_lower:
            print(f"【退化策略】从结论文本中匹配到扩展医学术语: {term}，对应选项: {num}")
            print(f"警告：这是基于内容匹配的弱推断，可能不准确")
            return num
    
    # 然后检查选项内容
    if option_content_to_num:
        for content, num in option_content_to_num.items():
            if content in conclusion_lower:
                print(f"【退化策略】从结论文本中匹配到选项内容: {content}，对应选项: {num}")
                print(f"警告：这是基于内容匹配的弱推断，可能不准确")
                return num
    
    # 如果依然无法提取，返回None
    print("警告：无法从回答中提取明确的选择")
    return None


# 初始化辩论
def initialize_debate(case_vignette, choices, force_disagree=False):
    """初始化辩论，获取三个模型的初始回答
    
    Args:
        case_vignette: 病例描述
        choices: 选项列表
        force_disagree: 是否强制模拟分歧（用于测试）
        
    Returns:
        dict: 三个模型的初始回答
    """
    print("="*50)
    print("开始医学案例辩论")
    print("="*80)
    
    # 建立选项编号到疾病名称的映射
    choice_to_disease = {}
    choice_lines = choices.strip().split('\n')
    for line in choice_lines:
        match = re.match(r'(\d+)\.\s*(.+)', line.strip())
        if match:
            option_num = int(match.group(1))
            disease_name = match.group(2).strip()
            choice_to_disease[option_num] = disease_name
    
    # 获取GPT的初始回答
    gpt_answer = generate_gpt_answer(case_vignette, choices)
    gpt_choice = extract_model_choice(gpt_answer, choices)
    
    print("\nGPT的诊断结论:")
    if gpt_choice:
        print(f"选择: 选项 {gpt_choice} ({choice_to_disease.get(gpt_choice, '未知疾病')})")
    else:
        print("未能提取明确选择")
    print("\nGPT的完整回答:")
    print("="*80)
    print(gpt_answer)
    print("="*80)
    
    # 获取Qwen的初始回答
    qwen_answer = generate_qwen_answer(case_vignette, choices)
    qwen_choice = extract_model_choice(qwen_answer, choices)
    
    print("\nQwen的诊断结论:")
    if qwen_choice:
        print(f"选择: 选项 {qwen_choice} ({choice_to_disease.get(qwen_choice, '未知疾病')})")
    else:
        print("未能提取明确选择")
    print("\nQwen的完整回答:")
    print("="*80)
    print(qwen_answer)
    print("="*80)
    
    # 获取DeepSeek的初始回答
    deepseek_answer = generate_deepseek_answer(case_vignette, choices)
    deepseek_choice = extract_model_choice(deepseek_answer, choices)
    
    print("\nDeepSeek的诊断结论:")
    if deepseek_choice:
        print(f"选择: 选项 {deepseek_choice} ({choice_to_disease.get(deepseek_choice, '未知疾病')})")
    else:
        print("未能提取明确选择")
    print("\nDeepSeek的完整回答:")
    print("="*80)
    print(deepseek_answer)
    print("="*80)
    
    # 强制模拟分歧
    if force_disagree:
        # 检查是否所有模型都选择了相同的选项
        if gpt_choice == qwen_choice == deepseek_choice and len(choice_to_disease) > 1:
            print("\n强制模拟模型分歧（用于测试）...")
            # 找到一个不同的选项作为Qwen的选择
            available_choices = list(choice_to_disease.keys())
            available_choices.remove(gpt_choice)
            qwen_choice = random.choice(available_choices)
            print(f"修改Qwen的选择为: 选项 {qwen_choice} ({choice_to_disease.get(qwen_choice, '未知疾病')})")
    
    # 检查是否达成一致
    if check_consensus([gpt_choice, qwen_choice, deepseek_choice]):
        print("\n三个模型初始诊断已达成一致！")
    else:
        print("\n三个模型初始诊断存在分歧！")
    
    # 返回初始结果
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

# 检查是否达成一致
def check_consensus(choices):
    """检查模型是否达成一致
    
    Args:
        choices: 模型选择的列表
    
    Returns:
        bool: 是否达成一致
    """
    # 过滤掉None值
    valid_choices = [c for c in choices if c is not None]
    
    # 如果没有有效选择，返回False
    if not valid_choices:
        return False
    
    # 检查所有有效选择是否相同
    return all(c == valid_choices[0] for c in valid_choices)


# 让Qwen回应其他模型的诊断
def qwen_responds_to_others(case_vignette, choices, gpt_answer, gpt_choice, deepseek_answer, deepseek_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """让Qwen对GPT和DeepSeek的诊断给出回应"""
    try:
        # 获取选项列表，建立选项编号到疾病名称的映射
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # 获取GPT和DeepSeek选择的疾病名称
        gpt_disease = choice_to_disease.get(gpt_choice, "未明确疾病") if gpt_choice else "未明确疾病"
        deepseek_disease = choice_to_disease.get(deepseek_choice, "未明确疾病") if deepseek_choice else "未明确疾病"

        # 获取自己之前的选择和疾病名称（如果有）
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "未明确疾病")
                # 构建提示，包含自己之前的选择和分析
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
        
        print("\nQwen正在回应其他模型的诊断...")
        
        # 构建请求数据 - 简化参数
        data = {
            "model": "Qwen/QwQ-32B",
            "messages": [
                {"role": "system", "content": "You are the Qwen medical reasoning model, engaged in an intense debate with other models."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.6,  # 升高温度增加批判性思维
            "max_tokens": 5000
        }
        
        # 发送请求到Qwen API
        response = requests.post(QWEN_API_URL, headers=QWEN_HEADERS, json=data,timeout=300)
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            choice = extract_model_choice(answer, choices)
            
            print(f"Qwen回应完成，选择: 选项 {choice}" if choice else "Qwen回应完成，未能提取明确选择")
            print("\nQwen对其他模型的回应内容:")
            print("="*80)
            print(answer)
            print("="*80)
            
            return {
                "answer": answer,
                "choice": choice
            }
        else:
            print(f"Qwen API错误: {response.status_code}")
            print(f"错误详情: {response.text}")
            return fallback_response("qwen")
    
    except Exception as e:
        print(f"生成Qwen回应时出错: {str(e)}")
        return fallback_response("qwen")

# 定义fallback_response函数
def fallback_response(model_name):
    """当API调用失败时返回的备用响应"""
    print(f"使用{model_name}的备用响应")
    return {
        "answer": f"{model_name}模型无法生成回应。可能是API限制或网络问题。",
        "choice": None
    }


# 让GPT回应其他模型的诊断
def gpt_responds_to_others(case_vignette, choices, qwen_answer, qwen_choice, deepseek_answer, deepseek_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """让GPT对Qwen和DeepSeek的诊断给出回应"""
    try:
        # 获取选项列表，建立选项编号到疾病名称的映射
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # 获取Qwen和DeepSeek选择的疾病名称
        qwen_disease = choice_to_disease.get(qwen_choice, "未明确疾病") if qwen_choice else "未明确疾病"
        deepseek_disease = choice_to_disease.get(deepseek_choice, "未明确疾病") if deepseek_choice else "未明确疾病"
        
        # 获取自己之前的选择和疾病名称（如果有）
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "未明确疾病")
        
        # 构建提示，包含自己之前的选择和分析
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
        
        print("\nGPT正在回应其他模型的诊断...")
        
        # 使用GPT API
        data = {
            "model": "o1-mini",
            "messages": [
                {"role": "system", "content": "You are the GPT medical reasoning model, engaged in an intense debate with other models."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 5000,
            #"temperature": 0.6 # 升高温度增加批判性思维
        }

        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['choices'][0]['message']['content'].strip()
            choice = extract_model_choice(answer, choices)
            
            print(f"GPT回应完成，选择: 选项 {choice}" if choice else "GPT回应完成，未能提取明确选择")
            print("\nGPT对其他模型的回应内容:")
            print("="*80)
            print(answer)
            print("="*80)
            
            return {
                "answer": answer,
                "choice": choice
            }
        else:
            print(f"API错误: {response.status_code}")
            return fallback_response("gpt")

    except Exception as e:
        print(f"生成GPT回应时出错: {str(e)}")
        return fallback_response("gpt")


# 让DeepSeek回应其他模型的诊断
def deepseek_responds_to_others(case_vignette, choices, gpt_answer, gpt_choice, qwen_answer, qwen_choice, debate_round, self_previous_answer=None, self_previous_choice=None):
    """让DeepSeek对GPT和Qwen的诊断给出回应"""
    try:
        # 获取选项列表，建立选项编号到疾病名称的映射
        choice_to_disease = {}
        choice_lines = choices.strip().split('\n')
        for line in choice_lines:
            match = re.match(r'(\d+)\.\s*(.+)', line.strip())
            if match:
                option_num = int(match.group(1))
                disease_name = match.group(2).strip()
                choice_to_disease[option_num] = disease_name
        
        # 获取GPT和Qwen选择的疾病名称
        gpt_disease = choice_to_disease.get(gpt_choice, "未明确疾病") if gpt_choice else "未明确疾病"
        qwen_disease = choice_to_disease.get(qwen_choice, "未明确疾病") if qwen_choice else "未明确疾病"
        
        # 获取自己之前的选择和疾病名称（如果有）
        self_previous_disease = ""
        if self_previous_choice and self_previous_answer:
            self_previous_disease = choice_to_disease.get(self_previous_choice, "未明确疾病")
        
        # 构建提示，包含自己之前的选择和分析
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
        
        print("\nDeepSeek正在回应其他模型的诊断...")
        
        t_generate_start = time.time()
        answer = ""
        choice = None
        
        # 尝试使用OpenAI客户端进行调用
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are the DeepSeek medical reasoning model, engaged in an intense debate with other models."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5000,
                temperature=0.6  # 升高温度增加批判性思维
            )
            
            answer = response.choices[0].message.content
            t_generate = time.time() - t_generate_start
            print(f"DeepSeek生成答案完成，耗时 {t_generate:.2f}秒")
            # 从生成的回答中提取DeepSeek的选择
            choice = extract_model_choice(answer, choices)
            
        except Exception as e:
            print(f"使用OpenAI客户端调用DeepSeek API失败: {str(e)}")
            print("尝试使用requests直接调用API...")
            
            # 备用方案：直接使用requests调用，硅基流动的API
            data = {
                "model": "Pro/deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "You are the DeepSeek medical reasoning model, engaged in an intense debate with other models."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 5000,
                "temperature": 0.6
            }
            url = "https://api.siliconflow.cn/v1/chat/completions" # 硅基流动的URL
            headers = {
                "Authorization": "Bearer sk-egbetwgfnaopvplrtpenocsbhsmferlbiyggubouibdpwulm",
                "Content-Type": "application/json"
            }
    
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['choices'][0]['message']['content'].strip()
                t_generate = time.time() - t_generate_start
                print(f"DeepSeek生成答案完成，耗时 {t_generate:.2f}秒")
                choice = extract_model_choice(answer, choices)
            else:
                print(f"DeepSeek API错误: {response.status_code}")
                print(f"错误详情: {response.text}")
                return fallback_response("deepseek")
        
        # 无论使用哪种方式获取答案，都确保输出回应内容
        if answer:
            print(f"DeepSeek回应完成，选择: 选项 {choice}" if choice else "DeepSeek回应完成，未能提取明确选择")
            print("\nDeepSeek对其他模型的回应内容:")
            print("="*80)
            print(answer)
            print("="*80)
            
            return {
                "answer": answer,
                "choice": choice,
                "_output_shown": True
            }
        else:
            print("未能获取DeepSeek回应")
            return fallback_response("deepseek")
    
    except Exception as e:
        print(f"生成DeepSeek回应时出错: {str(e)}")
        traceback.print_exc()
        return fallback_response("deepseek")




# 让模型进行辩论并判断是否达成共识
def conduct_debate(case_vignette, choices, correct_answer, max_rounds=3, force_disagree=False):
    """进行模型之间的辩论
    
    Args:
        case_vignette: 病例描述
        choices: 选项列表
        correct_answer: 正确答案
        max_rounds: 最大辩论轮次
        force_disagree: 是否强制模拟分歧（用于测试）
        
    Returns:
        dict: 辩论结果，包含最终选择和辩论历史
    """
    try:
        # 获取三个模型的初始回答
        initial_results = initialize_debate(case_vignette, choices, force_disagree)
        
        if not initial_results:
            print("初始化辩论失败，无法继续")
            return None
        
        gpt_result = initial_results["gpt"]
        qwen_result = initial_results["qwen"]
        deepseek_result = initial_results["deepseek"]
        
        # 记录初始选择，用于评估最终结果
        initial_gpt_choice = gpt_result["choice"]
        initial_qwen_choice = qwen_result["choice"]
        initial_deepseek_choice = deepseek_result["choice"]
        
        # 用于记录哪个模型更精准（如果我们知道正确答案）
        # 将疾病名称映射到选项编号，便于比较
        correct_choice = None
        gpt_initially_correct = False
        qwen_initially_correct = False
        deepseek_initially_correct = False
        
        if correct_answer:
            # 创建选项编号到疾病名称的映射
            choice_options = choices.strip().split('\n')
            option_mapping = {}
            for option in choice_options:
                match = re.match(r'(\d+)\.\s*(.+)', option.strip())
                if match:
                    number, disease = match.groups()
                    option_mapping[int(number)] = disease.strip()
            
            # 找到正确答案对应的选项编号
            correct_choice = None
            exact_match = False
            
            # 第一轮：寻找精确匹配
            for number, disease in option_mapping.items():
                if disease.lower() == correct_answer.lower() or disease.strip().lower() == correct_answer.lower():
                    correct_choice = number
                    exact_match = True
                    break
            
            # 第二轮：如果没有精确匹配，寻找单词边界匹配
            if not exact_match:
                for number, disease in option_mapping.items():
                    # 使用正则表达式进行单词边界匹配
                    if re.search(r'\b' + re.escape(correct_answer.lower()) + r'\b', disease.lower()):
                        correct_choice = number
                        break
            
            # 第三轮：如果前两轮都失败，使用更严格的部分匹配（仅作为后备）
            if not correct_choice:
                # 按选项编号排序，确保优先匹配顺序
                sorted_options = sorted(option_mapping.items())
                for number, disease in sorted_options:
                    # 只考虑短选项的完全匹配，避免将"X"匹配到"XIIa"
                    if correct_answer.lower() in disease.lower():
                        # 额外验证：如果是单字符答案，确保它是独立的
                        if len(correct_answer) == 1:
                            # 检查是否为独立的罗马数字或字母
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
                print(f"正确答案: 选项{correct_choice} ({correct_answer})")
                if gpt_initially_correct:
                    print("GPT的初始诊断是正确的")
                if qwen_initially_correct:
                    print("Qwen的初始诊断是正确的")
                if deepseek_initially_correct:
                    print("DeepSeek的初始诊断是正确的")
        
        # 检查初始是否已经达成一致
        initial_choices = [gpt_result["choice"], qwen_result["choice"], deepseek_result["choice"]]
        if check_consensus(initial_choices):
            consensus_choice = next((choice for choice in initial_choices if choice is not None), None)
            print(f"\n三个模型初始诊断已达成一致！所有模型都选择了选项{consensus_choice}")
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
            print(f"\n三个模型初始诊断存在分歧，开始辩论过程... GPT选择了选项{gpt_result['choice']}，Qwen选择了选项{qwen_result['choice']}，DeepSeek选择了选项{deepseek_result['choice']}")
        
        # 存储辩论历史
        debate_history = [{
            "round": 0,
            "gpt": gpt_result,
            "qwen": qwen_result,
            "deepseek": deepseek_result
        }]
        
        # 建立选项编号到疾病名称的映射
        choice_to_disease = {}
        choice_options = choices.strip().split('\n')
        for option in choice_options:
            match = re.match(r'(\d+)\.\s*(.+)', option.strip())
            if match:
                number, disease = match.groups()
                choice_to_disease[int(number)] = disease.strip()
        
        # 跟踪每轮辩论中每个模型的最新结果
        current_gpt_result = gpt_result
        current_qwen_result = qwen_result
        current_deepseek_result = deepseek_result


        # 开始辩论
        for round_num in range(1, max_rounds + 1):
            print(f"\n======== 辩论第{round_num}轮 ========")
            
            # GPT回应Qwen和DeepSeek，并传递自己之前的答案和选择
            gpt_response = gpt_responds_to_others(
                case_vignette, choices, 
                current_qwen_result["answer"], current_qwen_result["choice"],
                current_deepseek_result["answer"], current_deepseek_result["choice"], 
                round_num,
                self_previous_answer=current_gpt_result["answer"], 
                self_previous_choice=current_gpt_result["choice"]
            )
            # 更新当前GPT结果
            current_gpt_result = gpt_response

            gpt_choice = gpt_response["choice"]
            gpt_disease = choice_to_disease.get(gpt_choice, "未知疾病") if gpt_choice else "未明确疾病"
            qwen_disease = choice_to_disease.get(qwen_result["choice"], "未知疾病") if qwen_result["choice"] else "未明确疾病"
            deepseek_disease = choice_to_disease.get(deepseek_result["choice"], "未知疾病") if deepseek_result["choice"] else "未明确疾病"
            
            print(f"GPT回应后的选择：选项 {gpt_choice} ({gpt_disease})")
            print(f"Qwen选择：选项 {qwen_result['choice']} ({qwen_disease})")
            print(f"DeepSeek选择：选项 {deepseek_result['choice']} ({deepseek_disease})")
            
            # 如果GPT改变了立场并且原来是正确的
            if correct_choice and gpt_initially_correct and gpt_response["choice"] != correct_choice:
                print(f"警告: GPT从正确的选择 ({choice_to_disease.get(correct_choice, '')}) 改变为不正确的选择 ({gpt_disease})")
            
            # 检查是否达成一致
            current_choices = [gpt_response["choice"], qwen_result["choice"], deepseek_result["choice"]]
            if check_consensus(current_choices):
                consensus_choice = next((choice for choice in current_choices if choice is not None), None)
                consensus_disease = choice_to_disease.get(consensus_choice, "未知疾病")
                print(f"\n辩论第{round_num}轮: 所有模型已达成一致！(均选择了选项{consensus_choice} - {consensus_disease})")
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
        print(f"进行辩论时出错: {str(e)}")
        traceback.print_exc()
        return None

# 评估答案是否正确
def evaluate_answer(dataset, case_idx, model_choice):
    """评估模型选择是否正确
    
    Args:
        dataset: 数据集
        case_idx: 案例索引
        model_choice: 模型选择的选项编号
    
    Returns:
        bool: 是否正确
    """
    try:
        if model_choice is None:
            return False
            
        # 获取正确答案
        correct_answer = None
        
        # 检查answer_idx字段（JSONL格式中通常有此字段）
        if "answer_idx" in dataset.columns:
            correct_idx = dataset.iloc[case_idx]["answer_idx"]
            # 转换字母答案索引为数字
            if isinstance(correct_idx, str) and len(correct_idx) == 1 and correct_idx.isalpha():
                correct_answer_num = ord(correct_idx.upper()) - ord('A') + 1
                return model_choice == correct_answer_num
        
        # 直接检查answer字段
        if "answer" in dataset.columns:
            correct_text = dataset.iloc[case_idx]["answer"]
            
            # 获取选项列表
            choices = get_choices(dataset, case_idx)
            choice_options = choices.strip().split('\n')
            
            # 创建选项编号到选项内容的映射
            option_mapping = {}
            option_content_to_num = {}
            for option in choice_options:
                match = re.match(r'(\d+)\.\s*(.+)', option.strip())
                if match:
                    number, content = match.groups()
                    option_mapping[int(number)] = content.strip()
                    option_content_to_num[content.strip().lower()] = int(number)
            
            # 检查模型选择的选项内容是否与正确答案文本匹配
            if model_choice in option_mapping:
                model_answer_text = option_mapping[model_choice].lower()
                if correct_text.lower() == model_answer_text:
                    return True
                
            # 检查正确答案文本是否与某个选项内容匹配
            if correct_text.lower() in option_content_to_num:
                correct_answer_num = option_content_to_num[correct_text.lower()]
                return model_choice == correct_answer_num
                
        # 兜底：如果上述方法都失败，尝试原来的方法
        columns = dataset.columns
        answer_col = next((col for col in columns if col.lower() in ['answer', 'correct_answer', 'label']), None)
        
        if not answer_col:
            print("无法确定正确答案列名")
            return False
            
        # 尝试将答案转换为数字
        correct_answer = str(dataset.iloc[case_idx][answer_col])
        if correct_answer.isdigit():
            return model_choice == int(correct_answer)
        elif len(correct_answer) == 1 and correct_answer.upper() in "ABCDE":
            # 处理字母答案（A, B, C, D, E）
            correct_answer_num = ord(correct_answer.upper()) - ord('A') + 1
            return model_choice == correct_answer_num
            
        # 如果都失败，返回False
        print(f"无法比较答案: 模型选择 {model_choice}, 正确答案 {correct_answer}")
        return False
            
    except Exception as e:
        print(f"评估答案时出错: {str(e)}")
        traceback.print_exc()
        return False

# 处理单个案例的辩论
def process_single_debate(dataset_path, case_idx=0, max_rounds=3, force_disagree=False):
    """处理单个医学多选题案例的辩论
    
    Args:
        dataset_path: 数据集路径
        case_idx: 案例索引
        max_rounds: 最大辩论轮次
        force_disagree: 是否强制模拟分歧（用于测试）
        
    Returns:
        dict: 辩论结果
    """
    try:
        print(f"正在处理单个案例辩论（索引: {case_idx}）...")
        
        # 加载数据集
        dataset = load_medical_mcq_data(dataset_path)
        
        # 检查索引是否有效
        if case_idx < 0 or case_idx >= len(dataset):
            print(f"错误: 索引 {case_idx} 超出范围，数据集包含 {len(dataset)} 个案例")
            return None
        
        # 获取案例数据 - 调整字段以适应JSONL格式
        case_id = dataset.loc[case_idx, "question_id"] if "question_id" in dataset.columns else f"case_{case_idx}"
        case_vignette = dataset.loc[case_idx, "question"]  # JSONL中使用question而不是case_vignette
        category = dataset.loc[case_idx, "meta_info"] if "meta_info" in dataset.columns else "未知类别"
        choices = get_choices(dataset, case_idx)
        correct_answer = dataset.loc[case_idx, "answer"]
        
        print(f"案例ID: {case_id}")
        print(f"类别: {category}")
        print(f"病例描述: \n{case_vignette}")
        print(f"选项: \n{choices}")
        print(f"正确答案: {correct_answer}")
        
        # 进行辩论
        result = conduct_debate(case_vignette, choices, correct_answer, max_rounds, force_disagree)
        
        if not result:
            print("辩论过程失败")
            return None
        
        # 获取最终选择
        final_choice = result["final_choice"]
        
        # 创建选项编号到疾病名称的映射
        choice_to_disease = {}
        choice_options = choices.strip().split('\n')
        for option in choice_options:
            match = re.match(r'(\d+)\.\s*(.+)', option.strip())
            if match:
                number, disease = match.groups()
                choice_to_disease[int(number)] = disease.strip()
        
        # 最终诊断疾病
        final_disease = choice_to_disease.get(final_choice, "未知疾病") if final_choice else "未明确疾病"
        
        # 评估结果是否正确
        is_correct = evaluate_answer(dataset, case_idx, final_choice)
        
        # 输出最终结果
        print("\n========= 最终辩论结果 =========")
        if result["consensus"]:
            print(f"GPT，Qwen与DeepSeek-R1达成共识！最终诊断：选项{final_choice} - {final_disease}")
        else:
            # 三轮后未达成共识，通过多数投票决定
            print(f"GPT，Qwen与DeepSeek-R1未达成共识")
            print(f"通过多数投票决定，最终诊断：选项{final_choice} - {final_disease}")
        
        print(f"正确答案: 选项? - {correct_answer}")
        print(f"最终诊断是否正确: {'✓ 正确' if is_correct else '✗ 错误'}")
        
        # 构建输出结果
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
        print(f"处理单个案例辩论时出错: {str(e)}")
        traceback.print_exc()
        return None

# 主函数
def main():
    try:
        # 设置输出重定向到终端和日志文件
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        sys.stdout = TeeOutput(log_file)
        
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='医学模型辩论系统')
        parser.add_argument('-n', '--num_cases', type=int, default=1, help='要处理的病例数量 (默认: 1)')
        parser.add_argument('-s', '--start_idx', type=int, default=0, help='起始病例索引 (默认: 0)')
        parser.add_argument('-f', '--force_disagree', action='store_true', help='强制模拟模型分歧，以测试裁判功能')
        parser.add_argument('--force_judge', action='store_true', help='无论辩论结果如何都强制使用DeepSeek裁判')
        args = parser.parse_args()
        
        # 数据集路径
        dataset_path = "./test.jsonl"
        
        # 检查数据集是否存在
        if not check_file_exists(dataset_path):
            print("程序终止，数据集不存在。")
            exit(1)
        
        # 处理多个案例辩论
        print("="*50)
        print("GPT与Qwen与DeepSeek-R1医学诊断辩论，模型对抗与协作正式开始")
        print("="*50)
        
        # 创建结果目录
        results_dir = "debate_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 创建汇总结果列表
        summary_results = []
        
        # 处理多个案例
        for i in range(args.start_idx, args.start_idx + args.num_cases):
            print("\n" + "="*50)
            print(f"处理案例 {i+1}/{args.start_idx + args.num_cases} (索引: {i})")
            print("="*50)
            
            max_debate_rounds = 3  # 最大辩论轮次
            
            try:
                # 处理单个案例
                result = process_single_debate(dataset_path, case_idx=i, max_rounds=max_debate_rounds, force_disagree=args.force_disagree)
                
                if result:
                    # 保存单个案例结果到JSON文件
                    case_result_file = os.path.join(results_dir, f"debate_result_case_{i}.json")
                    with open(case_result_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"\n案例 {i} 的辩论结果已保存到 {case_result_file}")
                    
                    # 添加到汇总结果
                    summary_result = {
                        "case_id": result["case_id"],
                        "category": result["category"],
                        "correct_answer": result["correct_answer"],
                        "consensus": result["debate_result"]["consensus"],
                        "voting_needed": not result["debate_result"]["consensus"],  # 如果没有达成共识，就需要多数投票
                        "final_choice": result["debate_result"]["final_choice"],
                        "is_correct": result["is_correct"],
                        "stance_changes": result["debate_result"].get("stance_changes", {}),
                        "debate_result": {  # 为了保持与统计代码兼容，添加完整的debate_result
                            "stance_changes": result["debate_result"].get("stance_changes", {}),
                            "initial_choices": result["debate_result"].get("initial_choices", {}),
                            "debate_history": result["debate_result"].get("debate_history", []),
                            "correct_choice": result["debate_result"].get("correct_choice")
                        }
                    }
                    summary_results.append(summary_result)
            except Exception as e:
                print(f"处理案例 {i} 时出错: {str(e)}")
                traceback.print_exc()
        
        # 保存汇总结果
        summary_file = os.path.join(results_dir, "debate_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=2)
        
        # 输出汇总统计
        if summary_results:
            total_cases = len(summary_results)
            correct_cases = sum(1 for r in summary_results if r["is_correct"])
            consensus_cases = sum(1 for r in summary_results if r["consensus"])
            voting_cases = sum(1 for r in summary_results if not r["consensus"])  # 不达成共识的案例需要多数投票
            
            # 统计模型立场变化情况
            gpt_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("gpt_changed", False))
            qwen_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("qwen_changed", False))
            deepseek_changed_stance = sum(1 for r in summary_results if r["stance_changes"].get("deepseek_changed", False))
            
            # 统计从正确变为错误的情况
            gpt_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("gpt_changed_from_correct", False))
            qwen_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("qwen_changed_from_correct", False))
            deepseek_changed_from_correct = sum(1 for r in summary_results if r["stance_changes"].get("deepseek_changed_from_correct", False))
            
            # 统计从错误变为正确的情况（模型自我纠正的正面案例）
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
            print("辩论结果统计")
            print("="*50)
            print(f"总共处理案例数: {total_cases}")
            print(f"正确诊断案例数: {correct_cases} ({correct_cases/total_cases:.2%})")
            print("-" * 40)  # 分隔线
            print(f"模型达成共识案例数: {consensus_cases} ({consensus_cases/total_cases:.2%})")
            print(f"需要多数投票决定案例数: {voting_cases} ({voting_cases/total_cases:.2%})")
            print("-" * 40)  # 分隔线
            # 模型立场变化统计
            print(f"GPT改变立场案例数: {gpt_changed_stance} ({gpt_changed_stance/total_cases:.2%})")
            print(f"Qwen改变立场案例数: {qwen_changed_stance} ({qwen_changed_stance/total_cases:.2%})")
            print(f"DeepSeek改变立场案例数: {deepseek_changed_stance} ({deepseek_changed_stance/total_cases:.2%})")
            print("-" * 40)  # 分隔线
            # 从正确变为错误统计（负面影响）
            print(f"GPT从正确变为错误案例数: {gpt_changed_from_correct} ({gpt_changed_from_correct/total_cases:.2%})")
            print(f"Qwen从正确变为错误案例数: {qwen_changed_from_correct} ({qwen_changed_from_correct/total_cases:.2%})")
            print(f"DeepSeek从正确变为错误案例数: {deepseek_changed_from_correct} ({deepseek_changed_from_correct/total_cases:.2%})")
            print("-" * 40)  # 分隔线
            # 从错误变为正确统计（正面影响）
            print(f"GPT从错误变为正确案例数: {gpt_changed_to_correct} ({gpt_changed_to_correct/total_cases:.2%})")
            print(f"Qwen从错误变为正确案例数: {qwen_changed_to_correct} ({qwen_changed_to_correct/total_cases:.2%})")
            print(f"DeepSeek从错误变为正确案例数: {deepseek_changed_to_correct} ({deepseek_changed_to_correct/total_cases:.2%})")
            print("-" * 40)  # 分隔线
            print(f"\n汇总结果已保存到 {summary_file}")
            print(f"日志文件已保存到 {log_file}")
        
        # 关闭日志文件
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            # 恢复标准输出
            sys.stdout = sys.__stdout__
    
    except Exception as e:
        print(f"程序执行过程中出错: {str(e)}")
        traceback.print_exc()
        
        # 确保关闭日志文件
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()

# 运行示例：
# python ThreeLLM.py -n 3                         # 处理前3个案例
# python ThreeLLM.py -n 5 -s 10                   # 处理索引10-14的5个案例
# python ThreeLLM.py -n 2 -f                      # 处理前2个案例，强制模拟分歧
# python ThreeLLM.py --num_cases=3 --start_idx=5  # 处理索引5-7的3个案例 