# MCC
Model Confrontation and Collaboration


API 获取和Cost analysis. 
MCC收录了GPT-o1 (OpenAI), DeepSeek-R1 (DeepSeek), and Qwen‑QwQ (Alibaba)，对应的API获取和使用说明均可通过访问模型官网进行。需要注意的是，若由于API地区可用性限制，可通过访问https://cloud.siliconflow.cn/models获取DeepSeek-R1和Qwen‑QwQ的使用API。访问Autogen平台获取GPT-o1的使用API。The computational and financial cost of executing our MCC framework scales with the number of adversarial debates rounds and varies by task type. The MCC framework utilizes three reasoning models, with their associated token costs (at the time of our study) listed below: 官方价格如下
GPT-o1: Input $0.0011 / 1K tokens, Output $0.0044 / 1K tokens
Qwen‑QwQ: Input $0.00014 / 1K tokens, Output $0.00057 / 1K tokens
DeepSeek-R1: Input $0.00014 / 1K tokens, Output $0.00219 / 1K tokens
Note that these rates reflect pricing at the time of experimentation and may fluctuate. As newer models are released, the cost of LLM usage typically trends downward.
