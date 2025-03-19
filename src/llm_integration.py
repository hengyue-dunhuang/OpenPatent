import os
import requests
from typing import Dict, Optional, Generator
from dotenv import load_dotenv
import logging
from openai import OpenAI
load_dotenv()

class PatentGenerator:
    def __init__(self):
        self.api_base = "https://openrouter.ai/api/v1"
        self.api_key = os.getenv('open_router_key')
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.current_draft: Dict[str, str] = {}
        self.query: str = ""
        self.context: str = ""

    def generate_draft(self, query: str, context: str, doc_type: str) -> str:
        self.query = query
        self.context = context

        prompt = f'''你是一个专业的专利申请文档撰写助手。请参考以下技术文档的行文风格和格式，并基于提供的相关专利内容，生成一段高质量的{doc_type}。该内容应直接适用于专利申请书中的对应部分。

### 要求：
1. **风格和格式**：严格模仿【参考技术文档】的行文风格、段落结构和术语使用。
2. **格式要求**：输出必须使用XML标签包裹，结构示例：
<标题>技术领域</标题>
<段落>[0001] 内容...</段落>
每个段落必须用<段落>标签包裹，标题用<标题>标签。
3. **内容生成**：基于【相关专利内容】生成具体的技术描述，确保逻辑清晰、表达准确。
4. **专业性**：使用专利申请中常见的专业术语和表达方式。
5. **完整性**：确保生成的{doc_type}内容完整，包含所有必要的技术细节和描述。
6. **专利撰写规范**：
   - 如果{doc_type}是**权利要求**，请确保准确描述发明的技术特征和保护范围。
   - 如果{doc_type}是**说明书**部分，请详细描述技术方案的实施方式、优点和具体示例。
### 参考技术文档（请模仿其风格和格式）：
{context}

### 相关专利内容（请基于此内容生成）：
{query}

请根据以上要求，生成专业的{doc_type}。'''

        try:
            response = self.client.chat.completions.create(
                model="qwen/qwq-32b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                stream=False,
            )
            
            content = response.choices[0].message.content
            self.current_draft[doc_type] = content
            return content
        except Exception as e:
            error_msg = f'生成{doc_type}失败: {str(e)}'
            logging.error(error_msg)
            return error_msg

    def revise_draft(self, feedback: str, doc_type: str) -> str:
        if doc_type not in self.current_draft:
            return '请先生成初稿'

        prompt = f'''你是一个专业的专利申请文档撰写助手。请根据用户反馈修改{doc_type}，同时确保修订后的内容仍然符合原始技术文档的风格和格式，并基于相关专利内容。

### 要求：
1. **风格和格式**：继续模仿【原始技术文档】的行文风格和格式。
2. **内容修改**：根据【修改意见】对【当前版本】进行修订，确保修改后的内容准确、完整。
3. **专业性**：保持专利申请的专业术语和表达方式。
4. **一致性**：确保修订后的内容与【相关专利内容】保持一致。
5. **专利撰写规范**：
   - 如果{doc_type}是**权利要求**，请确保技术特征和保护范围的准确性。
   - 如果{doc_type}是**说明书**部分，请确保技术方案的描述详细且具有可实施性。

### 模仿和参考的原始技术文档（参考风格和格式）：
{self.context}

### 相关专利内容：
{self.query}

### 当前版本：
{self.current_draft[doc_type]}

### 修改意见：
{feedback}

请根据以上要求，生成修订后的{doc_type}。'''

        try:
            response = self.client.chat.completions.create(
                model="qwen/qwq-32b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                stream=False,
            )
            
            content = response.choices[0].message.content
            self.current_draft[doc_type] = content
            return content
        except Exception as e:
            error_msg = f'修订{doc_type}失败: {str(e)}'
            logging.error(error_msg)
            return error_msg
