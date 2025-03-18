import gradio as gr
from httpx import AsyncClient
from docx import Document
import numpy as np
# 在WebUI类初始化前配置HTTP客户端
gr.routes.client = AsyncClient(verify=False)

from pdf_processor import PDFProcessor
from vector_db import VectorDB
from llm_integration import PatentGenerator
import os
os.environ["SSL_CERT_FILE"] = r"H:\anadonda\envs\OpenPatent\Library\ssl\cacert.pem"

class WebUI:
    def __init__(self):
        self.patent_generator = PatentGenerator()
        self.current_stage = None
        self.current_doc_type = None
        self.db_paths = {
            '摘要': r'dbs\abstract',
            '说 明 书': r'dbs\specification',
            '权 利 要 求 书': r'dbs\claims'
        }
        self.use_existing_db = False

    def init_interface(self):
        with gr.Blocks(title="OpenPatent 专利生成系统") as demo:
            gr.Markdown("## OpenPatent 专利文档生成系统")
            
            with gr.Tab("1. 选择参考专利"):
                ref_patents = gr.Files(label="上传参考专利文件(PDF)")
                process_btn = gr.Button("处理专利文件")
                process_btn2 = gr.Button("已有本地知识库，点击这里")
                process_output = gr.Markdown()
            
            with gr.Tab("2. 上传技术文档"):
                tech_doc = gr.File(label="技术文档(.docx)")
                
            with gr.Tab("3. 生成专利文档"):
                with gr.Row():
                    gen_spec_btn = gr.Button("生成说明书")
                    gen_abstract_btn = gr.Button("生成摘要")
                    gen_claims_btn = gr.Button("生成权利要求书")
                
                with gr.Column():
                    output_preview = gr.Chatbot(label="专利生成过程", height=500, elem_id="centered-chat")
                
                with gr.Row():
                    user_feedback = gr.Textbox(label="修改意见", lines=3)
                    submit_feedback = gr.Button("提交反馈", variant="primary")
            
            # 绑定事件
            process_btn.click(self.process_patents, inputs=ref_patents, outputs=process_output)
            process_btn2.click(self.load_existing_db, outputs=process_output)
            gen_spec_btn.click(self.generate_specification, inputs=tech_doc, outputs=output_preview)
            gen_abstract_btn.click(self.generate_abstract, inputs=tech_doc, outputs=output_preview)
            gen_claims_btn.click(self.generate_claims, inputs=tech_doc, outputs=output_preview)
            submit_feedback.click(self.submit_feedback, inputs=user_feedback, outputs=output_preview)
            
        return demo

    def process_patents(self, files):
        if not files:
            return "请上传参考专利文件"
        processor = PDFProcessor()
        
        section_array = []
        for file in files:
            sections = processor.split_pdf(file.name)
            for db_type in self.db_paths:
                section_content = sections.get(db_type)
                section_array.append(section_content)
                if section_content:
                    print(f"dbtype:{db_type}")
                    print(section_content)
                else:
                    print(f"Section {db_type} not found in PDF")
        section_array = np.array(section_array)
        section_array = section_array.reshape((-1,3))
        print(f"翻转前：{section_array}")
        section_array = section_array.T
        print(f"翻转后：{section_array}")
        self.db_list = [0 for i in range(3)]
        for i,db_type in enumerate(self.db_paths):
            self.db_list[i] = VectorDB(db_type)
            self.db_list[i].create_index(section_array[i])
            self.db_list[i].save_index(self.db_paths[db_type])
        self.use_existing_db = True
        return "参考专利处理完成，已建立三个知识库！"

    def load_existing_db(self):
        self.use_existing_db = True
        self.db_list = [0 for i in range(3)]
        for i, db_type in enumerate(self.db_paths):
            self.db_list[i] = VectorDB(db_type)
            self.db_list[i].load_index(self.db_paths[db_type])
        return "已加载本地知识库"

    def generate_specification(self, tech_doc):
        self.current_stage = "specification"
        return self._generate_draft(tech_doc, "说 明 书", "说明书")

    def generate_abstract(self, tech_doc):
        return self._generate_draft(tech_doc, "摘要", "摘要")

    def generate_claims(self, tech_doc):
        return self._generate_draft(tech_doc, "权 利 要 求 书", "权利要求书")

    def _generate_draft(self, tech_doc, db_type: str, doc_type: str):
        if not self.use_existing_db:
            return [("系统", "请先处理参考专利或选择已有知识库")]
        if tech_doc is None:
            return [("系统", "请先上传技术文档")]

        # 读取技术文档内容
        doc = Document(tech_doc.name)
        query = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        # 加载向量数据库
        db_map = {"摘要":0,"说 明 书":1,"权 利 要 求 书":2}
        vector_db = self.db_list[db_map[db_type]]
        # 检索相关内容
        related_patents = vector_db.query(query, top_k=2)
        print(related_patents)
        context = "\n".join(related_patents) if related_patents else "无相关专利内容"
        
        # 生成专利文档
        try:
            content = self.patent_generator.generate_draft(query, context, doc_type)
            self.current_doc_type = doc_type
            return [
                ("系统", "开始生成专利文档..."),
                ("助手", content)
            ]
        except Exception as e:
            return [
                ("系统", "开始生成专利文档..."),
                ("助手", f"生成失败: {str(e)}")
            ]

    def submit_feedback(self, feedback):
        if not self.current_doc_type:
            return [("系统", "请先生成草案")]
        if not feedback.strip():
            return [("系统", "请输入修改意见")]

        try:
            revised_content = self.patent_generator.revise_draft(feedback, self.current_doc_type)
            return [
                ("系统", "开始修订文档..."),
                ("助手", revised_content)
            ]
        except Exception as e:
            return [
                ("系统", "开始修订文档..."),
                ("助手", f"修订失败: {str(e)}")
        ]

if __name__ == "__main__":
    WebUI().init_interface().launch()
