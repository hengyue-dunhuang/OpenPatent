import re
import pdfplumber

class PDFProcessor:
    """
    PDF 处理器类，用于将 PDF 文件按章节分割成多个部分。
    """
    SECTIONS = ['摘要', '权 利 要 求 书', '说 明 书', '说 明 书 附 图']
    
    def split_pdf(self, file_path):
        """
        将 PDF 文件按章节分割成多个部分，并返回一个字典，键为章节名，值为章节内容。

        参数:
        file_path (str): PDF 文件的路径。

        返回:
        dict: 包含章节名和对应内容的字典。
        """
        # 使用 pdfplumber 打开 PDF 文件并提取文本
        with pdfplumber.open(file_path) as pdf:
            full_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
            text = '\n'.join(full_text)
        
        # 初始化分割结果
        sections = {}
        section_starts = []
        
        # 为每个章节标题编译正则表达式
        patterns = {section: re.compile(rf'{section}') for section in self.SECTIONS}
        
        # 查找每个章节标题第一次出现的位置
        current_pos = 0
        for section in self.SECTIONS:
            match = patterns[section].search(text, pos=current_pos)
            if match:
                section_starts.append((section, match.start()))
                current_pos = match.start()
        
        # 添加文本开头和结尾作为边界
        section_starts = [('', 0)] + section_starts + [('', len(text))]
        
        # 根据位置分割文本
        for i in range(len(section_starts) - 1):
            start_section, start_pos = section_starts[i]
            _, end_pos = section_starts[i + 1]
            section_name = '前言' if i == 0 else section_starts[i][0]
            sections[section_name] = text[start_pos:end_pos].strip()
        
        return sections

if __name__ == "__main__":
    # 请替换为实际的 PDF 文件路径
    pdf_file_path = 'H:\项目\OpenPatent\参考专利\专利文件\基于大语言模型的xx系统\一种基于大语言模型的日志分析方法.pdf'
    processor = PDFProcessor()
    result = processor.split_pdf(pdf_file_path)
    for section, content in result.items():
        print(f"Section: {section}")
        print(content[:200])  # 打印前 200 个字符
        print("-" * 50)