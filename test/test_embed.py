import os
from docling.document_converter import DocumentConverter

file_path = r"D:\rag_project\test\test_sample.pdf"

if not os.path.exists(file_path):
    print(f"에러: 파일을 찾을 수 없습니다 -> {file_path}")
else:
    print("파일을 찾았습니다! 변환 시작")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    
    md_text = result.document.export_to_markdown()
    with open("result_docling.md", "w", encoding="utf-8") as f:
        f.write(md_text)
    print("변환 완료 : result_docling.md")