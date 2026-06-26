import os
import re

def clean_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 정규식: ## Q: (질문) 형태를 찾아서 제목으로 바꾸고, - (답변)을 본문으로 변환
    # 이 패턴은 님의 데이터 구조에 맞춰 조금씩 조정 가능합니다.
    new_content = re.sub(r'## Q: (.*)\n\n- (.*)', r'## \1\n\n\2', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

# 폴더 내 모든 파일 일괄 처리
data_dir = "."
for filename in os.listdir(data_dir):
    if filename.endswith(".md"):
        clean_markdown_file(os.path.join(data_dir, filename))
        print(f"{filename} 변환 완료!")