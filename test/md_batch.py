from pathlib import Path

def load_md_files(folder):
    files = []
    for f in Path(folder).glob("*.md"):
        files.append({
            "id": f.stem,
            "content": f.read_text(encoding="utf-8")
        })
    return files


def make_batches(docs, batch_size=5):
    batches = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]

        text_block = ""
        for d in batch:
            text_block += f"""
[DOC_ID]: {d['id']}
[CONTENT]:
{d['content']}

-------------------------
"""

        batches.append(text_block.strip())

    return batches


def save_batches(batches, out_dir="batches"):
    Path(out_dir).mkdir(exist_ok=True)

    for i, b in enumerate(batches):
        with open(f"{out_dir}/batch_{i:03d}.txt", "w", encoding="utf-8") as f:
            f.write(b)


if __name__ == "__main__":
    docs = load_md_files("./md_files")
    batches = make_batches(docs, batch_size=5)
    save_batches(batches)

    print(f"완료: {len(batches)}개 batch 생성됨")