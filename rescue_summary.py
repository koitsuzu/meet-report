import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 環境設定
BASE_DIR = Path("/Users/qaro/Desktop/bw/bw-meeting/fastapi-export-for-github")
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ 找不到 GEMINI_API_KEY，請檢查 .env")
    sys.exit(1)

# 2. 檔案路徑
transcript_path = Path("/Users/qaro/Desktop/bw/bw-meeting/fastapi-export-for-github/output/meetings/逐字稿_GMT20260414-004959_Recording_2560x1440_20260414_192623.md")
schema_path = BASE_DIR / "skills" / "meeting_summary" / "references" / "schema.json"
glossary_path = BASE_DIR / "skills" / "meeting_summary" / "references" / "glossary.json"
output_dir = BASE_DIR / "output" / "meetings"
script_path = BASE_DIR / "skills" / "meeting_summary" / "scripts" / "generate_docs.py"

# 3. 讀取資料
transcript = transcript_path.read_text(encoding="utf-8")
meeting_schema = schema_path.read_text(encoding="utf-8") if schema_path.exists() else "{}"
glossary = json.loads(glossary_path.read_text(encoding="utf-8")) if glossary_path.exists() else {}

# 4. 初始化 Gemini (處理能力強，適合長文本)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

# 5. 提示詞
glossary_section = ""
if glossary:
    glossary_section = "\n### 專有名詞規範：\n"
    if glossary.get("people"): glossary_section += f"- 人員：{', '.join(glossary['people'])}\n"
    if glossary.get("terms"): glossary_section += f"- 術語：{', '.join(glossary['terms'])}\n"

SUMMARY_SYSTEM = f"""
你是一位專業的會議記錄整理專家。
請從逐字稿中提取結構化的會議資訊，並輸出為嚴格符合以下 JSON Schema 的 JSON。

### JSON Schema 規範：
{meeting_schema}

{glossary_section}

### 輸出要求：
- 只返回純 JSON，不要有任何 markdown 標記。
- 使用繁體中文。
"""

user_query = "sap pp、mm模組 物料主檔作業流程會議"

print(f"🚀 開始摘要處理 (Gemini 2.5 Flash)... 逐字稿長度: {len(transcript)} 字")

try:
    result = llm.invoke([
        ("system", SUMMARY_SYSTEM),
        ("human", f"使用者描述：{user_query}\n\n逐字稿如下：\n{transcript}")
    ])

    raw_json = result.content.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("\n", 1)[1].rsplit("```", 1)[0]
    raw_json = raw_json.strip()

    meeting_data = json.loads(raw_json)
    print("✅ 結構化資料提取完成！")

    # 6. 儲存結果
    stem = f"會議記錄_{datetime.now().strftime('%Y%m%d_%H%M')}_SAP_PP_MM"
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(meeting_data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 7. 產生文件
    print("📄 正在生成 Word 與 Markdown 文件...")
    subprocess.run(
        [sys.executable, str(script_path), "--input", str(json_path), "--dir", str(output_dir), "--output", stem],
        check=True
    )
    print(f"✨ 任務完成！所有檔案已儲存於: {output_dir}")
    print(f"   - JSON: {stem}.json")
    print(f"   - Word: {stem}.docx")
    print(f"   - Markdown: {stem}.md")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")
