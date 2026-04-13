# -*- coding: utf-8 -*-
"""
Meeting Multi-Agent System
=========================
基於 LangGraph Supervisor Pattern 的會議記錄自動化系統。

流程：
  使用者輸入 (影片路徑 + 描述)
        │
        ▼
  Supervisor ──→ transcriber ──→ Supervisor ──→ summary ──→ Supervisor ──→ finish

技術棧：
  - Supervisor / Summary LLM: Gemini 2.5 Flash (via langchain-google-genai)
  - Transcription: Groq Whisper API (whisper-large-v3-turbo)
  - Document Generation: skills/meeting_summary/scripts/generate_docs.py
"""

import os
import sys
import json
import tempfile
import subprocess
from typing import Literal, TypedDict, List
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── LangGraph / LangChain imports ─────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ── Groq for Whisper ──────────────────────────────────────────────────────────
from groq import Groq

# ── moviepy for audio extraction ──────────────────────────────────────────────
from moviepy import VideoFileClip


# ============================================================================
# 1. 初始化模型
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    print("❌ 錯誤：請在 .env 中設定 GEMINI_API_KEY")
    sys.exit(1)

# Supervisor 用 Gemini 2.5 Flash（用量極少，適合雲端）
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)
print("✅ Supervisor LLM: Gemini 2.5 Flash")

# Summary 用 Groq Llama（用量最大，優先選可本地部署的開源模型）
if GROQ_API_KEY:
    summary_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    print("✅ Summary LLM: Groq Llama 3.3 70B (可本地替換為 Ollama)")
else:
    summary_llm = llm  # fallback 到 Gemini
    print("⚠️  GROQ_API_KEY 未設定，Summary 改用 Gemini")


# ============================================================================
# 2. 定義 State（共用工作白板）
# ============================================================================

class AgentState(TypedDict):
    user_query: str          # 使用者的描述/指令
    video_path: str          # 影片檔案路徑
    next_agent: str          # supervisor 決定的下一步
    transcript: str          # 逐字稿（transcriber 產出）
    meeting_json: str        # 結構化 JSON 字串（summary agent 產出）
    output_files: List[str]  # 最終產出的檔案路徑
    final_answer: str        # 最終回覆給使用者的訊息
    route_history: List[str] # 路由歷史（除錯用）


# ============================================================================
# 3. Supervisor 路由決策
# ============================================================================

class RouteDecision(BaseModel):
    next_agent: Literal["transcriber", "summary", "finish"] = Field(
        description="下一步要呼叫的 agent。"
    )
    reason: str = Field(
        description="簡短說明為什麼這樣路由。"
    )


SUPERVISOR_SYSTEM = """
你是一個會議記錄多代理系統中的 supervisor。
你的任務不是直接回答使用者，而是根據目前的狀態判斷下一步要交給哪個 agent。

可選擇的 next_agent 只有：
- transcriber：負責將影片轉成逐字稿
- summary：負責將逐字稿轉成結構化會議記錄
- finish：所有工作已完成

請遵守以下規則：
1. 如果有影片路徑但還沒有逐字稿 (transcript 為空)，選 transcriber。
2. 如果已有逐字稿但還沒有會議記錄 (meeting_json 為空)，選 summary。
3. 如果 meeting_json 和 output_files 都已存在，選 finish。
4. 你不要自己回答問題。
5. reason 要簡短、白話。
"""


def supervisor_node(state: AgentState) -> AgentState:
    """
    穩定版 supervisor：
    - 硬規則優先保護流程，避免繞圈
    - LLM 僅用於第一輪的初步判斷（教學展示用）
    """
    transcript = state.get("transcript", "")
    meeting_json = state.get("meeting_json", "")
    final_answer = state.get("final_answer", "")

    # 硬規則保護
    if final_answer:
        next_agent = "finish"
        reason = "已有 final_answer，流程結束"
    elif meeting_json:
        next_agent = "finish"
        reason = "已完成文件生成，流程結束"
    elif transcript:
        next_agent = "summary"
        reason = "已有逐字稿，交給 summary agent 提取結構化資料並生成文件"
    else:
        # 第一輪：可用 LLM 做判斷（教學目的保留）
        try:
            router = llm.with_structured_output(RouteDecision)
            decision = router.invoke([
                ("system", SUPERVISOR_SYSTEM),
                ("human", f"""
使用者指令：{state.get('user_query', '')}
影片路徑：{state.get('video_path', '')}
逐字稿：{'[尚未產生]' if not transcript else '[已產生]'}
會議記錄：{'[尚未產生]' if not meeting_json else '[已產生]'}

請判斷下一步應該找誰。
""")
            ])
            next_agent = decision.next_agent
            reason = decision.reason
        except Exception as e:
            next_agent = "transcriber"
            reason = f"LLM 路由失敗，安全預設 transcriber：{type(e).__name__}"

        # 強制修正：第一輪一定先轉錄
        if next_agent != "transcriber":
            next_agent = "transcriber"
            reason = f"第一輪固定先做轉錄；原判斷為：{reason}"

    history = state.get("route_history", []) + [
        f"supervisor → {next_agent}（{reason}）"
    ]

    return {
        **state,
        "next_agent": next_agent,
        "route_history": history,
    }


# ============================================================================
# 4. Transcriber Agent (Groq Whisper)
# ============================================================================

def transcriber_node(state: AgentState) -> AgentState:
    """
    負責：
    1. 從影片提取音訊 (moviepy)
    2. 呼叫 Groq Whisper API 取得逐字稿
    3. 將逐字稿寫入 state
    """
    video_path = Path(state["video_path"])
    if not video_path.exists():
        history = state.get("route_history", []) + [
            f"transcriber：❌ 影片檔案不存在 {video_path}"
        ]
        return {
            **state,
            "transcript": "",
            "final_answer": f"錯誤：找不到影片檔案 {video_path}",
            "route_history": history,
        }

    print(f"\n🎬 Transcriber Agent 開始處理：{video_path.name}")

    # 4a. 提取音訊
    temp_dir = Path(tempfile.mkdtemp())
    audio_path = temp_dir / f"{video_path.stem}.mp3"

    print(f"  📢 正在提取音訊...")
    try:
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path), logger=None)
        video.close()
        print(f"  ✅ 音訊已提取至 {audio_path.name}")
    except Exception as e:
        history = state.get("route_history", []) + [
            f"transcriber：❌ 音訊提取失敗 {e}"
        ]
        return {
            **state,
            "transcript": "",
            "final_answer": f"錯誤：音訊提取失敗 - {e}",
            "route_history": history,
        }

    # 4b. 檢查音訊檔大小，若超過 25MB 則需要分割
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"  📊 音訊大小：{file_size_mb:.1f} MB")

    # 4c. 呼叫 Groq Whisper API
    if not GROQ_API_KEY:
        print("  ⚠️  未設定 GROQ_API_KEY，無法使用 Groq Whisper")
        history = state.get("route_history", []) + [
            "transcriber：❌ 缺少 GROQ_API_KEY"
        ]
        return {
            **state,
            "transcript": "",
            "final_answer": "錯誤：請在 .env 中設定 GROQ_API_KEY",
            "route_history": history,
        }

    print(f"  🎙️  正在呼叫 Groq Whisper API (whisper-large-v3-turbo)...")
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as f:
            # prompt 參數注入詞彙表，提高專有名詞辨識準確度
            response = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(audio_path.name, f.read()),
                response_format="verbose_json",
                prompt=WHISPER_PROMPT_HINT if WHISPER_PROMPT_HINT else None,
            )

        # 組裝逐字稿文字 + 說話者分離
        segments = response.segments
        SPEAKER_CHANGE_GAP = 1.5  # 超過 1.5 秒停頓視為換人說話

        # ── 簡易說話者辨識（基於停頓偵測）──
        speaker_id = 0
        speaker_map = {}  # segment index → speaker label
        speaker_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for i, seg in enumerate(segments):
            if i == 0:
                speaker_map[i] = 0
            else:
                gap = seg["start"] - segments[i - 1]["end"]
                if gap >= SPEAKER_CHANGE_GAP:
                    speaker_id += 1  # 偵測到停頓，切換說話者
                speaker_map[i] = speaker_id

        # 統計不同說話者數量
        unique_speakers = len(set(speaker_map.values()))
        # 如果 speaker_id 超過字母表，就用數字
        def get_label(sid):
            if sid < len(speaker_labels):
                return f"Speaker {speaker_labels[sid]}"
            return f"Speaker {sid + 1}"

        # ── 組裝純文字逐字稿（給 Summary Agent 用）──
        transcript_lines = []
        for i, seg in enumerate(segments):
            start_min = int(seg["start"] // 60)
            start_sec = int(seg["start"] % 60)
            end_min = int(seg["end"] // 60)
            end_sec = int(seg["end"] % 60)
            label = get_label(speaker_map[i])
            transcript_lines.append(
                f"[{start_min:02d}:{start_sec:02d} → {end_min:02d}:{end_sec:02d}] ({label}) {seg['text'].strip()}"
            )

        transcript_text = "\n".join(transcript_lines)
        word_count = sum(len(line.split()) for line in transcript_lines)

        # ── 產出帶時間軸的精美 Markdown 逐字稿 ──
        output_dir = Path("output") / "meetings"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_md_path = output_dir / f"逐字稿_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        md_lines = []
        md_lines.append(f"# 逐字稿 — {video_path.name}")
        md_lines.append("")
        md_lines.append(f"| 項目 | 內容 |")
        md_lines.append(f"|------|------|")
        md_lines.append(f"| 檔案 | `{video_path.name}` |")
        md_lines.append(f"| 語言 | {response.language} |")
        md_lines.append(f"| 總段數 | {len(segments)} |")
        md_lines.append(f"| 偵測說話者 | {unique_speakers} 人 |")
        md_lines.append(f"| 產出時間 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # 按說話者分組顯示
        current_speaker = None
        for i, seg in enumerate(segments):
            sid = speaker_map[i]
            label = get_label(sid)
            start_min = int(seg["start"] // 60)
            start_sec = int(seg["start"] % 60)
            end_min = int(seg["end"] // 60)
            end_sec = int(seg["end"] % 60)
            ts = f"`{start_min:02d}:{start_sec:02d} → {end_min:02d}:{end_sec:02d}`"
            text = seg["text"].strip()

            if sid != current_speaker:
                # 換人說話：加標題
                current_speaker = sid
                md_lines.append(f"### 🎙️ {label}")
                md_lines.append("")

            md_lines.append(f"- {ts}  {text}")

        md_lines.append("")
        md_lines.append("---")
        md_lines.append(f"*Generated by Meeting Agent — Whisper transcription with pause-based speaker detection*")

        raw_md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"  ✅ 轉錄完成！語言：{response.language}，共 {len(segments)} 段，約 {word_count} 詞")
        print(f"  📄 逐字稿已儲存：{raw_md_path}")
        print(f"  👥 偵測到約 {unique_speakers} 位不同說話者")

    except Exception as e:
        print(f"  ❌ Groq Whisper API 錯誤：{e}")
        history = state.get("route_history", []) + [
            f"transcriber：❌ Whisper API 失敗 {e}"
        ]
        return {
            **state,
            "transcript": "",
            "final_answer": f"錯誤：Whisper 轉錄失敗 - {e}",
            "route_history": history,
        }
    finally:
        # 清理暫存
        if audio_path.exists():
            audio_path.unlink()

    history = state.get("route_history", []) + [
        f"transcriber：✅ 完成轉錄（{len(segments)} 段，{unique_speakers} 位說話者）"
    ]

    return {
        **state,
        "transcript": transcript_text,
        "route_history": history,
    }


# ============================================================================
# 5. Summary Agent (Gemini 2.5 Flash + generate_docs.py)
# ============================================================================

# 讀取 schema 供 prompt 參考
SCHEMA_PATH = Path(__file__).parent / "skills" / "meeting_summary" / "references" / "schema.json"
if SCHEMA_PATH.exists():
    MEETING_SCHEMA = SCHEMA_PATH.read_text(encoding="utf-8")
else:
    MEETING_SCHEMA = "{}"
    print(f"⚠️  找不到 schema.json: {SCHEMA_PATH}")

# 讀取詞彙表（用於 Whisper + Gemini 的名詞校正）
GLOSSARY_PATH = Path(__file__).parent / "skills" / "meeting_summary" / "references" / "glossary.json"
if GLOSSARY_PATH.exists():
    GLOSSARY = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
else:
    GLOSSARY = {"people": [], "terms": [], "custom_corrections": {}}
    print(f"⚠️  找不到 glossary.json: {GLOSSARY_PATH}")

# 為 Whisper 組合 prompt hint（提高專有名詞辨識率）
WHISPER_PROMPT_HINT = "、".join(GLOSSARY.get("people", []) + GLOSSARY.get("terms", []))

# 為 Gemini Summary 組合名詞校正指引
GLOSSARY_PROMPT_SECTION = ""
if GLOSSARY.get("people") or GLOSSARY.get("custom_corrections"):
    GLOSSARY_PROMPT_SECTION = "\n### 專有名詞規範：\n"
    if GLOSSARY.get("people"):
        GLOSSARY_PROMPT_SECTION += f"- 已知人員名單（請嚴格使用此拼寫）：{', '.join(GLOSSARY['people'])}\n"
    if GLOSSARY.get("terms"):
        GLOSSARY_PROMPT_SECTION += f"- 已知術語：{', '.join(GLOSSARY['terms'])}\n"
    if GLOSSARY.get("custom_corrections"):
        corrections = GLOSSARY["custom_corrections"]
        GLOSSARY_PROMPT_SECTION += "- 常見轉錄錯誤對照（逐字稿中若出現左邊的詞，請替換為右邊）：\n"
        for wrong, correct in corrections.items():
            GLOSSARY_PROMPT_SECTION += f"  - 「{wrong}」→「{correct}」\n"

SUMMARY_SYSTEM = f"""
你是一位專業的會議記錄整理專家。
你的任務是從逐字稿中提取結構化的會議資訊，並輸出為嚴格符合以下 JSON Schema 的 JSON。

### JSON Schema 規範：
{MEETING_SCHEMA}

### 提取規則：
1. **會議資訊 (meeting_info)**：從對話開頭或上下文識別主題、平台、時間、記錄人。若無法確定，可合理推測或留空字串。
2. **出席者 (attendees)**：
   - expected：如果逐字稿有提及應到名單則列出，否則與 present 相同。
   - present：從逐字稿中所有有發言的人名中提取。
3. **討論內容 (discussion)**：
   - 按發言者分組，每人的 points 應為 1-5 條簡潔的重點摘要。
   - 使用繁體中文。保留必要的英文術語。
   - 若某人未有實質內容，points 為空陣列。
4. **公告 (announcements)**：提取全體公告事項。
5. **待辦事項 (action_items)**：識別任何承諾、分配、追蹤項目。
6. **備注 (facilitator_note)**：如有輪值或分工資訊。

{GLOSSARY_PROMPT_SECTION}

### 輸出要求：
- 只返回純 JSON，不要有任何 markdown 標記或額外文字。
- JSON 必須是合法的、可被 json.loads() 解析的。
"""


def summary_node(state: AgentState) -> AgentState:
    """
    負責：
    1. 將逐字稿送給 Llama (Groq) 提取結構化 JSON
    2. 呼叫 generate_docs.py 生成 Word 與 Markdown
    """
    transcript = state.get("transcript", "")
    user_query = state.get("user_query", "")
    video_path = Path(state.get("video_path", ""))

    print(f"\n📝 Summary Agent 開始處理...")

    # 5a. 呼叫 Groq Llama 提取結構化資料
    print(f"  🤖 正在呼叫 Llama 3.3 70B 提取會議結構化資料...")
    try:
        result = summary_llm.invoke([
            ("system", SUMMARY_SYSTEM),
            ("human", f"""
使用者描述：{user_query}

以下是會議的逐字稿：
{transcript}

請從中提取結構化的會議記錄 JSON。
""")
        ])

        # 清理 JSON（移除可能的 markdown 標記）
        raw_json = result.content.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[1]  # 移除第一行的 ```json
        if raw_json.endswith("```"):
            raw_json = raw_json.rsplit("```", 1)[0]
        raw_json = raw_json.strip()

        meeting_data = json.loads(raw_json)
        print(f"  ✅ 結構化資料提取完成！")
        print(f"     - 討論者：{len(meeting_data.get('discussion', []))} 人")
        print(f"     - 待辦事項：{len(meeting_data.get('action_items', []))} 項")

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON 解析失敗：{e}")
        print(f"  原始回覆前 500 字：{raw_json[:500]}")
        history = state.get("route_history", []) + [
            f"summary：❌ JSON 解析失敗"
        ]
        return {
            **state,
            "final_answer": f"錯誤：Gemini 回傳的 JSON 無法解析 - {e}",
            "route_history": history,
        }
    except Exception as e:
        print(f"  ❌ Gemini API 錯誤：{e}")
        history = state.get("route_history", []) + [
            f"summary：❌ Gemini API 失敗 {e}"
        ]
        return {
            **state,
            "final_answer": f"錯誤：Gemini 摘要失敗 - {e}",
            "route_history": history,
        }

    # 5b. 儲存 JSON
    output_dir = Path("output") / "meetings"
    output_dir.mkdir(parents=True, exist_ok=True)

    dt_str = meeting_data.get("meeting_info", {}).get("datetime", "")
    topic = meeting_data.get("meeting_info", {}).get("topic", "Meeting")
    try:
        dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M")
        stem = f"會議記錄_{dt.strftime('%Y%m%d')}_{topic}"
    except Exception:
        stem = f"會議記錄_{datetime.now().strftime('%Y%m%d%H%M')}_{video_path.stem}"

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(meeting_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  💾 JSON 已儲存：{json_path}")

    # 5c. 呼叫 generate_docs.py 產出 Word + Markdown
    script_path = Path(__file__).parent / "skills" / "meeting_summary" / "scripts" / "generate_docs.py"
    print(f"  📄 正在生成 Word 與 Markdown 文件...")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--input", str(json_path), "--dir", str(output_dir), "--output", stem],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  ⚠️  generate_docs.py 錯誤：{result.stderr}")
    except Exception as e:
        print(f"  ⚠️  文件生成過程出錯：{e}")

    # 收集產出檔案
    output_files = [str(f) for f in output_dir.glob(f"{stem}.*")]
    print(f"  ✅ 文件生成完成！")
    for f in output_files:
        print(f"     📁 {f}")

    # 組裝最終回覆
    final_answer = f"""✅ 會議記錄已完成！

📋 會議主題：{meeting_data.get('meeting_info', {}).get('topic', 'N/A')}
📅 日期時間：{meeting_data.get('meeting_info', {}).get('datetime', 'N/A')}
👥 與會人員：{', '.join(meeting_data.get('attendees', {}).get('present', []))}
📝 討論者：{len(meeting_data.get('discussion', []))} 人
📌 待辦事項：{len(meeting_data.get('action_items', []))} 項

📁 產出檔案：
"""
    for f in output_files:
        final_answer += f"  - {f}\n"

    history = state.get("route_history", []) + [
        f"summary：✅ 完成結構化提取與文件生成（{len(output_files)} 個檔案）"
    ]

    return {
        **state,
        "meeting_json": json.dumps(meeting_data, ensure_ascii=False),
        "output_files": output_files,
        "final_answer": final_answer,
        "route_history": history,
    }


# ============================================================================
# 6. 用 LangGraph 把節點接起來
# ============================================================================

graph_builder = StateGraph(AgentState)

# 加入節點
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("transcriber", transcriber_node)
graph_builder.add_node("summary", summary_node)

# 起點
graph_builder.add_edge(START, "supervisor")

# supervisor 根據 next_agent 動態路由
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "transcriber": "transcriber",
        "summary": "summary",
        "finish": END,
    }
)

# transcriber / summary 做完後回到 supervisor 重新判斷
graph_builder.add_edge("transcriber", "supervisor")
graph_builder.add_edge("summary", "supervisor")

# 編譯
agent_app = graph_builder.compile()

print("✅ LangGraph 會議 Agent 編譯完成。")


# ============================================================================
# 7. CLI 入口
# ============================================================================

def run_meeting_agent(video_path: str, description: str = ""):
    """
    執行會議記錄 Agent。

    Args:
        video_path: 影片檔案路徑
        description: 使用者對這個會議的任何補充描述
    """
    print("=" * 60)
    print("🤖 Meeting Multi-Agent System")
    print("=" * 60)

    init_state: AgentState = {
        "user_query": description or f"請處理這個會議錄影：{video_path}",
        "video_path": video_path,
        "next_agent": "",
        "transcript": "",
        "meeting_json": "",
        "output_files": [],
        "final_answer": "",
        "route_history": [],
    }

    result = agent_app.invoke(
        init_state,
        config={"recursion_limit": 10}
    )

    print("\n" + "=" * 60)
    print("📊 路由歷史：")
    for i, step in enumerate(result["route_history"], 1):
        print(f"  {i}. {step}")

    print("\n" + "=" * 60)
    print("📋 最終結果：")
    print(result.get("final_answer", "沒有產生最終答案。"))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meeting Multi-Agent System")
    parser.add_argument("video", help="影片檔案路徑")
    parser.add_argument("--desc", "-d", default="", help="對會議的補充描述（例如：MIS 週會）")
    args = parser.parse_args()

    run_meeting_agent(args.video, args.desc)
