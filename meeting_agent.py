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

# ── FFmpeg utils ──────────────────────────────────────────────────────────────
def _get_ffmpeg_cmd():
    """動態尋找 ffmpeg，若無系統版則使用 imageio_ffmpeg 內建版。"""
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "找不到 ffmpeg！請確認已安裝系統版，或者執行 'pip install imageio-ffmpeg'\n"
            "  本機：brew install ffmpeg / apt install ffmpeg\n"
            "  Render：在 Build Command 加上 apt-get install -y ffmpeg"
        )

FFMPEG_CMD = _get_ffmpeg_cmd()


# ============================================================================
# 1. 初始化模型
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    print("❌ 錯誤：請在 .env 中設定 GEMINI_API_KEY")
    sys.exit(1)

# 1. 核心模型 (Supervisor / Fallback)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)
print("✅ Core LLM: Gemini 2.5 Flash")

# 2. 摘要模型 (優先使用 Groq Llama)
if GROQ_API_KEY:
    try:
        summary_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0,
        )
        print("✅ Primary Summary LLM: Groq Llama 3.3 70B")
    except Exception as e:
        summary_llm = None
        print(f"⚠️  Groq 初始化失敗，將改用 Gemini: {e}")
else:
    summary_llm = None
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
    1. 從影片提取音訊 (FFmpeg，記憶體用量 < 20 MB)
    2. 若音訊超過 20MB，自動切分為每 10 分鐘一段
    3. 依序呼叫 Groq Whisper API 取得各段逐字稿
    4. 校正各段時間軸偏移後，合併為一份完整逐字稿
    5. 將逐字稿寫入 state
    """
    CHUNK_SECONDS = 600       # 每段 10 分鐘
    MAX_FILE_SIZE_MB = 20     # 超過此大小才啟用切分

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

    print(f"  📢 正在提取音訊（FFmpeg）...")
    try:
        # ── 用 imageio_ffmpeg 取得影片總秒數（迴避 ffprobe 缺失問題）──
        import imageio_ffmpeg
        # count_frames_and_secs 回傳 (nframes, total_seconds)
        duration = float(imageio_ffmpeg.count_frames_and_secs(str(video_path))[1])

        # ── 用 ffmpeg 串流提取音訊，全程記憶體用量 < 20 MB ──
        subprocess.run(
            [
                FFMPEG_CMD, "-y",
                "-i", str(video_path),
                "-vn",                      # 不要影像
                "-acodec", "libmp3lame",    # 輸出 MP3
                "-q:a", "4",                # 品質 4（約 128 kbps，夠 Whisper 用）
                "-ar", "16000",             # 16 kHz 取樣（Whisper 最佳輸入）
                "-ac", "1",                 # 單聲道（進一步縮小檔案）
                str(audio_path),
            ],
            capture_output=True, check=True,
        )
        print(f"  ✅ 音訊已提取至 {audio_path.name}")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode(errors="replace") if e.stderr else str(e)
        history = state.get("route_history", []) + [
            f"transcriber：❌ 音訊提取失敗 {err_msg[:200]}"
        ]
        return {
            **state,
            "transcript": "",
            "final_answer": f"錯誤：音訊提取失敗 - {err_msg[:200]}",
            "route_history": history,
        }
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

    # 4b. 檢查音訊檔大小，決定是否需要切分
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"  📊 音訊大小：{file_size_mb:.1f} MB（影片長度：{int(duration // 60)} 分 {int(duration % 60)} 秒）")

    # 4c. 檢查 API Key
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

    # 4d. 準備音訊檔案清單（若需要切分）
    chunk_files = []  # list of (audio_file_path, time_offset_seconds)

    if file_size_mb <= MAX_FILE_SIZE_MB:
        # 小檔案：不切分，直接用原始音訊
        chunk_files.append((audio_path, 0))
        print(f"  📦 音訊大小未超過 {MAX_FILE_SIZE_MB}MB，無需切分")
    else:
        # 大檔案：使用 FFmpeg 切分為每 10 分鐘一段
        total_chunks = int(duration // CHUNK_SECONDS) + (1 if duration % CHUNK_SECONDS > 0 else 0)
        print(f"  ✂️  音訊超過 {MAX_FILE_SIZE_MB}MB，將切分為 {total_chunks} 段（每段 {CHUNK_SECONDS // 60} 分鐘）")

        try:
            for i, start_time in enumerate(range(0, int(duration), CHUNK_SECONDS)):
                end_time = min(start_time + CHUNK_SECONDS, duration)
                chunk_path = temp_dir / f"chunk_{i:03d}.mp3"

                # ── FFmpeg 直接從原始影片切出該區間的音訊 ──
                # -ss / -t 在輸入端做 seek，速度快且記憶體用量固定 < 10 MB
                subprocess.run(
                    [
                        FFMPEG_CMD, "-y",
                        "-ss", str(start_time),         # seek 到起始秒
                        "-t",  str(end_time - start_time),  # 擷取長度
                        "-i",  str(video_path),
                        "-vn",
                        "-acodec", "libmp3lame",
                        "-q:a", "4",
                        "-ar", "16000",
                        "-ac", "1",
                        str(chunk_path),
                    ],
                    capture_output=True, check=True,
                )

                chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
                print(f"    📎 片段 {i + 1}/{total_chunks}：{start_time // 60:.0f}m ~ {end_time // 60:.0f}m（{chunk_size_mb:.1f} MB）")
                chunk_files.append((chunk_path, start_time))

        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode(errors="replace") if e.stderr else str(e)
            history = state.get("route_history", []) + [
                f"transcriber：❌ 音訊切分失敗 {err_msg[:200]}"
            ]
            return {
                **state,
                "transcript": "",
                "final_answer": f"錯誤：音訊切分失敗 - {err_msg[:200]}",
                "route_history": history,
            }

        # 刪除原始的完整音訊檔（已切分完畢）
        if audio_path.exists():
            audio_path.unlink()

    # 4e. 依序將每段音訊送給 Groq Whisper API，校正時間軸後合併
    print(f"  🎙️  正在呼叫 Groq Whisper API (whisper-large-v3-turbo)...")
    try:
        # 支援多把 API Key 自動輪替
        keys_env = os.getenv("GROQ_API_KEYS", "")
        api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        if not api_keys and GROQ_API_KEY:
            api_keys = [GROQ_API_KEY]
            
        if not api_keys:
            raise ValueError("未設定任何 Groq API Key")
            
        current_key_idx = 0
        groq_client = Groq(api_key=api_keys[current_key_idx])
        
        all_segments = []      # 合併所有片段的 segments（時間軸已校正）
        detected_language = ""

        for chunk_idx, (chunk_path, time_offset) in enumerate(chunk_files):
            if len(chunk_files) > 1:
                print(f"    🔄 正在轉錄片段 {chunk_idx + 1}/{len(chunk_files)}...")

            while True:
                try:
                    with open(chunk_path, "rb") as f:
                        response = groq_client.audio.transcriptions.create(
                            model="whisper-large-v3-turbo",
                            file=(chunk_path.name, f.read()),
                            response_format="verbose_json",
                            prompt=WHISPER_PROMPT_HINT if WHISPER_PROMPT_HINT else None,
                        )
                    success = True
                    break # 成功，跳出 retry 迴圈
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                        # 還有下一把可以換？
                        current_key_idx = (current_key_idx + 1) % len(api_keys)
                        if current_key_idx != 0:
                            print(f"    ⚠️  第 {current_key_idx + 1} 把 API Key 遇到額度限制，切換至下一把...")
                            groq_client = Groq(api_key=api_keys[current_key_idx])
                            import time
                            time.sleep(2)
                            continue
                        else:
                            # 已經輪過一圈（代表所有鑰匙都滿了，或者都屬同一個號）
                            import re, time
                            wait_time = 120 # 預設等 2 分鐘
                            m = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_str)
                            if m:
                                mins = int(m.group(1)) if m.group(1) else 0
                                secs = float(m.group(2))
                                wait_time = int(mins * 60 + secs) + 15 # 多等 15 秒保險
                            
                            print(f"    ⏳ API 額度用盡！智能排隊：自動等待 {wait_time} 秒後恢復轉錄 (不中斷長影片)...")
                            time.sleep(wait_time)
                            continue # 等完繼續挑戰目前的 chunk
                    else:
                        print(f"    ❌ 未知嚴重錯誤：{e}")
                        break # 跳出 while，直接當作失敗
                        
            if not success:
                print(f"    ⚠️ 轉錄在本段遇到不可恢復錯誤。將保留已轉錄好的 {len(all_segments)} 段內容並進入下一步。")
                break # 放棄後續的 chunks，但不要當機，讓系統產生部分紀錄！

            # 記錄語言（取第一段的偵測結果）
            if not detected_language:
                detected_language = response.language

            # 校正時間軸偏移：每個 segment 的 start/end 加上該段的起始秒數
            for seg in response.segments:
                seg["start"] += time_offset
                seg["end"] += time_offset
                all_segments.append(seg)

            if len(chunk_files) > 1:
                print(f"    ✅ 片段 {chunk_idx + 1} 完成（{len(response.segments)} 段）")

            # 轉錄完畢，刪除該切片暫存檔
            if chunk_path.exists():
                chunk_path.unlink()

        segments = all_segments
        print(f"  📊 全部片段合併完成：共 {len(segments)} 段")

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
        md_lines.append(f"| 語言 | {detected_language} |")
        md_lines.append(f"| 切分片段 | {len(chunk_files)} 段 |")
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
        print(f"  ✅ 轉錄完成！語言：{detected_language}，共 {len(segments)} 段，約 {word_count} 詞")
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
        # 清理暫存目錄中剩餘的檔案
        import shutil as _shutil
        if temp_dir.exists():
            _shutil.rmtree(temp_dir, ignore_errors=True)

    history = state.get("route_history", []) + [
        f"transcriber：✅ 完成轉錄（{len(chunk_files)} 段切分，{len(segments)} 段文字，{unique_speakers} 位說話者）"
    ]

    return {
        **state,
        "transcript": transcript_text,
        "output_files": state.get("output_files", []) + [str(raw_md_path)],
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

    # 5a. 嘗試使用主要摘要模型 (Groq)
    meeting_data = None
    applied_llm_name = "Groq Llama 3.3"
    
    if summary_llm:
        print(f"  🤖 優先嘗試使用 Groq Llama 3.3 提取會議結構化資料...")
        try:
            result = summary_llm.invoke([
                ("system", SUMMARY_SYSTEM),
                ("human", f"使用者描述：{user_query}\n\n以下是會議的逐字稿：\n{transcript}\n\n請從中提取結構化的會議記錄 JSON。")
            ])
            raw_json = result.content.strip()
            # 清理 JSON
            if "```json" in raw_json:
                raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_json:
                raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
            meeting_data = json.loads(raw_json)
            print(f"  ✅ Groq 摘要成功！")
        except Exception as e:
            print(f"  ⚠️  Groq 摘要失敗（{type(e).__name__}），準備切換至 Gemini 代理執行...")
            applied_llm_name = "Gemini 2.5 Flash (Fallback)"
    else:
        applied_llm_name = "Gemini 2.5 Flash (Default)"

    # Fallback to Gemini if Groq failed or not available
    if meeting_data is None:
        print(f"  🚀 正在使用 Gemini 2.5 Flash 提取會議結構化資料...")
        try:
            result = llm.invoke([
                ("system", SUMMARY_SYSTEM),
                ("human", f"使用者描述：{user_query}\n\n以下是會議的逐字稿：\n{transcript}\n\n請從中提取結構化的會議記錄 JSON。")
            ])
            raw_json = result.content.strip()
            if "```json" in raw_json:
                raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_json:
                raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
            meeting_data = json.loads(raw_json)
            print(f"  ✅ Gemini 摘要成功！")
        except Exception as e:
            print(f"  ❌ 所有摘要模型均失敗：{e}")
            history = state.get("route_history", []) + [f"summary：❌ 摘要失敗（{applied_llm_name}）"]
            return {
                **state,
                "final_answer": f"錯誤：摘要提取失敗 - {e}",
                "route_history": history,
            }

    # 5a 結尾：記錄成功 log
    print(f"     - 使用模型：{applied_llm_name}")
    print(f"     - 討論者：{len(meeting_data.get('discussion', []))} 人")
    print(f"     - 待辦事項：{len(meeting_data.get('action_items', []))} 項")

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
    new_output_files = [str(f) for f in output_dir.glob(f"{stem}.*")]
    all_output_files = state.get("output_files", []) + new_output_files
    print(f"  ✅ 文件生成完成！")
    for f in new_output_files:
        print(f"     📁 {f}")

    # 組裝最終回覆
    final_answer = f"""✅ 會議記錄已完成！

📋 會議主題：{meeting_data.get('meeting_info', {}).get('topic', 'N/A')}
📅 日期時間：{meeting_data.get('meeting_info', {}).get('datetime', 'N/A')}
👥 與會人員：{', '.join(meeting_data.get('attendees', {}).get('present', []))}
📝 討論者：{len(meeting_data.get('discussion', []))} 人
📌 待辦事項：{len(meeting_data.get('action_items', []))} 項

📁 產出檔案（已打包）：
"""
    for f in all_output_files:
        final_answer += f"  - {Path(f).name}\n"

    history = state.get("route_history", []) + [
        f"summary：✅ 完成結構化提取與文件生成（使用 {applied_llm_name}，共 {len(new_output_files)} 個檔案）"
    ]

    return {
        **state,
        "meeting_json": json.dumps(meeting_data, ensure_ascii=False),
        "output_files": all_output_files,
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
