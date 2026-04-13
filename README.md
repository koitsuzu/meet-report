# Meeting Agent (FastAPI + LangGraph)

這是一個基於 FastAPI 與 LangGraph 打造的多代理（Multi-agent）AI 會議記錄系統。可以接受包含語音/影片的檔案，自動提取音訊並生成結構化會議記錄的 Word 與 Markdown 檔案。

## 架構說明

*   **FastAPI Backend (`server.py`)**: 接收檔案上傳，提供前端 RESTful API。
*   **多智能體協作 (`meeting_agent.py`)**: 
    *   **Supervisor Node**: 負責管控流程。
    *   **Transcriber Node**: 提取音訊並發送到 Groq Whisper API 生成精準逐字稿。
    *   **Summary Node**: 利用 Groq Llama 3 進行摘要提取，識別會議主題、與會者和追蹤事項。
*   **前端介面 (`static/index.html`)**: 美觀的互動式上傳介面。

## 安裝與執行

1. **安裝依賴套件**:
   ```bash
   pip install -r requirements.txt
   ```

2. **設定環境變數**:
   複製 `.env.example` 並命名為 `.env`，隨後填寫你的 API keys。
   ```bash
   cp .env.example .env
   ```

3. **啟動伺服器**:
   ```bash
   python server.py
   ```
   伺服器預設運行在 `http://127.0.0.1:8000`。透過瀏覽器即可開始使用介面！
