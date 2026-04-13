---
name: meeting_summary
description: >
  Use this skill when the task involves generating structured meeting minutes
  from a meeting transcript (逐字稿). This skill provides:
  (1) a step-by-step extraction workflow for any AI agent to follow,
  (2) a standard JSON schema (references/schema.json) that the extracted data must conform to,
  (3) a ready-to-run script (scripts/generate_docs.py) that converts the JSON into a
  polished Word (.docx) and Markdown (.md) document matching the MIS Weekly Meeting template.
  Trigger when the user says things like: "幫我整理這次 Zoom 會議的記錄", "把逐字稿轉成會議記錄",
  "generate meeting minutes", or provides a transcript file and asks for a formatted document.
---

# Meeting Summary Skill

This skill guides any AI agent through the process of converting a raw meeting transcript
into a finished meeting record document, without depending on a specific AI provider.

---

## Workflow Overview

```
逐字稿 (transcript)
      │
      ▼
Step 1: Entity Identification
      │
      ▼
Step 2: Per-Speaker Segmentation & Summarisation
      │
      ▼
Step 3: Action Item Extraction
      │
      ▼
Step 4: Output structured JSON (references/schema.json)
      │
      ▼
Step 5: Run generate_docs.py → .docx + .md
```

---

## Step 1: Entity Identification

Before summarising content, identify the following from the transcript:

| Field | What to look for |
|---|---|
| `meeting_info.topic` | Title announced at the start, or inferred from context (e.g. "MIS週會") |
| `meeting_info.room` | Platform or room name (e.g. "Zoom", "Teams") |
| `meeting_info.datetime` | Date and time mentioned or from the file metadata. Format: `YYYY/MM/DD HH:MM` |
| `meeting_info.recorder` | Person taking notes, often stated at the end or beginning |
| `attendees.expected` | Pre-announced attendee list (if available; otherwise omit or infer from roster) |
| `attendees.present` | Anyone who actually spoke or is acknowledged as present |

**Tip:** If the transcript does not explicitly state certain fields, use reasonable inference
from context. Mark truly unknown fields as empty strings `""` rather than fabricating values.

---

## Step 2: Per-Speaker Segmentation & Summarisation

### Identification
- Look for patterns like `"XXX："`, `"XXX 說"`, `"XXX:"`, or speaker labels in the transcript.
- If the transcript is a continuous block without labels, perform speaker diarisation based on pronouns, topic switches, and named references.

### Summarisation Rules
For each identified speaker, produce 1–5 bullet points under `discussion[].points`:

1. **Include**: concrete updates, completed tasks, issues raised, blockers.
2. **Exclude**: filler words, greetings, off-topic small talk, repetitions.
3. **Format**: Each point is a single concise sentence in the **same language as the original transcript** (preserve Traditional Chinese if the meeting was in Chinese).
4. **Empty reports**: If a speaker was present but said nothing substantive, set `points` to `[]`.
5. **Order**: Preserve the order speakers appeared in the meeting.

---

## Step 3: Action Item Extraction

Scan the entire transcript for commitments, follow-ups, and decisions. Signals include:
- "需要…", "請…負責", "下次前要完成", "追蹤", "待確認"
- Explicit deadlines or assignments made during discussion

For each action item, populate all five fields in `action_items[]`:
- `item`: Short label (e.g. "1.", "系統升級")
- `description`: What needs to be done
- `owner`: Responsible person's name
- `status`: Default to `"待確認"` if not stated
- `due_date`: In `YYYY/MM/DD` format, or `""` if not mentioned

Also capture:
- `announcements[]`: Any general info broadcast to the group (月會日期、繳報時間 etc.)
- `facilitator_note`: Final line credits like `"投影: David; 記錄：Gior"`

---

## Step 4: Output JSON

After completing Steps 1–3, produce a single JSON object that fully conforms to
`references/schema.json`. Validate that:
- All required fields are present.
- `attendees.present` is a subset of or equal to `attendees.expected` (when expected is known).
- Each `discussion` entry has a `speaker` string and a `points` array.
- Each `action_items` entry has all five required fields.

Save this JSON to a file (e.g. `meeting_data.json`).

---

## Step 5: Generate Documents

Run the generation script with the JSON file as input:

```bash
# Install dependency (first time only)
pip install python-docx

# Generate both .docx and .md
python skills/meeting_summary/scripts/generate_docs.py \
    --input meeting_data.json \
    --dir output/meetings/
```

The script will produce:
- `output/meetings/<stem>.docx` — Word file matching the MIS template layout
- `output/meetings/<stem>.md`  — Clean Markdown version

The output stem defaults to `會議記錄_YYYYMMDD_<topic>` based on the JSON's `datetime` field.
You can override it with `--output <custom_stem>`.

---

## Reference Files

- **`references/schema.json`** — Full JSON schema with field descriptions. Read this to understand every field and its constraints before generating the JSON.
- **`assets/template.docx`** — Original MIS meeting template. Reference it when verifying visual output, or as context for understanding expected formatting.
- **`scripts/generate_docs.py`** — Document generator. Do not modify unless the template structure changes.
