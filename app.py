"""
app.py
Streamlit 기반 Gemini API 고객 응대 챗봇
------------------------------------------------------
기능:
- 기본 모델: gemini-2.0-flash (목록에서 선택 가능, -exp 제외)
- 시스템 프롬프트: 고객 불만 응대 시 공감 & 이메일 요청
- API 키: st.secrets['GEMINI_API_KEY'] 또는 UI 입력
- 대화 히스토리, 429 재시도(최근 6턴 유지 후 재시작)
- CSV 자동 기록(옵션), 로그 다운로드, 대화 초기화, 모델/세션 표시
------------------------------------------------------
"""

import streamlit as st
import pandas as pd
import json
import time
import uuid
from datetime import datetime

# Google Gemini API
try:
    from google import genai
except ImportError:
    genai = None

# -----------------------------
# 상수 설정
# -----------------------------
DEFAULT_MODEL = "gemini-2.0-flash"
MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0",
    "gemini-1.5",
]

SYSTEM_PROMPT = (
    "1) 사용자는 쇼핑몰 구매 과정에서 겪은 불편/불만을 언급합니다. 정중하고 공감 어린 말투로 응답하세요.\n"
    "2) 사용자의 불편 사항을 구체적으로 정리하여(무엇이/언제/어디서/어떻게) 수집하고, 이를 고객 응대 담당자에게 전달한다는 취지로 안내하세요.\n"
    "3) 마지막에는 담당자 확인 후 회신을 위해 이메일 주소를 요청하세요. 만일 사용자가 연락 제공을 원치 않으면: "
    "“죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요.”라고 정중히 안내하세요."
)

# -----------------------------
# 함수 정의
# -----------------------------
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def build_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai 패키지가 설치되어 있지 않습니다.")
    return genai.Client(api_key=api_key)


def call_gemini(client, model: str, prompt: str):
    resp = client.models.generate_content(model=model, contents=prompt)
    try:
        return resp.text
    except Exception:
        return str(resp)


def trim_history(history, keep_turns=6):
    non_sys = [m for m in history if m["role"] != "system"]
    return [m for m in history if m["role"] == "system"][:1] + non_sys[-keep_turns:]


def export_csv(history):
    df = pd.DataFrame(history)
    return df.to_csv(index=False).encode("utf-8")


def export_json(history):
    return json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Gemini 고객응대 챗봇", layout="wide")
st.title("🟢 Gemini 기반 고객응대 챗봇")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("설정")

    model = st.selectbox("모델 선택", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    api_key = get_api_key()

    if not api_key:
        st.warning("st.secrets['GEMINI_API_KEY']가 설정되어 있지 않습니다.")
        api_key = st.text_input("Gemini API Key (임시 입력)", type="password")

    auto_csv = st.checkbox("대화 자동 CSV 저장", value=False)

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:8]
    st.text(f"세션 ID: {st.session_state['session_id']}")

    if st.button("대화 초기화"):
        st.session_state["history"] = [
            {"role": "system", "content": SYSTEM_PROMPT, "timestamp": datetime.utcnow().isoformat()}
        ]
        st.success("대화가 초기화되었습니다.")


with col2:
    st.subheader("대화창")

    if "history" not in st.session_state:
        st.session_state["history"] = [
            {"role": "system", "content": SYSTEM_PROMPT, "timestamp": datetime.utcnow().isoformat()}
        ]

    history = st.session_state["history"]

    for msg in history:
        role = msg["role"]
        text = msg["content"]
        if role == "user":
            st.markdown(f"**🧍 사용자:** {text}")
        elif role == "assistant":
            st.markdown(f"**🤖 챗봇:** {text}")

    user_input = st.text_area("메시지 입력", height=100)
    cols = st.columns(3)
    send_btn = cols[0].button("전송")
    download_btn = cols[1].button("로그 다운로드")
    reset_btn = cols[2].button("전체 초기화")

    if reset_btn:
        st.session_state.clear()
        st.experimental_rerun()

    if download_btn:
        csv_bytes = export_csv(history)
        json_bytes = export_json(history)
        st.download_button("CSV 다운로드", csv_bytes, "chat_log.csv", "text/csv")
        st.download_button("JSON 다운로드", json_bytes, "chat_log.json", "application/json")

    if send_btn and user_input.strip():
        st.session_state["history"].append(
            {"role": "user", "content": user_input.strip(), "timestamp": datetime.utcnow().isoformat()}
        )

        if not api_key:
            st.error("API 키가 필요합니다.")
        else:
            try:
                client = build_client(api_key)
                prompt = "\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in st.session_state["history"]])
                retries = 3
                for i in range(retries):
                    try:
                        reply = call_gemini(client, model, prompt)
                        break
                    except Exception as e:
                        if "429" in str(e):
                            st.warning("429 오류 감지 — 대화 축약 후 재시도 중...")
                            st.session_state["history"] = trim_history(st.session_state["history"], keep_turns=6)
                            time.sleep(2 ** i)
                            continue
                        else:
                            raise e
                else:
                    reply = "죄송합니다. 서버가 바쁩니다. 잠시 후 다시 시도해주세요."

                st.session_state["history"].append(
                    {"role": "assistant", "content": reply, "timestamp": datetime.utcnow().isoformat()}
                )
                st.experimental_rerun()

            except Exception as e:
                st.error(f"오류 발생: {e}")

    if auto_csv:
        csv_bytes = export_csv(st.session_state["history"])
        st.download_button(
            "CSV 자동저장 다운로드",
            csv_bytes,
            f"chat_{st.session_state['session_id']}.csv",
            "text/csv",
        )

st.write("---")
st.caption("© Gemini 고객응대 챗봇 | google-genai 기반 | Streamlit 예제")
