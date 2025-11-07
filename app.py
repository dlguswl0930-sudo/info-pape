import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import io
import datetime

# -------------------------------
# 초기 설정
# -------------------------------
st.set_page_config(page_title="Gemini 기반 고객응대 챗봇", page_icon="🪄", layout="wide")
st.title("🪄 Gemini 기반 고객응대 챗봇")

# -------------------------------
# 사이드바 - 설정
# -------------------------------
with st.sidebar:
    st.header("설정")

    # 모델 선택
    model_name = st.selectbox(
        "모델 선택",
        ["gemini-2.0-flash", "gemini-2.0-pro"],
        index=0
    )

    # API Key 설정
    api_key = None
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
        st.success("✅ st.secrets['GEMINI_API_KEY']가 설정되어 있습니다.")
    else:
        st.warning("⚠️ st.secrets['GEMINI_API_KEY']가 설정되어 있지 않습니다.")
        api_key = st.text_input("Gemini API Key (임시 입력)", type="password")

    # CSV 저장 옵션
    save_csv = st.checkbox("대화 자동 CSV 저장")

    # 세션 ID 표시
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(hex(int(time.time())))[2:10]
    st.text(f"세션 ID: {st.session_state.session_id}")

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()  # ✅ 최신 Streamlit용 함수로 수정됨

# -------------------------------
# API 설정
# -------------------------------
if not api_key:
    st.error("Gemini API 키가 필요합니다.")
    st.stop()

genai.configure(api_key=api_key)

# -------------------------------
# 시스템 프롬프트 설정
# -------------------------------
system_prompt = (
    "당신은 쇼핑몰 고객센터의 AI 상담원입니다.\n"
    "1) 사용자는 쇼핑몰 구매 과정에서 겪은 불편/불만을 언급합니다. 정중하고 공감 어린 말투로 응답하세요.\n"
    "2) 사용자의 불편 사항을 구체적으로 정리하여(무엇이/언제/어디서/어떻게) 수집하고, "
    "이를 고객 응대 담당자에게 전달한다는 취지로 안내하세요.\n"
    "3) 마지막에는 담당자 확인 후 회신을 위해 이메일 주소를 요청하세요. "
    "만일 사용자가 연락 제공을 원치 않으면 ‘죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요.’라고 정중히 안내하세요."
)

# -------------------------------
# 대화 히스토리 초기화
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# 대화창 UI
# -------------------------------
st.subheader("💬 대화창")

for msg in st.session_state.messages:
    role = "🧑‍💼 고객" if msg["role"] == "user" else "🤖 챗봇"
    st.markdown(f"**{role}:** {msg['content']}")

user_input = st.text_area("메시지 입력", key="user_input", height=100)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    send_btn = st.button("전송")
with col2:
    download_btn = st.button("로그 다운로드")
with col3:
    clear_btn = st.button("전체 초기화")

# -------------------------------
# 전체 초기화
# -------------------------------
if clear_btn:
    st.session_state.messages = []
    st.rerun()  # ✅ 최신 함수 사용

# -------------------------------
# 로그 다운로드
# -------------------------------
if download_btn:
    if len(st.session_state.messages) == 0:
        st.warning("다운로드할 대화 내용이 없습니다.")
    else:
        df = pd.DataFrame(st.session_state.messages)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        st.download_button(
            label="💾 CSV 다운로드",
            data=csv_buffer.getvalue(),
            file_name=f"chat_log_{st.session_state.session_id}.csv",
            mime="text/csv"
        )

# -------------------------------
# Gemini 응답 처리
# -------------------------------
if send_btn and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.write("⌛ 응답 생성 중...")

    try:
        model = genai.GenerativeModel(model_name)
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]]
        )

        prompt = f"{system_prompt}\n\n대화 이력:\n{history_text}\n\n고객의 최신 메시지: {user_input}"
        response = model.generate_content(prompt)

        answer = response.text
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()  # ✅ 최신 rerun 사용

    except Exception as e:
        st.error(f"오류 발생: {e}")

# -------------------------------
# CSV 자동 저장
# -------------------------------
if save_csv and len(st.session_state.messages) > 0:
    df = pd.DataFrame(st.session_state.messages)
    df.to_csv(
        f"chat_log_{st.session_state.session_id}.csv",
        index=False,
        encoding="utf-8-sig"
    )

# -------------------------------
# 푸터
# -------------------------------
st.markdown("---")
st.caption("© Gemini 고객응대 챗봇 | Google Gemini API | Streamlit 기반")
