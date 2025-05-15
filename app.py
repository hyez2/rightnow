import streamlit as st
try:
    import faiss
except ImportError:
    faiss = None
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 세팅
st.set_page_config(page_title="고용노동부 법령해석 기반 RAG 챗봇", page_icon="🤖")
st.title("📚 고용노동부 법령해석 기반 RAG 챗봇")
st.caption("🔍 질문을 입력하면 관련 질의·회답을 검색해 답변해드려요.")

# 모델 및 데이터 로딩
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open("moel_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)
    index = faiss.read_index("moel_index.faiss")
    return model, sentences, index

model, sentences, index = load_model_and_data()

# 검색 함수
def retrieve_similar_sentences(query, k=3):
    vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(vec), k)
    return [sentences[i] for i in I[0]]

# 프롬프트 구성 + GPT 응답
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

def build_prompt(query, docs):
    context = "\n---\n".join(docs)
    return f"""[사용자 질문]
{query}

[참고 문서]
{context}

→ 위 문서를 참고하여 법령에 기반한 정확하고 공감 있는 답변을 제공해주세요.
"""

def detect_emotion(query: str) -> str:
    emotion_prompt = f"""
    아래 사용자 발화에서 느껴지는 주된 감정을 다음 중에서 분류해주세요: [중립, 불안, 분노, 슬픔, 좌절, 기쁨]

    발화: "{query}"

    감정:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": emotion_prompt}]
    )
    emotion = response.choices[0].message.content.strip()
    return emotion

emotion_messages = {
    "불안": "지금 느끼는 불안은 당연한 거예요. 함께 해결 방법을 찾아드릴게요.",
    "분노": "부당한 상황에 분노하는 건 당연해요. 지금부터 하나씩 정리해봐요.",
    "슬픔": "그동안 정말 힘드셨죠. 괜찮아요, 곧 나아질 거예요.",
    "좌절": "너무 지치셨겠어요. 이제 도움받을 시간이에요.",
    "기쁨": "좋은 소식이네요! 함께 기뻐요",
    "중립": "",
}


def gpt_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 메인 인터페이스
query = st.text_input("💬 질문을 입력하세요", placeholder="예: 프로젝트 종료 시점을 계약 기간으로 적어도 되나요?")

if query:
    with st.spinner("🔍 관련 문서 검색 및 답변 생성 중..."):
        docs = retrieve_similar_sentences(query)
        prompt = build_prompt(query, docs)
        answer = gpt_answer(prompt)
        emotion = detect_emotion(query)

    st.markdown("### 🤖 챗봇의 답변")
    st.success(answer)

    # 감성 케어 메시지 추가 출력
    if emotion in emotion_messages and emotion_messages[emotion]:
        st.info(f"💌 {emotion_messages[emotion]}")

    with st.expander("📄 RAG 참고 문서 보기"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**문서 {i}**\n\n{doc}")