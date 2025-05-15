# 고용노동부 법령해석 기반 RAG 챗봇

고용노동부 법령 해석 자료를 기반으로 질의응답을 제공하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 기능

- 고용노동부 법령 해석 자료 검색
- 질문에 대한 법령 기반 답변 제공
- 사용자 감정 분석 및 공감 메시지 제공

## 배포 방법

### 로컬 실행

1. 필요한 패키지 설치:
```
pip install -r requirements.txt
```

2. Streamlit 앱 실행:
```
streamlit run app.py
```

### Streamlit Cloud 배포

1. 이 저장소를 GitHub에 업로드합니다.
2. [Streamlit Cloud](https://streamlit.io/cloud)에 접속하여 로그인합니다.
3. "New app" 버튼을 클릭하고 이 저장소를 선택합니다.
4. 메인 파일로 `app.py`를 선택합니다.
5. 고급 설정에서 OpenAI API 키를 시크릿으로 추가합니다:
   - 시크릿 이름: `openai.api_key`
   - 시크릿 값: 자신의 OpenAI API 키
6. "Deploy" 버튼을 클릭합니다.

## 필요한 파일

- `app.py`: 메인 애플리케이션 코드
- `moel_sentences.json`: 법령 해석 문장 데이터
- `moel_index.faiss`: 벡터 검색을 위한 FAISS 인덱스 