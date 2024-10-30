import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "포토폴리오 주제가 뭔가요?",
    "모델은 어떤 것이 인가요",
    "프로젝트 인원은 어떻게 되나요?",
    "프로젝트 기간은 어떻게 되나요?",
    "데이터는 무엇을 이용했나요?",
    "프로젝트 하는데 어려움은 없었나요?",
]

answers = [
    "NCF 추천 알고리즘을 활용해서 영화추천 웹페이지를 만드는 것 입니다.",
    "협업필터링 알고리즘을 응용하여 딥러닝을 기술을 추가한 NCF 모델입니다",
    "3명 입니다.",
    "기간은 약 3주 정도입니다.",
    "직접 영화 데이터를 플랫폼에서 수집하여 포스터 이미지를 크롤링하였습니다.",
    "웹홈페이지와 DB연동을 하는 작업이 힘들었습니다. 하지만 DB 설계 흐름을 이해할 수 있었기에 좋은 경험이였다고 생각합니다.",
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("포토폴리오 챗본")

# 이미지 추가 
st.image("추천알고리즘.png", caption="Welcome to the Restaurant Chatbot", use_column_width=True)

st.write("최종 프로젝트 관하여 질문해보세요. 예) 프로젝트 주제는 무엇인가요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
