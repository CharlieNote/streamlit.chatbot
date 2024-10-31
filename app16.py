import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("비디오 사물 검출 앱")

# 모델 파일 업로드
model_file = st.file_uploader("모델 파일을 업로드하세요", type=["pt"])
if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
        temp_model_file.write(model_file.read())
        model_path = temp_model_file.name
    model = YOLO(model_path)
    st.success("모델이 성공적으로 로드되었습니다.")

# 비디오 파일 업로드
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# 전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("원본 영상")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:
        st.header("사물 검출 결과 영상")
        result_placeholder = st.empty()
        if "processed_video" in st.session_state and st.session_state["processed_video"] is not None:
            result_placeholder.video(st.session_state["processed_video"])
        else:
            result_placeholder.markdown(
                """
                <div style='width:100%; height:620px; background-color:#d3d3d3; display:flex; align-items:center; justify-content:center; border-radius:5px;'>
                    <p style='color:#888;'>여기에 사물 검출 결과가 표시됩니다.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 버튼 스타일 설정
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4d4d4d;
        color: #ffffff;
        font-weight: bold;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 사물 검출 버튼 클릭 이벤트 처리
if st.button("사물 검출 실행") and uploaded_file and model_file: # 버튼을 눌렀고 영상도 업로드했다
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name # 임시 비디오 파일을 생성하고 output_path 에 저장해라

    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name # 또 다른 임시 파일을 생성하여 업로도된 비디오를 temp_input_path에 저장해라

    cap = cv2.VideoCapture(temp_input_path) # 원본 비디오를 cap 을 생성해서 읽을 준비를 합니다.
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # XVID 코덱을 사용합니다.
    fps = cap.get(cv2.CAP_PROP_FPS) # 원본 비디오의 속도를 가져옵니다. 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 원본 비디오 해상도 넓이
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 원본 비디오 해상도 높이
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) 
    # Yolo 모델 결과를 기록할 비디오 파일을 준비합니다.

    frame_count = 0 # 프레임수를 기록하기 위한 변수 생성
    while cap.isOpened(): # 비디오가 끝날 때 까지 
        ret, frame = cap.read() # 프레임을 하나씩 읽어옵니다.
        if not ret: # 더 이상 읽으 프레임이 없으면 
            break # 종료해라 

        # YOLO 모델로 예측 수행 및 디버깅
        results = model(frame) # 모델에 frame 을 넣어서 객체를 검출합니다.
        detections = results[0].boxes if len(results) > 0 else []
        # 검출된 객체가 있으면 detections 에 정보가 들어가고 없으면, 빈 리스트를 반환
        
        if len(detections) > 0: # 만약 dectections 의 값이 있다면,
            for box in detections: # 바스 바운딩을 수행합니다.
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # 박스 바운딩 4개의 좌표
                confidence = box.conf[0] # 바운딩에 해당 물체가 맞을 확률
                class_id = int(box.cls[0]) # 클래스 번호
                class_name = model.names[class_id] # 클래스 이름(선수 이름)
                label = f"{class_name} {confidence:.2f}" # 그 선수일 확률

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 사각형을 그립니다.
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 검출 결과가 없을 때, 로그 출력
            st.write(f"Frame {frame_count}: No detections")

        out.write(frame) #out 비디오 파일에 저장합니다. 
        frame_count += 1 # 프레임스루르 증가 시킵니다.

    cap.release()
    out.release()

    # 결과 비디오를 st.session_state에 저장하여 스트림릿에 표시
    st.session_state["processed_video"] = output_path
    result_placeholder.video(output_path) # 회식 박스 부분에 비디오 플레이이
    st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")
