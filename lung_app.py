import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import platform

# 1. 한글 폰트 설정
def set_korean_font():
    system_name = platform.system()
    if system_name == "Windows":
        plt.rc('font', family='Malgun Gothic')
    elif system_name == "Darwin":
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False)

set_korean_font()

# --- 웹 앱 다크 모드 및 스타일 커스텀 (CSS) ---
st.markdown("""
    <style>
    /* 전체 배경색을 깊은 밤하늘색으로 */
    .stApp {
        background-color: #0A192F;
        color: #E2E8F0;
    }
    /* 제목과 텍스트를 별빛처럼 반짝이게 */
    h1, h2, h3 {
        color: #F6AD55 !important; /* 금색 */
        text-shadow: 0 0 10px #F6AD55;
    }
    .stSlider > div > div > div > div {
        background-color: #F6AD55 !important;
    }
    .stMarkdown p {
        font-size: 1.1rem;
    }
    /* 버튼 스타일 */
    div.stButton > button:first-child {
        background-color: #F6AD55;
        color: #0A192F;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        box-shadow: 0 0 15px #F6AD55;
    }
    </style>
    """, unsafe_allow_html=True)

# 페이지 제목 설정
st.set_page_config(page_title="건강 별자리 분석", layout="centered")

@st.cache_resource
def load_resources():
    base_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_path, "lung.csv"))
    model = joblib.load(os.path.join(base_path, "lung.pkl"))
    scaler = joblib.load(os.path.join(base_path, "lung_scaler.pkl"))
    return df, model, scaler

try:
    df, model, scaler = load_resources()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# --- 메인 화면 ---
st.title("🌃 나의 건강 별자리 관측소")
st.write("깊은 밤, 당신의 생활 습관이 어떤 별자리를 그리고 있는지 분석합니다.")
st.markdown("---")

# --- 세분화된 입력 단계 ---
st.header("🔭 관측 데이터 입력")

# 1. 나이 관측
age = st.number_input("🎂 당신이 지구에서 보낸 시간 (나이)", min_value=1, max_value=120, value=30)

st.write("---")

# 2. 흡연 습관 (더 세밀한 단계)
st.subheader("🚬 흡연의 궤적")
st.write("하루 평균 흡연량을 정교하게 설정해주세요.")
smokes = st.slider("흡연 수치 (0: 비흡연 ~ 50: 헤비 스모커)", 
                   min_value=0, max_value=50, value=10, step=2)

# 수치에 따른 감성적인 설명
if smokes == 0: st.caption("🍃 맑은 공기가 느껴지는 밤입니다.")
elif smokes < 10: st.caption("☁️ 옅은 구름이 조금 끼어있네요.")
elif smokes < 25: st.caption("🌫️ 안개가 자욱해 별이 잘 보이지 않아요.")
else: st.caption("🔥 거친 연기가 밤하늘을 가리고 있습니다.")

st.write("---")

# 3. 음주 습관 (더 세밀한 단계)
st.subheader("🍶 음주의 파동")
st.write("일주일간의 음주 빈도와 양을 고려해 설정해주세요.")
alkhol = st.slider("음주 수치 (0: 금주 ~ 50: 주당)", 
                   min_value=0, max_value=50, value=10, step=2)

if alkhol == 0: st.caption("🥤 고요하고 평온한 호수 같은 상태입니다.")
elif alkhol < 10: st.caption("🍺 잔잔한 물결이 일렁이고 있어요.")
elif alkhol < 25: st.caption("🍷 파도가 점점 높아지고 있습니다.")
else: st.caption("🌊 거대한 폭풍우가 몰아치고 있군요.")

st.markdown("---")

# --- 분석 실행 ---
if st.button("🌟 나의 별자리 분석 시작", use_container_width=True):
    new_patient = pd.DataFrame([[age, smokes, alkhol]], columns=['나이', '흡연', '음주'])
    
    try:
        # 데이터 변환 및 예측
        new_patient_scaled = scaler.transform(new_patient)
        cluster = model.predict(new_patient_scaled)[0]
        
        # 밤하늘 테마 결과 출력
        st.balloons()
        st.subheader(f"✨ 분석 완료: 당신은 **[{cluster}번 별자리 그룹]**에 속합니다.")
        
        # 군집 설명
        if cluster == 0:
            st.success("🌟 **오로라 그룹**: 가장 밝고 깨끗하게 빛나는 건강을 가진 별자리입니다.")
        elif cluster == 1:
            st.warning("🌙 **반달 그룹**: 관리가 조금 필요하지만 여전히 아름답게 빛날 가능성이 큽니다.")
        else:
            st.error("☄️ **유성우 그룹**: 급격한 변화가 필요한 상태입니다. 건강의 궤도를 수정해보세요.")

        st.markdown("---")

        # --- 밤하늘 시각화 (Matplotlib Dark) ---
        st.subheader("🌌 밤하늘의 성도 (Star Chart)")
        
        # 다크 배경 스타일 적용
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#0A192F') # 웹 배경과 통일
        ax.set_facecolor('#0A192F')

        # 1. 기존 데이터들 (은하수처럼 연하게)
        ax.scatter(df['흡연'], df['음주'], c=df['cluster'], alpha=0.2, cmap='YlGnBu', s=30, label='다른 별들')

        # 2. 나의 위치 (빛나는 황금성 효과)
        # 여러 겹의 후광 효과
        ax.scatter(smokes, alkhol, c='gold', s=2000, alpha=0.1, marker='o') 
        ax.scatter(smokes, alkhol, c='gold', s=1000, alpha=0.2, marker='o')
        # 중심의 빛나는 별
        ax.scatter(smokes, alkhol, c='#FFFFFF', s=400, marker='*', edgecolors='gold', linewidth=1.5, label='나의 별자리')

        # 그래프 꾸미기
        ax.set_xlabel('흡연 지수 (Smoking Track)', color='#F6AD55', fontsize=12)
        ax.set_ylabel('음주 지수 (Alcohol Wave)', color='#F6AD55', fontsize=12)
        ax.set_title('나의 건강 궤적 관측 결과', color='#F6AD55', fontsize=15, pad=20)
        
        # 테두리 및 그리드 설정
        ax.spines['bottom'].set_color('#F6AD55')
        ax.spines['left'].set_color('#F6AD55')
        ax.grid(True, linestyle=':', alpha=0.3, color='#F6AD55')
        ax.legend(facecolor='#0A192F', edgecolor='#F6AD55')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"관측 장비에 오류가 발생했습니다: {e}")

# 원본 데이터 숨기기
with st.expander("📂 관측 기록 저장소 보기"):
    st.dataframe(df.style.background_gradient(cmap='Blues'))