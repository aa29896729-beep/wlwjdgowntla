import streamlit as st
from kiwipiepy import Kiwi
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import matplotlib.font_manager as fm

# 한글 폰트 설정 (NanumGothic.ttf 사용)
FONT_PATH = "NanumGothic.ttf"
plt.rcParams['axes.unicode_minus'] = False


@st.cache_resource
def load_kiwi():
    """Kiwi 형태소 분석기를 초기화하고 캐싱합니다."""
    return Kiwi()


def apply_user_dictionary(kiwi, user_vocab_str):
    """
    사용자 입력창에 쉼표로 구분된 단어들을 Kiwi 사전에 고유명사(NNP)로 등록합니다.
    """
    if user_vocab_str:
        words = [w.strip() for w in user_vocab_str.split(',') if w.strip()]
        for word in words:
            try:
                kiwi.add_user_word(word, 'NNP')
            except Exception:
                pass
    return kiwi


def preprocess_text(text, kiwi):
    """
    텍스트를 형태소 분석하여 2글자 이상의 명사만 추출합니다.
    """
    tokens = kiwi.tokenize(text)
    nouns = [token.form for token in tokens if token.tag.startswith('NN') and len(token.form) >= 2]
    return nouns


def get_word_frequencies(nouns):
    """
    추출된 명사 리스트의 빈도수를 계산하여 pandas DataFrame으로 반환합니다.
    """
    counter = Counter(nouns)
    freq_df = pd.DataFrame(counter.most_common(), columns=['단어', '빈도'])
    return freq_df


# 감성 분석을 위한 샘플 사전 (각 10개 이상)
POSITIVE_WORDS = {'좋다', '행복하다', '기쁘다', '훌륭하다', '최고다', '만족하다', '감사하다', '유익하다', '뛰어나다', '사랑하다', '성공하다', '편하다'}
NEGATIVE_WORDS = {'나쁘다', '슬프다', '화나다', '최악이다', '불만이다', '실망하다', '어렵다', '부족하다', '힘들다', '괴롭다', '실패하다', '불편하다'}


def analyze_sentiment(text, kiwi):
    """
    형태소 분석을 통해 동사(VV), 형용사(VA)의 경우 기본형(~다)으로 변환하고,
    긍정/부정 사전에 비교하여 개수와 일치하는 단어 목록을 반환합니다.
    """
    tokens = kiwi.tokenize(text)
    pos_count = 0
    neg_count = 0
    found_pos = []
    found_neg = []

    for token in tokens:
        if token.tag.startswith('VV') or token.tag.startswith('VA'):
            base_form = token.form + '다'
            if base_form in POSITIVE_WORDS:
                pos_count += 1
                found_pos.append(base_form)
            elif base_form in NEGATIVE_WORDS:
                neg_count += 1
                found_neg.append(base_form)
        elif token.form in POSITIVE_WORDS:
            pos_count += 1
            found_pos.append(token.form)
        elif token.form in NEGATIVE_WORDS:
            neg_count += 1
            found_neg.append(token.form)

    return pos_count, neg_count, list(set(found_pos)), list(set(found_neg))


def plot_bar_chart(freq_df, top_n):
    """
    상위 N개 단어를 가로 막대그래프로 시각화합니다 (NanumGothic 폰트 적용).
    """
    df_plot = freq_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(df_plot['단어'], df_plot['빈도'], color='skyblue')
    ax.set_xlabel('빈도 수')
    ax.set_ylabel('단어')
    ax.set_title(f'상위 {top_n}개 단어 빈도')

    font_prop = fm.FontProperties(fname=FONT_PATH, size=12)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)

    st.pyplot(fig)


def plot_word_cloud(nouns):
    """
    추출된 명사를 바탕으로 워드클라우드를 생성하여 시각화합니다 (NanumGothic 폰트 적용).
    """
    text_data = " ".join(nouns)
    wc = WordCloud(
        font_path=FONT_PATH,
        width=800,
        height=400,
        background_color='white'
    ).generate(text_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


def plot_sentiment_pie(pos_count, neg_count):
    """
    긍정/부정 감성 분석 결과를 원그래프(Pie Chart)로 시각화합니다.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ['긍정', '부정']
    counts = [pos_count, neg_count]
    colors = ['#ff9999', '#66b3ff']

    if sum(counts) == 0:
        ax.text(0.5, 0.5, '분석된 감성 단어가 없습니다.', horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax.axis('off')
    else:
        ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
        ax.axis('equal')
        ax.set_title('긍정 / 부정 감성 비율')

    st.pyplot(fig)


def main():
    st.title("📊 단어 빈도 및 감성 분석 프로그램")

    # Kiwi 객체 로드
    kiwi = load_kiwi()

    # 1. 텍스트 파일 업로드
    st.header("1. 텍스트 파일 업로드")
    uploaded_file = st.file_uploader("분석할 텍스트 파일(.txt)을 업로드하세요.", type=["txt"])

    # [추가 기능] 사용자 사전 설정 입력창
    st.subheader("📌 사용자 사전 설정")
    user_vocab_input = st.text_input(
        "쪼개지지 않게 등록할 단어들을 쉼표(,)로 구분하여 입력하세요.",
        placeholder="예: 인공지능, 워드클라우드, 자연어처리"
    )
    kiwi = apply_user_dictionary(kiwi, user_vocab_input)

    if uploaded_file is not None:
        # 파일 내용을 문자열로 디코딩
        file_bytes = uploaded_file.read()
        text = file_bytes.decode('utf-8', errors='ignore')

        # 2. 파일 내용 보기
        st.header("2. 파일 내용 보기")
        with st.expander("업로드된 텍스트 미리보기", expanded=False):
            st.text_area("내용", text, height=150, disabled=True)

        # 3. 전처리 결과
        st.header("3. 전처리 결과 (2글자 이상 명사 추출)")
        nouns = preprocess_text(text, kiwi)
        st.write(f"추출된 총 명사 개수: **{len(nouns)}개**")
        st.write("추출된 명사 샘플:", nouns[:30])

        # 4. 단어 빈도
        st.header("4. 단어 빈도")
        freq_df = get_word_frequencies(nouns)

        if not freq_df.empty:
            max_slider = min(50, len(freq_df))
            default_val = min(10, max_slider)
            top_n = st.slider("표시할 상위 단어 개수 선택", min_value=5, max_value=max_slider, value=default_val)

            st.subheader(f"상위 {top_n}개 단어 빈도표")
            st.dataframe(freq_df.head(top_n), use_container_width=True)

            # 5. 시각화 (탭 2개)
            st.header("5. 시각화")
            tab1, tab2 = st.tabs(["가로 막대그래프", "워드클라우드"])

            with tab1:
                st.subheader("상위 단어 가로 막대그래프")
                plot_bar_chart(freq_df, top_n)

            with tab2:
                st.subheader("워드클라우드")
                plot_word_cloud(nouns)

            # [추가 기능] 6. 감성 분석
            st.header("6. 감성 분석")
            pos_count, neg_count, found_pos, found_neg = analyze_sentiment(text, kiwi)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="긍정 단어 총 개수", value=f"{pos_count}개")
                st.write("**감지된 긍정 단어 목록:**", found_pos if found_pos else "없음")
            with col2:
                st.metric(label="부정 단어 총 개수", value=f"{neg_count}개")
                st.write("**감지된 부정 단어 목록:**", found_neg if found_neg else "없음")

            st.subheader("감성 비율 원그래프")
            plot_sentiment_pie(pos_count, neg_count)

        else:
            st.warning("전처리 결과 추출된 명사가 없습니다. 다른 텍스트 파일을 업로드해 주세요.")


if __name__ == '__main__':
    main()
