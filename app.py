"""SentimentScope: an interactive English sentiment analysis dashboard."""
import pandas as pd
import streamlit as st
from model import MODEL_ID, classify

st.set_page_config(page_title="SentimentScope | NLP Studio", page_icon="💬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif}
[data-testid="stAppViewContainer"] {background:linear-gradient(140deg,#f7f9fe 0%,#edf3fa 100%)}
[data-testid="stSidebar"] {background:#101d36}
[data-testid="stSidebar"] * {color:#dce8fa !important}
[data-testid="stSidebar"] hr {border-color:#2c405d}
.block-container {max-width:1300px;padding-top:2.2rem;padding-bottom:3rem}
h1,h2,h3 {color:#162341 !important;letter-spacing:-.03em}
.hero {background:radial-gradient(circle at 88% 20%,#3c64a7 0%,transparent 32%),linear-gradient(115deg,#101c35,#1b3a65);border-radius:24px;padding:38px 44px;margin-bottom:26px;box-shadow:0 15px 35px #182b4b18;position:relative;overflow:hidden}
.hero .eyebrow {font-size:.75rem;font-weight:800;letter-spacing:.17em;color:#83d7f7;text-transform:uppercase}
.hero h1 {font-size:3.15rem;color:white !important;letter-spacing:-.055em;margin:12px 0 10px;font-weight:800}
.hero p {font-size:1.05rem;color:#d4e6fb;margin:0;max-width:680px}
.hero .pill {display:inline-block;margin-top:22px;padding:7px 13px;border:1px solid #88d5f780;border-radius:99px;color:#b5eaf9;font-size:.78rem;font-weight:700}
.sidebar-brand {font-size:1.5rem;font-weight:800;letter-spacing:-.04em;margin:12px 0 8px}
.sidebar-caption {font-size:.88rem;line-height:1.7;color:#aebfd7 !important;margin-bottom:25px}
.sidebar-label {font-size:.72rem;letter-spacing:.14em;font-weight:800;color:#91b8df !important;margin:16px 0 10px}
.sidebar-step {border:1px solid #344766;border-radius:12px;padding:12px 14px;margin:9px 0;font-size:.88rem}
.section-lead {color:#61708b;font-size:.96rem;margin-bottom:18px}
.stButton>button[kind="primary"],.stDownloadButton>button {background:linear-gradient(95deg,#3868df,#138db1);border:0;border-radius:12px;color:white;font-weight:700;min-height:46px}
.stButton>button[kind="primary"]:hover {background:linear-gradient(95deg,#254fc4,#08708e);color:white}
[data-testid="stMetric"] {background:white;border:1px solid #e3eaf4;border-radius:16px;padding:18px;box-shadow:0 5px 18px #1d35570b}
[data-testid="stMetricLabel"] {color:#667792}
[data-testid="stTabs"] button {font-weight:700}
[data-testid="stFileUploader"], [data-testid="stTextArea"] {border-radius:12px}
.footer-note {color:#7c8ca3;font-size:.8rem;margin-top:26px}
</style>""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">◈ SentimentScope</div>',unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">An interactive NLP workspace for exploring the tone of English text.</div>',unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-label">WORKFLOW</div>',unsafe_allow_html=True)
    st.markdown('<div class="sidebar-step">01 &nbsp; Enter a review or upload a CSV</div>',unsafe_allow_html=True)
    st.markdown('<div class="sidebar-step">02 &nbsp; Run sentiment classification</div>',unsafe_allow_html=True)
    st.markdown('<div class="sidebar-step">03 &nbsp; Explore scores and export data</div>',unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-label">CLASSIFICATION</div>',unsafe_allow_html=True)
    st.write('🟢 Positive  ·  🟡 Neutral  ·  🔴 Negative')
    st.caption('Model scores are estimates, not guarantees of correctness. English-language inputs work best.')

st.markdown('<div class="hero"><div class="eyebrow">Natural Language Processing / Interactive Studio</div><h1>💬 SentimentScope</h1><p>Turn reviews and feedback into clear, interpretable sentiment insights — one message or an entire dataset at a time.</p><span class="pill">✦ 3 sentiment categories &nbsp; • &nbsp; Text + CSV analysis</span></div>',unsafe_allow_html=True)

tab1,tab2=st.tabs(['✍️  Text Studio','📊  Dataset Explorer'])
with tab1:
    st.write('')
    left,right=st.columns([1.08,1],gap='large')
    with left:
        st.subheader('Analyze a message')
        st.markdown('<p class="section-lead">Paste a sentence, customer review, or short paragraph to explore its sentiment.</p>',unsafe_allow_html=True)
        sample=st.selectbox('Quick examples',['Write my own','Positive review','Neutral statement','Negative review'])
        examples={'Write my own':'','Positive review':'The service was fantastic and the team was incredibly helpful!','Neutral statement':'The package arrived on Tuesday afternoon.','Negative review':'The item broke immediately and customer support was disappointing.'}
        message=st.text_area('Your text',value=examples[sample],height=190,max_chars=4000,placeholder='Type or paste your text here...')
        go=st.button('✦ Analyze sentiment',type='primary',use_container_width=True)
        st.caption('Supports up to 4,000 characters. Longer inputs are truncated to the model context window.')
    with right:
        st.subheader('Sentiment insights')
        st.markdown('<p class="section-lead">Predicted label and probability distribution across the three classes.</p>',unsafe_allow_html=True)
        if go:
            try:
                with st.spinner('Analyzing language patterns...'):
                    result=classify(message)
                emoji={'positive':'🟢','neutral':'🟡','negative':'🔴'}.get(result['label'],'🔵')
                st.metric('Predicted sentiment',f"{emoji} {result['label'].title()}")
                st.metric('Top model score',f"{result['confidence']:.1%}")
                scores=pd.DataFrame({'Sentiment':[k.title() for k in result['scores']],'Model probability':[float(v) for v in result['scores'].values()]})
                st.bar_chart(scores.set_index('Sentiment'))
                st.caption('Scores are model probabilities, not guarantees of accuracy.')
            except Exception as exc:
                st.error(f'Unable to analyze text: {exc}')
        else:
            st.info('Your prediction will appear here. Choose an example or enter your own text, then select **Analyze sentiment**.')
with tab2:
    st.write('')
    st.subheader('Explore sentiment at scale')
    st.markdown('<p class="section-lead">Upload a CSV of feedback, visualize the overall distribution, and download annotated results.</p>',unsafe_allow_html=True)
    file=st.file_uploader('Upload a CSV dataset',type=['csv'])
    if file is not None:
        try:
            if file.size>2_000_000:raise ValueError('CSV must be smaller than 2 MB.')
            df=pd.read_csv(file)
            if df.empty:raise ValueError('The CSV has no rows.')
            column=st.selectbox('Choose the column containing text',df.columns.tolist())
            st.caption(f'{len(df):,} rows detected · Up to 100 rows per run')
            if st.button('✦ Classify dataset',type='primary'):
                output=df.head(100).copy()
                predictions=[]
                progress=st.progress(0)
                for i,value in enumerate(output[column].fillna('').astype(str)):
                    try:
                        r=classify(value)
                        predictions.append((r['label'],r['confidence']))
                    except ValueError:
                        predictions.append(('invalid',0.0))
                    progress.progress((i+1)/len(output))
                output['sentiment']=[p[0] for p in predictions]
                output['model_score']=[p[1] for p in predictions]
                st.success(f'Analyzed {len(output)} rows.')
                a,b,c=st.columns(3)
                for panel,label in zip([a,b,c],['positive','neutral','negative']):
                    panel.metric(label.title(),int((output['sentiment']==label).sum()))
                st.subheader('Sentiment distribution')
                st.bar_chart(output['sentiment'].value_counts().rename_axis('Sentiment').to_frame('Reviews'))
                st.subheader('Detailed predictions')
                st.dataframe(output,use_container_width=True)
                st.download_button('⬇ Download analyzed CSV',output.to_csv(index=False).encode('utf-8'),'sentimentscope_results.csv','text/csv')
                if len(df)>100:st.warning('Only the first 100 rows were processed.')
        except Exception as exc:
            st.error(f'Could not process CSV: {exc}')
    else:
        st.info('Start with the included `sample_reviews.csv` file, or upload a CSV containing a column of English text.')
st.divider()
st.markdown('<div class="footer-note">SentimentScope · Python · Streamlit · Transformers · Educational demonstration · Predictions require human interpretation.</div>',unsafe_allow_html=True)
