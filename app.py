import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. CONFIGURATION ---
MODEL_ID = "Sid1907/BertModel"

st.set_page_config(
    page_title="Hate-AI Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (Modern Look) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    
    /* Title Styling */
    .title-text {
        font-weight: 800;
        font-size: 48px;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    
    /* Subtitle */
    .subtitle-text {
        text-align: center;
        color: #B0B0B0;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Result Cards */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #1F2229;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        text-align: center;
        animation: fadeIn 0.8s;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: 600;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE (For Clickable Examples) ---
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

def set_text(text):
    st.session_state.text_input = text

# --- 4. MODEL LOADER ---
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        return tokenizer, model
    except Exception as e:
        return None, None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("Project Details")
    st.info(f"**Model:** BERT-Base Uncased")
    st.info(f"**Dataset:** Davidson et al.")
    st.markdown("---")
    st.write("Created by **Siddhant Tamgadge**")
    st.caption("M.Tech Project")
    st.caption("IIIT Bhubaneswar")

# --- 6. MAIN UI ---
st.markdown('<p class="title-text">🛡️ Hate-AI Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Real-time Social Media Content Analyzer</p>', unsafe_allow_html=True)

# Load Model
with st.spinner("🔄 Activating Neural Network..."):
    tokenizer, model = load_model()

if model is None:
    st.error("❌ Error: Could not connect to Model Registry. Check MODEL_ID.")
else:
    # --- QUICK TEST BUTTONS ---
    st.markdown("##### ⚡ Quick Test Examples (Click one):")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        if st.button("🔴 Hate Example"):
            set_text("Black people are criminals and White people are Cheaters.")
            
    with col_ex2:
        if st.button("🟠 Offensive Example"):
            set_text("Your are a very idiotic creature and a bastard")
            
    with col_ex3:
        if st.button("🟢 Safe Example"):
            set_text("I love learning about Tech!")

    # --- TEXT INPUT ---
    user_text = st.text_area(
        "Or type your own text:", 
        value=st.session_state.text_input, 
        height=100, 
        placeholder="Type here..."
    )

    # --- ACTION BUTTONS ---
    c1, c2 = st.columns([3, 1])
    with c1:
        analyze = st.button("🚀 Analyze Text", type="primary")
    with c2:
        if st.button("🗑️ Clear"):
            set_text("")
            st.rerun()

    # --- ANALYSIS LOGIC ---
    if analyze and user_text.strip():
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
        idx = np.argmax(probs)
        confidence = probs[idx]
        
        # Definitions
        labels = ["Hate Speech", "Offensive Language", "Normal / Neither"]
        colors = ["#FF4B4B", "#FFA500", "#2ECC71"]
        icons = ["🚨", "⚠️", "✅"]
        
        # MAIN RESULT CARD
        st.markdown(f"""
        <div class="result-card" style="border-top: 5px solid {colors[idx]};">
            <h1 style="color: {colors[idx]}; font-size: 30px;">{icons[idx]} {labels[idx]}</h1>
            <p style="font-size: 18px; color: #CCCCCC;">Confidence: <b>{confidence*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        # CONFIDENCE METRICS
        st.write("")
        st.write("##### 📊 Confidence Breakdown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Hate Speech", f"{probs[0]*100:.1f}%")
        m2.metric("Offensive", f"{probs[1]*100:.1f}%")
        m3.metric("Normal", f"{probs[2]*100:.1f}%")
        
        # Visual Bars
        st.progress(float(probs[0]), text="Hate Probability")
        
        # FEEDBACK
        if idx == 0:
            st.toast("Hateful content detected!", icon="🚨")
        elif idx == 2:
            st.balloons()