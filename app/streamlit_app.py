"""
==========================================================
Streamlit Web Application
==========================================================

PURPOSE:
    A beautiful, interactive web interface for the hate speech
    detection system. Users can:
    1. Type or paste text
    2. Get instant predictions
    3. See LIME explanations (which words matter)
    4. Try sample texts

TO RUN:
    cd hate-speech-detector
    streamlit run app/streamlit_app.py
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="🛡️ Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS — Premium Dark Theme
# ============================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #8b95a5;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
    }
    
    /* Label badges */
    .label-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: 0.5px;
    }
    
    .label-neutral {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
    }
    
    .label-offensive {
        background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        color: #2d3436;
    }
    
    .label-hate {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }
    
    /* Confidence meter */
    .confidence-text {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
    }
    
    /* Word highlight */
    .word-highlight {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 6px;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .word-highlight:hover {
        transform: scale(1.1);
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 12px;
    }
    
    /* Sample text buttons */
    .sample-btn {
        width: 100%;
        text-align: left;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        background: rgba(102, 126, 234, 0.08);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .sample-btn:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Hide streamlit specific elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8b95a5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Divider */
    .gradient-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        border: none;
        margin: 20px 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODEL (cached for performance)
# ============================================
@st.cache_resource
def load_predictor():
    """Load the model — cached so it only loads once."""
    from utils.predictor import HateSpeechPredictor
    return HateSpeechPredictor()


@st.cache_resource
def load_explainer(_predictor):
    """Load the LIME explainer — cached."""
    from explainability.lime_explainer import LimeExplainer
    return LimeExplainer(_predictor)


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown('<p class="sidebar-title">🛡️ Hate Speech Detector</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown("""
        This system uses **Multilingual BERT** (mBERT) to detect 
        hate speech in **Hinglish** (Hindi + English) text.
        
        **Features:**
        - 🌐 Multilingual support
        - 😀 Emoji understanding
        - 🗣️ Slang normalization
        - 🔍 LIME explainability
        
        **Model:** `bert-base-multilingual-cased`  
        **Classes:** Neutral, Offensive, Hate Speech
        """)
    
    with st.expander("🧪 How LIME Works", expanded=False):
        st.markdown("""
        **LIME** (Local Interpretable Model-Agnostic Explanations) 
        works by:
        
        1. **Perturbing** the input text (randomly removing words)
        2. **Observing** how predictions change
        3. **Identifying** which words are most influential
        
        🟥 **Red words** = Support the prediction  
        🟩 **Green words** = Oppose the prediction
        
        *More intense color = stronger influence*
        """)
    
    st.markdown("---")
    
    # Sample texts
    st.markdown('<p class="sidebar-title">📝 Sample Texts</p>', unsafe_allow_html=True)
    
    sample_texts = {
        "🟢 Neutral - Weather": "aaj mausam bahut accha hai, bahar chalo",
        "🟢 Neutral - Birthday": "happy birthday bhai, enjoy karo 🎉🥳",
        "🟢 Neutral - Food": "maa ka khana sabse best hota hai 😋",
        "🟡 Offensive - Insult": "kya bakwas movie thi, time waste hua 😤",
        "🟡 Offensive - Mockery": "pagal hai kya tu, dimag kharab hai 🤡",
        "🟡 Offensive - Slang": "chapri log hain sab yahaan pe 🤮",
        "🔴 Hate - Violence": "in logon ko desh se nikal do, yahaan rehne layak nahi",
        "🔴 Hate - Dehumanize": "yeh log insaan nahi hain, janwar hain sab",
        "🔴 Hate - Threat": "aise logon ko maar daalo, duniya acchi hogi 😡🔪",
    }
    
    for label, text in sample_texts.items():
        if st.button(label, key=f"sample_{label}", use_container_width=True):
            st.session_state['input_text'] = text
    
    st.markdown("---")
    
    # Settings
    st.markdown('<p class="sidebar-title">⚙️ Settings</p>', unsafe_allow_html=True)
    
    enable_lime = st.checkbox("Enable LIME Explanation", value=True, 
                              help="Shows which words influenced the prediction")
    
    lime_features = st.slider("LIME Features", 5, 15, 10, 
                              help="Number of important words to highlight")
    
    lime_samples = st.slider("LIME Samples", 100, 1000, 100, step=100,
                             help="More samples = more accurate but slower")


# ============================================
# MAIN CONTENT
# ============================================

# Title
st.markdown('<h1 class="main-title">🛡️ Hate Speech Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Explainable Multilingual Detection using mBERT + LIME</p>', 
            unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Load model
try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading model: {str(e)}")
    st.info("Please train the model first by running: `python training/train.py`")
    model_loaded = False

if model_loaded:
    # Input section
    col_input, col_spacer = st.columns([4, 1])
    
    with col_input:
        # Get text from session state (set by sample buttons) or empty
        default_text = st.session_state.get('input_text', '')
        
        input_text = st.text_area(
            "Enter text to analyze:",
            value=default_text,
            height=120,
            placeholder="Type or paste Hinglish/Hindi/English text here...\n\nExample: yeh bakwas hai 😂 pagal ho kya",
            key="text_input"
        )
    
    # Analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_clicked and input_text.strip():
        # Clear session state
        if 'input_text' in st.session_state:
            del st.session_state['input_text']
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # ============================================
        # PREDICTION
        # ============================================
        with st.spinner("🔄 Analyzing text..."):
            result = predictor.predict(input_text)
        
        # Determine badge class
        label = result['label']
        if label == 'Neutral':
            badge_class = 'label-neutral'
            result_emoji = '🟢'
            alert_type = 'success'
        elif label == 'Offensive':
            badge_class = 'label-offensive'
            result_emoji = '🟡'
            alert_type = 'warning'
        else:
            badge_class = 'label-hate'
            result_emoji = '🔴'
            alert_type = 'error'
        
        # Results display
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Prediction</div>
                <div style="margin-top: 8px;">
                    <span class="label-badge {badge_class}">{result_emoji} {label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence_pct = result['confidence'] * 100
            conf_color = result['color']
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Confidence</div>
                <div class="confidence-text" style="color: {conf_color};">
                    {confidence_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Preprocessed Text</div>
                <div style="font-size: 0.9rem; color: #b0b0b0; margin-top: 12px; word-wrap: break-word;">
                    {result['cleaned_text'][:100]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ============================================
        # PROBABILITY CHART
        # ============================================
        st.markdown("### 📈 Class Probabilities")
        
        probs = result['probabilities']
        
        # Create a beautiful horizontal bar chart
        fig = go.Figure()
        
        colors = {
            'Neutral': '#2ecc71',
            'Offensive': '#f39c12',
            'Hate Speech': '#e74c3c'
        }
        
        for label_name in ['Hate Speech', 'Offensive', 'Neutral']:
            prob = probs[label_name]
            fig.add_trace(go.Bar(
                y=[label_name],
                x=[prob * 100],
                orientation='h',
                marker_color=colors[label_name],
                text=f'{prob*100:.1f}%',
                textposition='auto',
                textfont=dict(size=14, color='white', family='Inter'),
                name=label_name,
                hovertemplate=f'{label_name}: {prob*100:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            showlegend=False,
            height=200,
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis=dict(
                range=[0, 100],
                title='Probability (%)',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)'
            ),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#b0b0b0')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ============================================
        # LIME EXPLANATION
        # ============================================
        if enable_lime:
            st.markdown("### 🔍 LIME Explanation")
            st.markdown("*Which words influenced the prediction?*")
            
            with st.spinner("🧪 Generating LIME explanation... (this may take a moment)"):
                try:
                    explainer = load_explainer(predictor)
                    
                    # Use cleaned text for explanation
                    explanation_data = explainer.explain(
                        result['cleaned_text'],
                        num_features=lime_features,
                        num_samples=lime_samples
                    )
                    
                    # Get highlighted text data
                    highlighted = explainer.get_highlighted_text(
                        result['cleaned_text'],
                        num_features=lime_features,
                        num_samples=lime_samples
                    )
                    
                    # Render highlighted text
                    html_parts = []
                    for word_data in highlighted:
                        word = word_data['word']
                        color = word_data['color']
                        importance = word_data['importance']
                        
                        if word_data['is_important']:
                            tooltip = f"Importance: {importance:+.4f}"
                            html_parts.append(
                                f'<span class="word-highlight" style="background-color: {color};" '
                                f'title="{tooltip}">{word}</span>'
                            )
                        else:
                            html_parts.append(
                                f'<span class="word-highlight" style="background-color: transparent;">'
                                f'{word}</span>'
                            )
                    
                    highlighted_html = ' '.join(html_parts)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size: 1.1rem; line-height: 2.2;">
                            {highlighted_html}
                        </div>
                        <div style="margin-top: 12px; font-size: 0.8rem; color: #8b95a5;">
                            🟥 Red = supports prediction &nbsp;&nbsp; 🟩 Green = opposes prediction
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Word importance bar chart
                    if explanation_data['important_words']:
                        words = [w for w, _ in explanation_data['important_words']]
                        weights = [w for _, w in explanation_data['important_words']]
                        
                        # Sort by absolute importance
                        sorted_pairs = sorted(
                            zip(words, weights), 
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )
                        words_sorted = [w for w, _ in sorted_pairs]
                        weights_sorted = [w for _, w in sorted_pairs]
                        
                        bar_colors = ['#e74c3c' if w > 0 else '#2ecc71' for w in weights_sorted]
                        
                        fig_lime = go.Figure()
                        fig_lime.add_trace(go.Bar(
                            y=words_sorted[::-1],
                            x=weights_sorted[::-1],
                            orientation='h',
                            marker_color=bar_colors[::-1],
                            text=[f'{w:+.3f}' for w in weights_sorted[::-1]],
                            textposition='auto',
                            textfont=dict(size=12, color='white', family='Inter')
                        ))
                        
                        fig_lime.update_layout(
                            title='Word Importance Scores',
                            height=max(250, len(words_sorted) * 35),
                            margin=dict(l=10, r=30, t=40, b=10),
                            xaxis=dict(
                                title='Importance (+ supports, - opposes)',
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.05)',
                                zeroline=True,
                                zerolinecolor='rgba(255,255,255,0.2)'
                            ),
                            yaxis=dict(showgrid=False),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='Inter', color='#b0b0b0'),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_lime, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not generate LIME explanation: {str(e)}")
                    st.info("LIME requires the model to be fully trained. "
                           "Try training the model first.")
        
        # ============================================
        # PREPROCESSING DETAILS (Expandable)
        # ============================================
        with st.expander("🔧 Preprocessing Details", expanded=False):
            from preprocessing.cleaner import TextCleaner
            cleaner = TextCleaner()
            steps = cleaner.clean_with_steps(input_text)
            
            for step_name, step_output in steps.items():
                step_label = step_name.replace("_", " ").title()
                st.markdown(f"**{step_label}:** `{step_output}`")
    
    elif analyze_clicked:
        st.warning("⚠️ Please enter some text to analyze.")

# ============================================
# FOOTER
# ============================================
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.8rem; padding: 20px 0;">
    Built with ❤️ using <b>Multilingual BERT</b> + <b>LIME</b> + <b>Streamlit</b><br>
    Explainable Multilingual Hate Speech Detection System
</div>
""", unsafe_allow_html=True)
