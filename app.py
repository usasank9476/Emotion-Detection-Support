import os
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.graph_objects as go

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_emotions, EMOTIONS
from src.preprocessing import resolve_path

# Load environment variables
load_dotenv(resolve_path('.env'))

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI-Driven Emotion Detection & Learning Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS injection for premium look)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .panel-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .guidance-card {
        background: rgba(99, 102, 241, 0.09);
        border-left: 5px solid #6366f1;
        border-radius: 8px;
        padding: 24px;
        margin-top: 15px;
        line-height: 1.6;
    }
    
    .emotion-tag {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: white;
        margin: 5px 0;
    }
    
    .tag-frustrated { background-color: #ef4444; box-shadow: 0 0 12px rgba(239, 68, 68, 0.4); }
    .tag-confused { background-color: #f59e0b; box-shadow: 0 0 12px rgba(245, 158, 11, 0.4); }
    .tag-curious { background-color: #3b82f6; box-shadow: 0 0 12px rgba(59, 130, 246, 0.4); }
    .tag-confident { background-color: #10b981; box-shadow: 0 0 12px rgba(16, 185, 129, 0.4); }
    .tag-bored { background-color: #6b7280; box-shadow: 0 0 12px rgba(107, 114, 128, 0.4); }
    
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
gemini_configured = False
if api_key and not api_key.lower().startswith("your_"):
    try:
        genai.configure(api_key=api_key)
        gemini_configured = True
    except Exception as e:
        st.sidebar.error(f"Failed to configure Gemini: {e}")

# Sidebar
st.sidebar.markdown("### Platform Configurations")
st.sidebar.markdown(f"**Gemini Connected:** {'✅ Yes' if gemini_configured else '⚠️ No (Using Local Fallback)'}")
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Directory Layout")
st.sidebar.code("""
- data/
- models/
  ├── bltsm/
  └── bert_emotion_model_final/ 
- src/
  ├── preprocessing.py
  ├── model.py
  ├── bert_model.py
  └── predict.py
- app.py
- requirements.txt
""")

# Dashboard Header
st.markdown('<h1 class="main-title">AI-Driven Emotion Detection & Support</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Providing personalized, emotionally intelligent academic mentorship in real-time</p>', unsafe_allow_html=True)

# Logger function
def log_entry(student_text, result, guidance_text):
    log_file = resolve_path('logs.csv')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        'Timestamp': [timestamp],
        'StudentText': [student_text],
        'PrimaryEmotion': [result['primary']],
        'SecondaryEmotion': [result['secondary']],
        'BiLSTM_Scores': [str(result['bilstm_scores'])],
        'BERT_Scores': [str(result['bert_scores'])],
        'Aggregated_Scores': [str(result['aggregated_scores'])],
        'GeminiResponse': [guidance_text]
    }
    df = pd.DataFrame(new_entry)
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, mode='w', header=True, index=False)

def generate_guidance_fallback(student_text, detected_emotions):
    primary = detected_emotions.get('primary', 'Frustrated')
    if primary == 'Frustrated':
        part1 = "It is completely understandable to feel stuck when dealing with stack overflow errors after debugging for hours."
        part2 = "Recursion errors usually occur because the recursive function calls itself indefinitely without hitting a base case. Double-check your function: ensure there is a clear condition that returns a value (the base case) without making another recursive call, and verify that your recursive step moves your arguments closer to that base case on every cycle."
        part3 = "Print or log the input arguments at the very start of your recursive function to see exactly where the infinite loop triggers."
    elif primary == 'Confused':
        part1 = "Getting lost in complex coding concepts like loops and variables is a normal part of the learning process."
        part2 = "Let's break down the logic: a loop runs a block of code repeatedly as long as a condition remains true. If your loop variables are overlapping or not updating correctly, trace the variable values manually line-by-line using a simple paper test to see where they deviate from your expectations."
        part3 = "Add print statements inside the loop body to display the values of all loop counter variables at each iteration."
    elif primary == 'Curious':
        part1 = "It is fantastic that you want to dive deeper into the differences between BiLSTM and BERT models."
        part2 = "BiLSTMs process tokens sequentially in both forward and backward directions to capture local context, which makes them highly efficient for sequence mapping. In contrast, BERT uses self-attention mechanisms to evaluate relationships between all words in a sentence simultaneously, allowing it to capture richer, bidirectional semantic representations."
        part3 = "Try searching for a quick tutorial comparing transformer attention vs LSTM gating mechanisms to see their visual architectures."
    elif primary == 'Confident':
        part1 = "Congratulations on getting your project code working and solving the bug successfully."
        part2 = "Now that the core logic is solid, the next logical progression is to modularize your code into reusable functions or add exception handling. This prevents future bugs and makes your solution robust against invalid inputs."
        part3 = "Review your code and extract any repetitive blocks into separate, focused helper functions."
    else: # Bored
        part1 = "It is common to feel unengaged when a programming concept feels repetitive or dry."
        part2 = "To make this topic more exciting, consider its real-world utility: this exact algorithm is used at scale by companies like Google and Netflix to optimize content delivery networks and search indexing. Applying it to a project you care about—like a personal movie recommendation system—can instantly re-engage your attention."
        part3 = "Think of one hobby or interest of yours and jot down a single sentence about how this algorithm could automate or improve it."
    return f"{part1}\n\n{part2}\n\n{part3}"

def generate_guidance(student_text, result):
    primary = result['primary']
    confidence = result['confidence']
    secondary = result['secondary']
    detected_emotions_str = f"Primary: {primary} (conf: {confidence}), Secondary: {secondary}"
    
    system_instruction = (
        "You are the elite, deeply empathetic generative engine of the AI-Driven Emotion Detection & Personalized Learning Support Platform. "
        "Your purpose is to act as an expert academic mentor and TA. You transform raw, free-text student challenges into immediate, high-quality, emotionally intelligent academic guidance.\n\n"
        "INPUT DATA SCHEMA:\n"
        "1. <student_text>: [The raw, unedited description of the academic challenge or roadblock]\n"
        "2. <detected_emotions>: [A dictionary containing the primary emotion, secondary emotions, and confidence scores]\n\n"
        "SYSTEM GUARDRAILS & BEHAVIORAL DIRECTIVES:\n"
        "- RESPONSE TIME: Prioritize rapid synthesis to keep end-to-end processing under seconds.\n"
        "- STYLE & TONE: Use plain, everyday language. Eliminate meandering setups, dramatic introductions, abstract corporate jargon, or academic-sounding filler (e.g., write 'helps you fix the error' instead of 'facilitates an enhanced debugging methodology'). Do not use overly intimate or dramatic language.\n"
        "- ACTIVE VOICE: Default to strong, active verbs. Prefer direct address ('You can configure this by...') over passive framing ('This can be configured by...').\n"
        "- ACCURACY: Root your technical guidance strictly in correct, actionable concepts related to the student's problem. If the core technical issue is vague, provide a clear, low-friction diagnostic next step.\n\n"
        "EMOTION-SPECIFIC GENERATION RULES:\n"
        "- If 'Confused': Deliver highly structured, step-by-step technical documentation, code snippets, or alternative conceptual metaphors. Break down complexity instantly.\n"
        "- If 'Frustrated': Validate the difficulty directly and neutrally in a single brief phrase, then pivot immediately to lower the barrier to entry with high-impact, easy-to-implement adjustments.\n"
        "- If 'Bored': Introduce a compelling, high-level real-world utility for the concept, or increase the technical depth to re-engage their attention.\n"
        "- If 'Curious': Provide immediate conceptual clarity, followed by a deeper extension resource or a stimulating question to spark further exploration.\n"
        "- If 'Confident': Authentically validate their success, double-check their structural understanding, and suggest an optimal logical progression to maintain their momentum.\n\n"
        "OUTPUT FORMAT SCHEMA:\n"
        "Your response must strictly follow this clean, 3-part layout without markdown headers or labels. Use plain paragraphs for scannability:\n\n"
        "[Part 1: A brief, single-sentence emotional validation tailored strictly to their state.]\n\n"
        "[Part 2: Clear, concrete, and actionable technical advice or step-by-step guidance solving the specific problem context.]\n\n"
        "[Part 3: One singular, low-friction action step the student can execute right now to make progress.]"
    )
    
    if gemini_configured:
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction
            )
            prompt = f"<student_text>: {student_text}\n<detected_emotions>: {detected_emotions_str}"
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return generate_guidance_fallback(student_text, result)
    else:
        return generate_guidance_fallback(student_text, result)

def build_emotion_chart(result):
    emotions = EMOTIONS
    bilstm_vals = [result['bilstm_scores'][e] for e in emotions]
    bert_vals = [result['bert_scores'][e] for e in emotions]
    agg_vals = [result['aggregated_scores'][e] for e in emotions]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=emotions,
        y=bilstm_vals,
        name='BiLSTM Model',
        marker_color='#3b82f6',
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=emotions,
        y=bert_vals,
        name='BERT Model',
        marker_color='#a855f7',
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=emotions,
        y=agg_vals,
        name='Aggregated (Final)',
        marker_color='#f43f5e',
        opacity=0.95
    ))
    
    fig.update_layout(
        title={
            'text': 'Dual-Model vs Aggregated Emotion Scores',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', showgrid=True),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', showgrid=True, range=[0, 1]),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    return fig

# Main Layout split in two columns
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("### Step 1: Input Student Challenge")
    student_input = st.text_area(
        "Describe your programming roadblock, conceptual struggle, or bug here:",
        height=180,
        placeholder="E.g., I've been trying to debug this recursion error for hours and my code keeps hitting a stack overflow. I feel completely stuck."
    )
    submit_btn = st.button("Analyze & Generate Support", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("### Step 2: System Logs & History")
    log_file = resolve_path('logs.csv')
    if os.path.exists(log_file):
        try:
            log_df = pd.read_csv(log_file)
            st.dataframe(
                log_df[['Timestamp', 'PrimaryEmotion', 'SecondaryEmotion', 'StudentText']].tail(5),
                use_container_width=True
            )
        except Exception:
            st.info("Log file is initialized but currently empty.")
    else:
        st.info("No system runs have been logged yet. Complete a test run to populate logs.")
    st.markdown('</div>', unsafe_allow_html=True)

if submit_btn and student_input.strip():
    with st.spinner("Processing input through classification pipelines..."):
        # Run inference
        result = predict_emotions(student_input)
        
        # Run generator
        guidance = generate_guidance(student_input, result)
        
        # Log to file
        log_entry(student_input, result, guidance)
        
    st.markdown("---")
    
    # Display Results Card
    col_res_left, col_res_right = st.columns([1, 1], gap="large")
    
    with col_res_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Classification Output")
        
        # Aggregated stats columns
        st.markdown("##### Detected Emotional State")
        tag_class_p = f"emotion-tag tag-{result['primary'].lower()}"
        tag_class_s = f"emotion-tag tag-{result['secondary'].lower()}" if result['secondary'] else ""
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="metric-title">Primary Emotion</p>', unsafe_allow_html=True)
            st.markdown(f'<span class="{tag_class_p}">{result["primary"]}</span>', unsafe_allow_html=True)
            st.metric("Aggregated Confidence", f"{result['confidence']:.2%}")
            
        with c2:
            st.markdown('<p class="metric-title">Secondary Emotion</p>', unsafe_allow_html=True)
            if result['secondary']:
                st.markdown(f'<span class="{tag_class_s}">{result["secondary"]}</span>', unsafe_allow_html=True)
                sec_conf = result['aggregated_scores'].get(result['secondary'], 0.0)
                st.metric("Secondary Confidence", f"{sec_conf:.2%}")
            else:
                st.markdown('<span class="emotion-tag" style="background-color: #4b5563;">None</span>', unsafe_allow_html=True)
                st.metric("Secondary Confidence", "0.00%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Interactive Score Comparison")
        fig = build_emotion_chart(result)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_res_right:
        st.markdown('<div class="panel-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("### Empathetic Mentorship Guidance")
        
        # Render the 3-part structured guidance
        st.markdown('<div class="guidance-card">', unsafe_allow_html=True)
        # Split by paragraph
        paragraphs = [p.strip() for p in guidance.split('\n\n') if p.strip()]
        for idx, para in enumerate(paragraphs):
            # Give clear structure visually to match the parts
            if idx == 0:
                st.markdown(f"**Mentor Validation:**\n{para}")
            elif idx == 1:
                st.markdown(f"**Conceptual Advice:**\n{para}")
            elif idx == 2:
                st.markdown(f"**Immediate Next Step:**\n*{para}*")
            else:
                st.markdown(para)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
elif submit_btn:
    st.warning("Please enter a valid challenge text before generating guidance.")