import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import tensorflow as tf
import logging
from streamlit_drawable_canvas import st_canvas
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
import numpy as np
import cv2
import pickle
import base64
import time
from pathlib import Path
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from pathlib import Path 
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
# Page Config

st.set_page_config(
    page_title="Myanmar Handwriting OCR",
    page_icon="✍",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "phase2_model1.keras"
LABEL_PATH = "label_encoder1.pkl"
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder
model, label_encoder = load_resources()

# ==============================
# Character Segmentation
# ==============================



def segment_characters(bw_img):
    # bw_img:
    # background = black
    # character = white
    # (THRESH_BINARY_INV output)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_img,connectivity=8)
    boxes = []
    # Skip background label 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # remove small noise
        if area > 80:
            boxes.append((x, y, w, h))
    # left to right sorting
    boxes = sorted(boxes,key=lambda b: (b[1]//50, b[0]))
    chars = []
    for x, y, w, h in boxes:
        pad = 35
        x1 = max(0, x-pad)
        y1 = max(0, y-pad)
        x2 = min(bw_img.shape[1],x+w+pad)
        y2 = min(bw_img.shape[0],y+h+pad)
        crop = bw_img[y1:y2, x1:x2]
        h2, w2 = crop.shape
        # create white background
        size = max(h2, w2) + 70
        canvas = np.ones((size, size),dtype=np.uint8) * 255
        # IMPORTANT:
        # training image = white background + black character
        crop = 255 - crop
        sx = (size-w2)//2
        sy = (size-h2)//2
        canvas[
            sy:sy+h2,
            sx:sx+w2
        ] = crop
        chars.append(canvas)
    return chars


#def segment_characters(bw_img):
    #bw_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,20)
    contours, _ = cv2.findContours(bw_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 80:
            boxes.append((x,y,w,h))
    boxes = sorted(boxes, key=lambda b:(b[1]//40,b[0]))
    chars = []
    for (x,y,w,h) in boxes:
        pad = 12
        x1=max(0,x-pad)
        y1=max(0,y-pad)
        x2=min(bw_img.shape[1],x+w+pad)
        y2=min(bw_img.shape[0],y+h+pad)
        crop = bw_img[y1:y2,x1:x2]
        h2,w2=crop.shape
        size=max(h2,w2)+24
        canvas=np.ones((size,size),dtype=np.uint8)*255
        sx=(size-w2)//2
        sy=(size-h2)//2
        canvas[sy:sy+h2,sx:sx+w2]=255-crop
        chars.append(canvas)
    return chars

def preprocess_character(img):
    img = Image.fromarray(img)
    img = img.convert("RGB")
    img = img.resize((224,224),Image.Resampling.LANCZOS)
    img = np.array(img)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img,0)
    return img

def predict_character(char_img):
    #img = preprocess_character(char_img)
    prediction = model.predict(char_img,verbose=0)
    index = np.argmax(prediction)
    label = label_encoder.inverse_transform([index])[0]
    confidence = prediction[0][index]
    return label, confidence

# Session State
if "prediction_text" not in st.session_state:
    st.session_state.prediction_text = ""
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# CSS.main{background:#f5f5f5;}.stApp {background: linear-gradient(to bottom, #FFF8E7, #F5F5DC);}

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #FFFDF5 0%, #FFF8E7 50%, #F8F1E5 100%);}
            
/* ---------------- Title ---------------- */
.title{text-align:center;font-size:40px;font-weight:bold;color:#1E3A8A;}
.subtitle{text-align:center;color:="#72A806FF";}
h3{color:#00897B !important;font-weight:bold;}
            
/* ---------------- Text ---------------- */
p, label, span, div{color:black;}
.result-box{background:white;border:3px solid #2196F3;border-radius:15px;padding:25px;font-size:45px;font-weight:bold;text-align:center;
/* min-height:100px; */
}
            
/* ---------------- Prediction Cursor ---------------- */
.stTextInput input{color:black !important;font-size:22px;font-weight:bold;border:2px solid #00897B !important;border-radius:10px;}
div[data-testid="stMarkdownContainer"] p{color:#000000 !important;}         
.stTextInput input{color:black !important;font-size:22px;font-weight:bold;border:2px solid #00897B !important;border-radius:10px;}

.stButton > button{
    background:#43A047;
    color:#558804FF;
}   
.stTextInput > div > div > input{
    background-color: #FFFDE7 !important;   
    color: #000000 !important;              
    border: 2px solid #43A047 !important;   
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
.stTextInput > div > div > input:focus{border: 2px solid #00897B !important;box-shadow: 0 0 8px rgba(0,137,123,0.3);}
    
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<div class='title'>Myanmar Handwriting OCR</div>",unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Draw multiple Myanmar characters and recognize them.</div>",unsafe_allow_html=True)
#st.divider()
st.subheader("White Drawing Board")
canvas_result = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=7,
    stroke_color="#030500FF",
    background_color="#FEFFFFFF",
    width=900,
    height=400,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)
# Buttons
# -----------------------------
col1, col2, col3, col4 = st.columns([1,1,1,5])
with col1:
    predict_btn = st.button("Predict")

with col2:
    if st.button("Clear Board"):
        st.session_state.canvas_key += 1
        st.rerun()

with col3:
    if st.button("Clear Text"):
        st.session_state.prediction_text = ""
        st.rerun()

# Prediction Result
if predict_btn:
    if canvas_result.image_data is None:
        st.warning("Please draw a Myanmar character first.")
    else:
        img = canvas_result.image_data.astype(np.uint8)
        print(canvas_result.image_data.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.GaussianBlur(gray, (5,5), 0)
        bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,20)
        print(bw.shape)
        kernel = np.ones((3,3), np.uint8)
        # ======================
        # Segment characters
        # ======================
        characters = segment_characters(bw)
        st.write("Detected characters:", len(characters))
        if len(characters)==0:
            st.warning("No character detected")
            st.stop()
        result=""
        confidence=[]
        # Predict each character
        for i, char in enumerate(characters):
            mapping = {
                "r0_c0":"က","r0_c1":"ခ","r0_c2":"ဂ","r0_c3":"ဃ","r0_c4":"င",
                "r1_c0":"စ","r1_c1":"ဆ","r1_c2":"ဇ","r1_c3":"ဈ","r1_c4":"ည",
                "r2_c0":"ဋ","r2_c1":"ဌ","r2_c2":"ဍ","r2_c3":"ဎ","r2_c4":"ဏ",
                "r3_c0":"တ","r3_c1":"ထ","r3_c2":"ဒ","r3_c3":"ဓ","r3_c4":"န",
                "r4_c0":"ပ","r4_c1":"ဖ","r4_c2":"ဗ","r4_c3":"ဘ","r4_c4":"မ",
                "r5_c0":"ယ","r5_c1":"ရ","r5_c2":"လ","r5_c3":"ဝ","r5_c4":"သ",
                "r6_c1":"ဟ","r6_c2":"ဠ","r6_c3":"အ"
            }

            #st.image(char, width=120)
            st.image(char, caption=f"Character {i+1}", width=100)
            cv2.imwrite(f"debug_{i}.png", char)
            preprocess_img = preprocess_character(char)
            print(preprocess_img.size)
            print(preprocess_img.shape)
            pred, conf = predict_character(preprocess_img)
            
            st.markdown(
                f"""
                <div style="background:white;padding:12px;border-radius:10px;border-left:5px solid #43A047;margin-bottom:10px;">
                    <span style="font-size:20px;color:black;">
                    <b>Prediction:</b> {mapping.get(pred)}
                    </span><br>
                    <span style="font-size:18px;color:black;">
                        <b>Confidence:</b> {conf:.2%}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            #st.write(pred)
            result += mapping.get(pred,pred)
            confidence.append(conf)
        # append like keyboard
        st.session_state.prediction_text += result
        st.markdown(
            f"""
            <div style="background:#E8F5E9;border-left:5px solid #43A047;padding:12px;border-radius:10px;color:black;font-size:20px;font-weight:bold;">
                Mean Confidence : {np.mean(confidence):.2%}
            </div>
            """,
            unsafe_allow_html=True
        )
        # Auto clear drawing board after prediction
        #st.session_state.canvas_key += 1
        #st.rerun()
st.subheader("Prediction Cursor")
st.text_input(
    "Prediction",
    value=st.session_state.prediction_text,
    key="cursor_box",
    label_visibility="collapsed"
)