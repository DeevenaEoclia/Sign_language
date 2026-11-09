"""
ISL Sign2Sub - Final webpage
Indian Sign Language -> English Subtitles

Run LOCALLY with:
    streamlit run app.py

This does NOT run on Google Colab (no live browser webcam / server there).
Train the model on Colab (see notebooks/3_train.ipynb), then download
model/sign_model.pth and model/class_names.json from your Google Drive into
this project's local model/ folder before running this app.

This file matches exactly what was built and trained in the notebooks:
  - MediaPipe Tasks API (HandLandmarker + PoseLandmarker), not the old
    mp.solutions API, which no longer exists in current MediaPipe versions.
  - 225 numbers per frame: left hand (63) + right hand (63) + pose (99).
  - Every video resampled to a fixed length (ckpt["target_len"], 64 frames
    for the notebooks-as-written) using evenly spaced frame indices.
  - Landmarks normalized relative to the shoulder midpoint/width, exactly
    as in notebooks/3_train.ipynb's normalize_sequences(), because the
    saved model was trained on normalized input.

If any of these steps were skipped, the model receives input it was never
trained on, so this file is intentionally kept in sync with the notebooks
rather than reimplemented differently.
"""

import os
import json

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn

MODEL_PATH = "model/sign_model.pth"
CLASS_NAMES_PATH = "model/class_names.json"
HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"
OUTPUT_DIR = "output"

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

st.set_page_config(page_title="ISL Sign2Sub", page_icon="🤟")
st.title("ISL Sign2Sub")
st.caption("Indian Sign Language → English Subtitles")


# ---------------------------------------------------------------------------
# One-time setup: download MediaPipe's .task model files if not present
# ---------------------------------------------------------------------------
def ensure_mediapipe_models():
    import urllib.request
    if not os.path.exists(HAND_MODEL_PATH):
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    if not os.path.exists(POSE_MODEL_PATH):
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)


@st.cache_resource
def load_landmarkers():
    ensure_mediapipe_models()
    BaseOptions = mp.tasks.BaseOptions
    RunMode = mp.tasks.vision.RunningMode
    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=RunMode.IMAGE,
            num_hands=2,
        )
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=RunMode.IMAGE,
        )
    )
    return hand_landmarker, pose_landmarker


class SignLSTM(nn.Module):
    """Same architecture as notebooks/3_train.ipynb - must match for loading weights."""
    def __init__(self, input_size, hidden_size=64, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n[-1])


@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return None, None, None
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = SignLSTM(ckpt["input_size"], ckpt["hidden_size"], len(class_names)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    target_len = ckpt.get("target_len", 64)
    return model, class_names, target_len


def frame_to_mp_image(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def extract_frame_features(frame_bgr, hand_landmarker, pose_landmarker):
    """225 numbers per frame: left hand (63) + right hand (63) + pose (99).
    Must match notebooks/2_landmarks.ipynb exactly, including the fixed
    left/right slot assignment and zero-fill for a missing hand.
    """
    mp_image = frame_to_mp_image(frame_bgr)
    hand_result = hand_landmarker.detect(mp_image)
    pose_result = pose_landmarker.detect(mp_image)

    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    pose = np.zeros(99, dtype=np.float32)

    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
        if handedness[0].category_name == "Left":
            left = coords
        else:
            right = coords

    if pose_result.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]], dtype=np.float32
        ).flatten()

    return np.concatenate([left, right, pose])


def resample_sequence(seq, target_len):
    """Evenly spaced frame indices, same as notebooks/3_train.ipynb."""
    idx = np.linspace(0, len(seq) - 1, target_len).round().astype(int)
    return seq[idx]


def normalize_sequence(seq):
    """Shoulder-relative normalization, matching notebooks/3_train.ipynb's
    normalize_sequences() exactly (applied here to a single sequence with
    a batch dimension of 1).
    """
    X = seq[None]  # add batch dim -> (1, T, 225)
    N, T, _ = X.shape
    pose = X[:, :, 126:].reshape(N, T, 33, 3)
    ls, rs = pose[:, :, 11, :], pose[:, :, 12, :]  # left/right shoulder
    center = (ls + rs) / 2
    scale = np.linalg.norm((ls - rs)[:, :, :2], axis=2, keepdims=True)
    scale = np.where(scale < 1e-3, 1.0, scale)

    out = X.copy()
    for start in (0, 63):
        hand = X[:, :, start:start + 63].reshape(N, T, 21, 3)
        missing = np.all(hand == 0, axis=(2, 3), keepdims=True)
        hand_n = (hand - center[:, :, None, :]) / scale[:, :, None, :]
        out[:, :, start:start + 63] = np.where(missing, 0.0, hand_n).reshape(N, T, 63)
    out[:, :, 126:] = ((pose - center[:, :, None, :]) / scale[:, :, None, :]).reshape(N, T, 99)
    return out.astype(np.float32)[0]  # drop batch dim


def words_to_sentence(words):
    if not words:
        return ""
    sentence = " ".join(w.replace("_", " ").capitalize() for w in words)
    question_words = {"how", "how_are_you", "what", "why", "where", "who"}
    if words[0].lower() in question_words:
        sentence += "?"
    else:
        sentence += "."
    return sentence


def generate_srt(lines, seconds_per_line=2.0, out_path=f"{OUTPUT_DIR}/subtitles.srt"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    def ts(s):
        h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
        ms = int((s - int(s)) * 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    with open(out_path, "w") as f:
        for i, line in enumerate(lines):
            start, end = i * seconds_per_line, (i + 1) * seconds_per_line
            f.write(f"{i+1}\n{ts(start)} --> {ts(end)}\n{line}\n\n")
    return out_path


class FrameCollector:
    """Collects raw video frames coming from the browser via streamlit-webrtc."""
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        self.frames.append(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


if "collector" not in st.session_state:
    st.session_state.collector = FrameCollector()

model, class_names, target_len = load_model_and_classes()

if model is None:
    st.warning(
        "No trained model found yet. Download model/sign_model.pth and "
        "model/class_names.json from your Google Drive (ISL_Sign2Sub/model/) "
        "into this project's local model/ folder, then restart this app. "
        "This app will not fabricate a prediction without a real trained model."
    )
else:
    with st.spinner("Loading MediaPipe hand/pose detectors (first run only)..."):
        hand_landmarker, pose_landmarker = load_landmarkers()

st.subheader("1. Start camera and perform an ISL sign")

ctx = webrtc_streamer(
    key="isl-camera",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=st.session_state.collector.recv,
    media_stream_constraints={"video": True, "audio": False},
)

col1, col2 = st.columns(2)
with col1:
    done = st.button("DONE", disabled=model is None)
with col2:
    if st.button("Record Again"):
        st.session_state.collector.frames = []
        st.rerun()

if done:
    frames = st.session_state.collector.frames
    if len(frames) == 0:
        st.error("No frames captured. Make sure the camera is running and you performed a sign.")
    else:
        with st.spinner("Processing your sign language..."):
            landmark_seq = np.stack([
                extract_frame_features(f, hand_landmarker, pose_landmarker) for f in frames
            ])
            landmark_seq = resample_sequence(landmark_seq, target_len)
            landmark_seq = normalize_sequence(landmark_seq)

            x = torch.tensor(landmark_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1))
                confidence = float(probs[0, pred_idx])

            predicted_word = class_names[str(pred_idx)]

        st.success("Done!")
        st.subheader("Translation")
        st.markdown(f"### {predicted_word.replace('_', ' ').capitalize()}")
        st.caption(f"Confidence: {confidence:.2%}")

        sentence = words_to_sentence([predicted_word])
        srt_path = generate_srt([predicted_word.replace("_", " ").capitalize()])
        txt_path = f"{OUTPUT_DIR}/subtitles.txt"
        with open(txt_path, "w") as f:
            f.write(sentence)

        st.subheader("Subtitle")
        st.text(sentence)

        c1, c2 = st.columns(2)
        with c1:
            with open(srt_path, "rb") as f:
                st.download_button("Download SRT", f, file_name="subtitles.srt")
        with c2:
            with open(txt_path, "rb") as f:
                st.download_button("Download TXT", f, file_name="subtitles.txt")

st.divider()
st.caption(
    "This is a sign-recognition prototype trained on 8 Greetings-category "
    "ISL signs (~81% test accuracy), not a full continuous-sentence "
    "translator. It recognizes one sign per recording. See README.md and "
    "notebooks/5_subtitles.ipynb for the stated limitation."
)
