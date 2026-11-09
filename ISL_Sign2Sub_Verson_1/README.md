# ISL Sign2Sub

### Indian Sign Language → English Text → Subtitles

A college-level AI/ML project that recognizes **Indian Sign Language (ISL)**
signs from video and converts them into English subtitles.

```text
CAMERA → ISL SIGNING → VIDEO → FRAME EXTRACTION → MEDIAPIPE
      → HAND + POSE LANDMARKS → LSTM → ISL SIGN WORD
      → ENGLISH WORD/SENTENCE → SUBTITLE (.srt / .txt)
```

This is a **sign recognition** system, not a full continuous-translation
system. It recognizes individual signs and stitches them into a simple
sentence. See `notebooks/5_subtitles.ipynb` for why that's an intentional
limitation, not an oversight.

---

## Current status (as actually trained and tested)

- **Signs trained:** 8, all from the INCLUDE dataset's "Greetings" category:
  `HELLO, THANK_YOU, ALRIGHT, GOOD_MORNING, GOOD_AFTERNOON, GOOD_EVENING,
  HOW_ARE_YOU, PLEASED`
- **Training data:** 169 real videos (118 train / 25 validation / 26 test)
- **Test accuracy:** ~80.8% (21/26 correct) on videos never seen during training
- **Confirmed working end to end:** video → landmarks → trained model →
  prediction → subtitle file, tested on both dataset videos and verified
  live in `app.py`
- **Not yet done:** training on the full 263-class INCLUDE dataset. Notebook
  2's Section 9 has the code to extract landmarks for more categories if you
  want to expand beyond Greetings — see that section for details.

---

## Folder structure

```text
ISL_Sign2Sub/
│
├── notebooks/
│   ├── 1_video.ipynb        # Video/frames/FPS basics (OpenCV)
│   ├── 2_landmarks.ipynb    # MediaPipe Tasks API: hand + pose landmarks
│   ├── 3_train.ipynb        # LSTM training (resampling + normalization)
│   ├── 4_test.ipynb         # Test the trained model on new videos
│   └── 5_subtitles.ipynb    # Predicted words → sentence → .srt / .txt
│
├── data/
│   ├── videos/       # Downloaded INCLUDE videos go here (not included)
│   └── landmarks/    # Extracted .npy landmark files go here (not included)
│
├── model/
│   └── (sign_model.pth, class_names.json — not included; see below)
│
├── output/           # Generated .srt / .txt subtitle files
│
├── app.py            # Final Streamlit webcam webpage (run LOCALLY, not Colab)
├── requirements.txt
└── README.md
```

**Why `data/`, and `model/` are empty in this download:** the real dataset
videos, extracted landmarks, and trained model all live in Google Drive
(from training on Colab), not on any local machine — see "Getting your
trained model" below for how to bring them into a fresh local copy of this
project.

---

## Running this project

### On Google Colab (where all actual training happened)

Every notebook auto-detects Colab vs. local and sets `PROJECT_DIR`
accordingly — no manual path editing needed. Just:

1. Upload this whole folder to Google Drive, e.g. `My Drive/ISL_Sign2Sub`.
2. Open a notebook in Colab (File → Open notebook → Google Drive tab).
3. Run the first cell — it mounts Drive and sets `PROJECT_DIR`.
4. For `3_train.ipynb`: turn on a free GPU first (Runtime → Change runtime
   type → T4 GPU).

### Locally, for `app.py` only

Colab can't run a live webcam + local server, so the final webpage must run
on your own computer:

```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

`app.py` auto-downloads the MediaPipe `.task` model files on first run.

### Getting your trained model onto your local machine

1. Go to `drive.google.com` → `ISL_Sign2Sub/model/`.
2. Download `sign_model.pth` and `class_names.json`.
3. Place both files into this project's local `model/` folder.
4. Run `streamlit run app.py`.

---

## Dataset: INCLUDE (AI4Bharat / IIT Madras)

| | |
|---|---|
| **Name** | INCLUDE — *A Large Scale Dataset for Indian Sign Language Recognition* |
| **Hosted at** | https://zenodo.org/record/4010759 |
| **Code/paper repo** | https://github.com/AI4Bharat/INCLUDE |
| **Type** | Isolated signs (one sign per video clip), NOT continuous sentences |
| **Full dataset** | 4,292 videos across 263 sign classes, ~15 categories |
| **Used in this project** | Greetings category only: 169 videos, 8 classes |
| **License** | MIT (code repo) |

### Why INCLUDE and not ASL datasets
WLASL and ASLLVD are American Sign Language — grammatically and lexically
different from ISL. This project does not use them anywhere except as a
one-line comparison note.

### How videos were obtained
Downloaded directly inside Colab from Zenodo's API (not redistributed in
this repo — see `notebooks/2_landmarks.ipynb` Section 1 for the exact
commands), then copied into Google Drive at `data/videos/<SIGN_NAME>/`.

### Expanding beyond Greetings
`notebooks/2_landmarks.ipynb` Section 9 has a `process_category()` function
that downloads+extracts+cleans up one category at a time (so raw videos
never pile up on Drive — only the small `.npy` landmark files are kept).
Expect roughly 9 seconds per video; the full dataset is several hours of
processing, resumable across sessions. Accuracy will likely be well below
the current 80.8%, since 263 classes means far less data per class and a
much harder classification problem — that's expected, not a bug.

---

## Build order

1. `notebooks/1_video.ipynb` — video/frames/FPS with OpenCV
2. `notebooks/2_landmarks.ipynb` — MediaPipe landmark extraction
3. `notebooks/3_train.ipynb` — LSTM training
4. `notebooks/4_test.ipynb` — test on new videos
5. `notebooks/5_subtitles.ipynb` — words → sentence → subtitles
6. `app.py` — Streamlit webcam webpage (run locally, after 1–5 and after
   downloading the trained model from Drive)

## Tech stack

- **OpenCV** — video/frame handling
- **MediaPipe (Tasks API)** — hand + pose landmark extraction (pre-trained,
  not trained by us)
- **PyTorch (LSTM)** — the one model trained from scratch for this project
- **scikit-learn** — train/validation/test splitting
- **Streamlit + streamlit-webrtc** — local webcam UI (not Colab)

## Known limitations (be upfront about these in any report)

- Recognizes **one isolated sign at a time**, not continuous/grammatical
  ISL sentences.
- Trained on only 8 signs from one category, and a small number of signers,
  so it won't generalize well to signs or signers outside the training data.
- Test accuracy (~80.8%) is based on only 26 videos and may include
  near-duplicate takes split across train/test, making it somewhat
  optimistic versus true real-world performance.
- Sentence generation (`words_to_sentence`) is simple capitalization +
  punctuation, not real grammatical translation.
