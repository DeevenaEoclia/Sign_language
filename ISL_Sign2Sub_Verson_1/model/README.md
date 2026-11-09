# Trained model (not included in this download)

Already trained and saved to Google Drive:
- `sign_model.pth` — LSTM weights, ~80.8% test accuracy on 8 Greetings signs
- `class_names.json` — maps prediction index to sign name

To run `app.py` locally, download both files from
`drive.google.com → ISL_Sign2Sub/model/` and place them in this folder.

The checkpoint also stores `target_len` (64) and the normalization method
used, so `app.py` and `notebooks/4_test.ipynb` preprocess new video
identically to how the model was trained.
