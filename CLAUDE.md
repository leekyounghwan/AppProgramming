# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
python3 digit_recognizer.py
```

## Dependencies

- Python 3.13+
- `pillow` — drawing buffer (PIL Image/ImageDraw)
- `scikit-learn` — MLP classifier + StandardScaler
- `numpy` — array manipulation

Install if missing:
```bash
pip3 install pillow scikit-learn numpy --break-system-packages
```

## Architecture

Single-file app (`digit_recognizer.py`) with two logical layers:

**Model layer** (`train_model` function)
- Loads sklearn's built-in `digits` dataset (1797 samples, 8×8 grayscale images, pixel range 0–16)
- Fits a `StandardScaler` then trains a `MLPClassifier(hidden_layer_sizes=(256, 128))`
- Called in a background thread at startup so the UI is immediately usable

**GUI layer** (`DigitRecognizerApp` class)
- Maintains two parallel drawing surfaces: a `tk.Canvas` (on-screen) and a `PIL.Image` buffer (off-screen, grayscale "L" mode, 280×280)
- On mouse release, `_predict` resizes the PIL buffer to 8×8, rescales pixels from [0,255] → [0,16], flattens to shape `(1,64)`, applies the scaler, then calls `predict_proba`
- Results are displayed as a text label and a 10-row bar chart drawn on per-digit `tk.Canvas` widgets

**Key constant:** `MODEL_INPUT_SIZE = 8` — changing this breaks the scaler/model contract; the dataset format must match.
