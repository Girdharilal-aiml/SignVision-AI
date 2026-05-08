# 🤟 SignVision AI — Real-Time ASL Sign Language Translator

> Translate American Sign Language hand gestures into text and speech using just your webcam.
> **No training. No setup. Just run it.**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20%7C%20Linux-lightgrey)

---

## 🚀 Quick Start

```bash
pip install opencv-python mediapipe numpy pyttsx3
python signvision.py
```

That's it. The app opens your webcam and starts recognising hand gestures immediately.

---

## ✨ Features

- 🖐️ **Real-time hand tracking** — 21 MediaPipe landmarks per frame
- 🔤 **ASL letter recognition** — A B C D E F I L O R S U V W X Y + SPACE
- ⏱️ **Hold-to-type** — hold a gesture still for ~1 second to type the letter
- 🔵 **Visual progress ring** — a circle fills around your fingertip showing when it will type
- 🔊 **Offline text-to-speech** — press ENTER to hear your sentence read aloud
- 🧠 **No ML model needed** — pure geometry, works instantly out of the box
- 💻 **Runs on CPU** — no GPU required

---

## 🎮 How to Use

| Action | Result |
|--------|--------|
| Hold any ASL letter gesture still | Letter types after ~1 second |
| Open palm (all 5 fingers) | Adds a SPACE |
| Press `ENTER` | Speaks the sentence aloud |
| Press `Z` | Deletes last letter |
| Press `C` | Clears everything |
| Press `Q` or `ESC` | Quit |

---

## ✋ Gesture Reference

| Gesture | Letter |
|---------|--------|
| Fist, thumb resting to side | **A** |
| Four fingers up, thumb tucked in | **B** |
| Hand curved into a C shape | **C** |
| Index finger up, others curl to thumb | **D** |
| All fingers bent flat, thumb tucked under | **E** |
| Index + thumb touch, three fingers up | **F** |
| Only pinky finger raised | **I** |
| Index up + thumb out (L shape) | **L** |
| All fingers curve to form an O | **O** |
| Index + middle fingers crossed | **R** |
| Tight fist, thumb across front | **S** |
| Index + middle up, held together | **U** |
| Index + middle up, spread apart (peace sign) | **V** |
| Index + middle + ring all up | **W** |
| Index finger hooked | **X** |
| Thumb + pinky out, others folded | **Y** |
| Fully open palm, all 5 fingers spread | **SPACE** |

---

## 📁 Project Structure

```
SignVision-AI/
├── signvision.py      # Everything — detection, recognition, UI, TTS
├── requirements.txt   # Dependencies
└── README.md
```

The entire app is **one file** — easy to read, easy to modify.

---

## 🧠 How It Works

### No Machine Learning Required

Instead of training a neural network, SignVision uses **hand geometry rules**:

```
Webcam → MediaPipe (21 landmarks) → Geometry rules → Letter → Sentence → TTS
```

Each letter is recognised by measuring distances and ratios between landmarks:

```python
# A finger is "up" when its tip is farther from the wrist than its knuckle.
# This works at any hand angle — rotation invariant.
def is_up(landmarks, tip, mcp):
    return distance(tip, wrist) > distance(mcp, wrist) * 1.2
```

### Hold-to-Type System

A letter is only typed after being held for **22 consecutive frames** (~1 second).
This prevents accidental typing from brief gestures.
A visual progress ring fills around the fingertip so you can see exactly when it will fire.

### Flickering Prevention

A rolling buffer of 10 frames keeps recent predictions.
A letter is only shown when it appears in **more than half** of recent frames.
This eliminates single-frame flickering completely.

---

## 📦 Requirements

```
opencv-python>=4.9.0
mediapipe>=0.10.9
numpy>=1.26.0
pyttsx3>=2.90
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera opens and closes immediately | Change `CAMERA_INDEX = 0` to `1` or `2` in `signvision.py` |
| Hand not detected | Improve lighting — avoid dark rooms or strong backlighting |
| Wrong letters being recognised | Keep hand 40–70 cm from camera, face it forward |
| Letters typing too fast | Increase `HOLD_FRAMES = 22` to `30` in `signvision.py` |
| Letters typing too slowly | Decrease `HOLD_FRAMES = 22` to `15` |
| No speech output | Run `pip install pyttsx3` |
| Low FPS | Close other apps; reduce camera resolution in the settings |

---

## 🔮 Future Ideas

- [ ] Support for full ASL words (dynamic gestures)
- [ ] LSTM model for motion-based letters (J, Z)
- [ ] Word autocomplete / prediction
- [ ] Multi-language text output
- [ ] PyQt6 desktop app wrapper
- [ ] Web version using TensorFlow.js

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🙏 Built With

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — hand landmark detection by Google
- [OpenCV](https://opencv.org/) — real-time video processing
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) — offline text-to-speech
