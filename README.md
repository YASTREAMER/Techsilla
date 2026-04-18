# Techsilla

An end-to-end AI-powered mock interview system that generates domain-specific questions, transcribes spoken answers, and evaluates responses using large language models — all running locally on modest hardware.

---

## 🚀 Overview

This project simulates a technical interview experience using a fully automated pipeline:

1. User selects a domain (e.g., Machine Learning, Web Development, Data Structures)
2. LLM dynamically generates interview questions
3. User answers by speaking
4. Speech is transcribed to text
5. Grammar and clarity are improved
6. Answer is evaluated using an LLM
7. Feedback is returned to the user

The entire system is optimized to run locally on a laptop with limited GPU resources.

---

## ⚙️ Key Features

* 🎯 Domain-specific question generation using LLMs
* 🎤 Speech-based answer input
* 🧠 Multi-stage evaluation pipeline:

  * Speech-to-text transcription
  * Grammar correction
  * Semantic answer evaluation
* ⚡ Runs locally on **4GB VRAM GPU + 16GB RAM**
* 🔌 Built using lightweight transformer-based models
* 🧪 Hackathon prototype with real-time inference capability

---

## 🏗️ Architecture

```
User Input (Domain)
        ↓
Question Generation (LLM)
        ↓
User Speech Input
        ↓
Speech-to-Text (Whisper-based model)
        ↓
Grammar Correction (LLM)
        ↓
Answer Evaluation (LLM)
        ↓
Feedback Output
```

---

## 🧰 Tech Stack

* Python
* Hugging Face Transformers
* Whisper (speech-to-text)
* Local LLM inference (optimized for low VRAM)
* Jupyter Notebooks (prototype development)

---

## 💻 System Requirements

Minimum setup used:

* GPU: 4GB VRAM
* RAM: 16GB
* Python 3.8+

The system is designed to work efficiently under constrained hardware by using optimized model loading and inference strategies.

---

## ⚡ Optimization Highlights

* Efficient model loading using Hugging Face Transformers
* Lightweight model selection to fit within 4GB VRAM
* Sequential pipeline execution to manage memory usage
* Minimal latency suitable for interactive use

---

## 📦 Installation

```bash
git clone https://github.com/YASTREAMER/Techsilla.git
cd Techsilla
```

---

## ▶️ Usage

```bash
python main.py
```


---


## 🧠 Future Improvements

* Add scoring rubric for more structured evaluation
* Improve latency with model quantization
* Add frontend UI for better user experience
* Store interview history and analytics
* Introduce adaptive questioning based on performance

---

## 🤝 Contribution

This was built as part of a hackathon project. Contributions, suggestions, and improvements are welcome.

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* Open-source LLM ecosystem
* Hugging Face for model hosting and tooling
* Whisper for speech recognition

---

## 📌 Note

This was build as a part of a hackathon and the original code was never fully uploaded thus this was uploaded as of now.
