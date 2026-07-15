# AuraStudy: Your Emotion-Aware Academic Mentor

In today's fast-paced educational environment, students often feel overwhelmed by the challenges they face. Whether it's tackling complex subjects or managing time effectively, these hurdles can lead to frustration and disengagement. AuraStudy is here to change that. Our intelligent, emotion-aware academic mentor transforms raw, free-text student challenges into high-quality, emotionally intelligent guidance tailored to each individual's needs. We understand that every student is unique, and our mission is to provide support that truly resonates.

## 📺 Project Demonstration
https://drive.google.com/file/d/199bJNbExfvu0orx-LfvHmy6x_vq6PlYD/view?usp=sharing

## Architecture Overview

### Student Journey Diagram
Below is a visual representation of the student journey within AuraStudy, illustrating how we transform queries into meaningful guidance:

```
+------------------+
|   Student Query  |
+------------------+
          |
          v
+------------------+
|   Preprocessing  |
+------------------+
          |
          v
+------------------+
|  Neural Pipeline |
+------------------+
          |
          v
+----------------------+
|   Emotion Inference  |
+----------------------+
          |
          v
+-----------------------+
|  Guidance Generation  |
+-----------------------+
```

### Dual Neural Pipeline Architecture Diagram
The following diagram details the internal workings of our dual neural pipeline architecture, showcasing how BiLSTM and BERT models collaborate to analyze student emotions:

```
+------------------+       +------------------+
|    Student Data  |       |    Student Data  |
+------------------+       +------------------+
          |                          |
          |                          |
          v                          v
+------------------+       +------------------+
|   Feature        |       |   Feature        |
|   Extraction     |       |   Extraction     |
|   (BiLSTM)       |       |   (BERT)         |
+------------------+       +------------------+
          |                          |
          |                          |
          +------------+-------------+
                       |
                       v
+-----------------------------+
|         Data Fusion         |
+-----------------------------+
                       |
                       v
+-----------------------------+
|      Emotion Classification |
+-----------------------------+
```

## Core Features with Purpose

- **Empathetic Insight**: Our system doesn't just read your words; it understands how you feel, ensuring you receive guidance that is truly relevant and supportive.
- **Personalized Support**: AuraStudy tailors its responses based on individual emotional contexts, allowing students to receive advice that speaks directly to their current challenges.
- **Adaptive Learning**: As students interact with AuraStudy, the system learns from their feedback, continuously improving the relevance and quality of the guidance provided.
- **User-Centric Design**: The intuitive interface makes it easy for students to express their challenges and receive immediate support, fostering a more engaging learning experience.

## Tech Stack
- **Languages**: 
  - Python
- **Frameworks**: 
  - Keras
- **Models**: 
  - BERT
  - BiLSTM
- **Data Handling**: 
  - CSV for logging

## Quick Start / Installation
To get started with AuraStudy, follow these simple steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/usasank9476/Emotion-Detection-Support.git
   cd Emotion-Detection-Support
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

With these straightforward instructions, you'll be ready to experience the transformative power of AuraStudy in your academic journey!
Testing Git
