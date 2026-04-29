
Deep Learning
Lab Title	Research Paper Implementation with Pre-trained Model
Project Title	LSTM-Based Sequence Prediction System
Submission Date	April 2026
Group Members
Name	PRN
Shreyasi Bidkar	202301040042
Nikita Bansod	  202301040110
Shravni Karhale	202301040139
Nishi Agarwal  	202301040219

Project Overview

This project implements an LSTM (Long Short-Term Memory) based text sequence prediction system.
The model is trained on Shakespeare’s Complete Works dataset and predicts the next word based on an input sequence.

The system is deployed using FastAPI, enabling real-time predictions via REST API.

Objectives
Build an LSTM-based next-word prediction model
Train the model on Shakespeare dataset
Perform sequence learning using NLP techniques
Deploy the model using FastAPI
Create a complete end-to-end AI system

Dataset
Field	Details
Dataset Name	Shakespeare Complete Works
Source	Project Gutenberg
Link	https: https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays?resource=download

Size	~5.5 MB
Content	Plays + Sonnets
🏗️ Project Structure
lstm-text-prediction/
│
├── model/
│   ├── lstm_model.keras
│   ├── lstm_best.keras
│   ├── tokenizer.pkl
│   └── max_seq_len.pkl
│
├── LSTM_Text_Prediction.ipynb
├── main.py
├── requirements.txt
├── training_curves.png
└── README.md
 LSTM Mathematical Model
🔹 Gates
Forget Gate:
f
t
	​

=σ(W
f
	​

[h
t−1
	​

,x
t
	​

]+b
f
	​

)
Input Gate:
i
t
	​

=σ(W
i
	​

[h
t−1
	​

,x
t
	​

]+b
i
	​

)
Cell State:
C
t
	​

=f
t
	​

⊙C
t−1
	​

+i
t
	​

⊙
C
~
t
	​

Output Gate:
o
t
	​

=σ(W
o
	​

[h
t−1
	​

,x
t
	​

]+b
o
	​

)
Hidden State:
h
t
	​

=o
t
	​

⊙tanh(C
t
	​

)
 Concept
Cell State (Cₜ): Long-term memory
Hidden State (hₜ): Short-term output
 Model Architecture
Embedding Layer (100)
        ↓
LSTM (150 units)
        ↓
LSTM (100 units)
        ↓
Dropout (0.3)
        ↓
Dense (Softmax)

Training Configuration
Parameter	Value
Optimizer	Adam
Loss	Sparse Categorical Crossentropy
Epochs	50
Batch Size	64
Validation Split	0.1
