
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
 Project Structure
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
​ 1.Forget Gate
ft​=σ(Wf​⋅[ht−1​,xt​]+bf​
 Function

Determines which information from the previous cell state (Cₜ₋₁) should be removed.

How it Works
Takes previous hidden state h
t−1
and current input x t
Passes through a sigmoid function (0 to 1)
0 → completely forget
1 → completely keep

2.Input Gate
it​=σ(Wi​[ht−1​,xt​]+bi​)
C~t​=tanh(Wc​[ht−1​,xt​]+bc​)

Purpose:
Adds new relevant information to the cell state.

Explanation:
The input gate selects useful information from the current input and prepares it to be stored.

3.Cell State Update
Ct​=ft​⊙Ct−1​+it​⊙C~t​
Purpose:
Updates the long-term memory of the network.

Explanation:
Combines previous memory and new information to form updated memory

4.Output Gate
t​=σ(Wo​[ht−1​,xt​]+bo​)
ht​=ot​⊙tanh(Ct​)

Purpose:
Determines the output of the current time step.

Explanation:
Controls which information from the cell state is sent as output.
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
Loss	   Sparse Categorical Crossentropy
Epochs	   50
Batch Size	64
Validation Split	0.1
