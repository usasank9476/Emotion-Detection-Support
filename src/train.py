import os
import pickle
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from keras.utils import to_categorical
from transformers import BertTokenizer, BertForSequenceClassification

# Add project root to path if running directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, pad_sequence_data, resolve_path
from src.model import create_bilstm_model

# 1. Synthetic training dataset aligning with targets: Bored, Confident, Confused, Curious, Frustrated
EMOTIONS = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']
data = [
    # Bored (Class 0)
    ("I am bored with this class, nothing is interesting.", 0),
    ("This topic is dry and boring, I feel sleepy.", 0),
    ("I have no interest in this lecture and feel disconnected.", 0),
    ("It feels repetitive and unengaging.", 0),
    ("I am completely bored and tired of this subject.", 0),
    # Confident (Class 1)
    ("I understand this concept perfectly, it is very clear.", 1),
    ("I solved the debugging issue easily, I feel great.", 1),
    ("I am confident in my programming skills.", 1),
    ("This project was simple and I got it working easily.", 1),
    ("I feel confident I will pass the exam without problems.", 1),
    # Confused (Class 2)
    ("I do not understand recursion at all, it makes no sense.", 2),
    ("I am confused about this error, why is it crashing?", 2),
    ("I feel lost with these complex instructions.", 2),
    ("I do not know what this line of code does.", 2),
    ("I am confused by the loop variables, they overlap.", 2),
    # Curious (Class 3)
    ("I want to learn more about how deep learning works.", 3),
    ("This is interesting, I wonder how we can optimize it.", 3),
    ("I am curious about the difference between BiLSTM and BERT.", 3),
    ("I would like to explore this topic further and research.", 3),
    ("I wonder why this pattern happens, it's fascinating.", 3),
    # Frustrated (Class 4)
    ("I have been debugging this for hours and it still fails, so annoying!", 4),
    ("My code keeps hitting stack overflow, I am completely stuck.", 4),
    ("This error is so frustrating, I want to give up.", 4),
    ("I hate this bug, nothing works and I am losing my mind.", 4),
    ("I feel completely stuck and angry at this stack overflow error.", 4)
]

def train_bilstm():
    print("--- Training Keras BiLSTM ---")
    cleaned_texts = [clean_text(text) for text, _ in data]
    labels = [label for _, label in data]

    # Fit Tokenizer
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_texts)

    # Save Tokenizer
    tokenizer_dir = resolve_path('models/bltsm')
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Saved Tokenizer to {tokenizer_path}")

    # Vectorize and Pad
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequence_data(sequences, max_len=100)
    y = to_categorical(labels, num_classes=5)

    # Create and Fit BiLSTM Model
    model = create_bilstm_model(vocab_size=1000, num_classes=5)
    model.fit(padded, y, epochs=25, batch_size=4, verbose=1)

    # Save Model
    model_save_path = resolve_path('models/bltsm.keras')
    model.save(model_save_path)
    print(f"Saved BiLSTM model to {model_save_path}")

    # Also save inside models/bltsm/ directory to match the layout
    bltsm_dir = resolve_path('models/bltsm')
    os.makedirs(bltsm_dir, exist_ok=True)
    model_dir_save_path = os.path.join(bltsm_dir, 'model.keras')
    model.save(model_dir_save_path)
    print(f"Saved BiLSTM model copy to {model_dir_save_path}")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train_bert():
    print("--- Fine-tuning Tiny BERT ---")
    bert_save_path = resolve_path('models/bert_emotion_model_final')
    os.makedirs(bert_save_path, exist_ok=True)

    # Prajjwal's Tiny BERT is extremely lightweight and fast
    model_name = "prajjwal1/bert-tiny"
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

    texts = [text for text, _ in data]
    labels = [label for _, label in data]

    dataset = EmotionDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(5):
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            batch_labels = batch['labels']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f}")

    # Save model and tokenizer
    model.save_pretrained(bert_save_path)
    tokenizer.save_pretrained(bert_save_path)
    print(f"Saved BERT model to {bert_save_path}")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(resolve_path('data'), exist_ok=True)
    
    # Train both models
    train_bilstm()
    train_bert()
    print("Bootstrap Training Complete!")
