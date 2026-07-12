import os
import pickle
import numpy as np
import tensorflow as tf

# Add project root to path if running directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, pad_sequence_data, resolve_path
from src.bert_model import BERTClassifierWrapper

EMOTIONS = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']

# Monkeypatch load_model for Keras 3 path resiliency
_orig_load_model = tf.keras.models.load_model

def resilient_load_model(filepath, *args, **kwargs):
    fp = str(filepath)
    # Check if this points to models/bltsm
    if fp.endswith('bltsm') or fp.endswith('bltsm/'):
        alt1 = fp + '.keras'
        if os.path.exists(alt1):
            return _orig_load_model(alt1, *args, **kwargs)
        alt2 = os.path.join(fp, 'model.keras')
        if os.path.exists(alt2):
            return _orig_load_model(alt2, *args, **kwargs)
    return _orig_load_model(filepath, *args, **kwargs)

tf.keras.models.load_model = resilient_load_model

# Cached models
_bilstm_model = None
_bilstm_tokenizer = None
_bert_classifier = None

def load_pipelines():
    """
    Loads both Keras BiLSTM and HuggingFace BERT models from the models directory.
    Uses dynamic path resolution and caching for efficiency.
    """
    global _bilstm_model, _bilstm_tokenizer, _bert_classifier
    
    if _bilstm_model is None:
        model_path = resolve_path('models/bltsm')
        _bilstm_model = tf.keras.models.load_model(model_path)
        
        tokenizer_path = os.path.join(model_path, 'tokenizer.pkl')
        with open(tokenizer_path, 'rb') as f:
            _bilstm_tokenizer = pickle.load(f)
            
    if _bert_classifier is None:
        bert_path = resolve_path('models/bert_emotion_model_final')
        _bert_classifier = BERTClassifierWrapper(bert_path)

def predict_emotions(text):
    """
    Unified dual-model inference. Aggregates results of BiLSTM and BERT
    to find the primary and secondary emotional states of a student query.
    """
    load_pipelines()
    
    # 1. BiLSTM prediction (cleaned text)
    cleaned = clean_text(text)
    seq = _bilstm_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequence_data(seq, max_len=100)
    bilstm_pred = _bilstm_model.predict(padded, verbose=0)[0]
    
    # 2. BERT prediction (raw text)
    bert_pred = _bert_classifier.predict_probabilities(text)
    
    # Create prediction dictionaries
    bilstm_scores = {EMOTIONS[i]: float(bilstm_pred[i]) for i in range(len(EMOTIONS))}
    bert_scores = {EMOTIONS[i]: float(bert_pred[i]) for i in range(len(EMOTIONS))}
    
    # 3. Aggregate scores
    aggregated_scores = {}
    for emotion in EMOTIONS:
        aggregated_scores[emotion] = (bilstm_scores[emotion] + bert_scores[emotion]) / 2.0
        
    # Find primary and secondary emotions
    sorted_emotions = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
    
    primary = sorted_emotions[0][0]
    confidence = sorted_emotions[0][1]
    secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
    
    return {
        'primary': primary,
        'confidence': round(confidence, 4),
        'secondary': secondary,
        'bilstm_scores': bilstm_scores,
        'bert_scores': bert_scores,
        'aggregated_scores': aggregated_scores
    }

if __name__ == "__main__":
    # Test script offline verification
    test_sentence = "I've been trying to debug this recursion error for hours and my code keeps hitting a stack overflow. I feel completely stuck."
    print(f"Testing text: {test_sentence}")
    result = predict_emotions(test_sentence)
    print("\nResult:")
    import json
    print(json.dumps(result, indent=2))
