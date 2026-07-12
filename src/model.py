from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input

def create_bilstm_model(vocab_size, embedding_dim=128, max_len=100, num_classes=5):
    """
    Constructs a Bidirectional LSTM classifier.
    Designed for Keras 3 compatibility.
    """
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
