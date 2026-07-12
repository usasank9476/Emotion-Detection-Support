import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dynamic path resolution helper
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

def resolve_path(relative_path):
    """
    Resolves relative path against the project root directory.
    This guarantees path resiliency across terminal runs and Streamlit launches.
    """
    return os.path.abspath(os.path.join(ROOT_DIR, relative_path))

# Bootstrap NLTK resources
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.data.find(f"corpora/{resource}.zip")
    except LookupError:
        nltk.download(resource, quiet=True)

def clean_text(text):
    """
    Cleans raw student input: lowercases, removes non-alphabetic characters,
    removes stopwords, and applies lemmatization.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase and strip
    text = text.lower().strip()
    
    # Remove non-alphabetic chars
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Simple whitespace tokenization
    words = text.split()
    
    # Stopword filtering & Lemmatization
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
        
    lemmatizer = WordNetLemmatizer()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned)

def pad_sequence_data(sequences, max_len=100):
    """
    Pads tokenized sequences to a fixed max length.
    Ensures post-padding and post-truncation.
    """
    
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
