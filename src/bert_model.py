from transformers import BertTokenizer, BertForSequenceClassification
import torch

class BERTClassifierWrapper:
    def __init__(self, model_dir):
        """
        Loads the BERT model and tokenizer from a local directory.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict_probabilities(self, text):
        """
        Runs BERT inference on the input text and returns probability distribution.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Standard softmax to calculate probabilities
            probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
            
        return probabilities
