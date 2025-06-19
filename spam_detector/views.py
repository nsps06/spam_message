import re
import nltk
import pickle
import json
from django.shortcuts import render
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')

# Load model and tokenizer
model = load_model("model/spam_classifier_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def clean_english_only(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Render form + prediction
def predict_view(request):
    if request.method == 'POST':
        msg = request.POST.get('message', '')
        cleaned = clean_english_only(msg)
        cleaned = clean_text(cleaned)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=160, padding='post', truncating='post')
        pred = model.predict(padded)[0][0]
        result = "Spam ✅" if pred >= 0.5 else "Not Spam ❌"
        return render(request, 'spam_detector/form.html', {'message': msg, 'result': result, 'confidence': f"{pred:.4f}"})
    return render(request, 'spam_detector/form.html')
