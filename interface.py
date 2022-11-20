import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

st.write("# Определение животного по картинке")
url = st.text_input("Введите ссылку на картинку >>: ")
try:
    image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    if len(url):
        st.write("Это: ", model.config.id2label[predicted_class_idx])
except: pass
