import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')

# Sample FAQs
faqs = {
    "What is your return policy?": "You can return items within 30 days with the original receipt.",
    "How can I track my order?": "Track your order using the tracking number in your email.",
    "What payment methods do you accept?": "We accept credit/debit cards, UPI, and net banking.",
    "Do you ship internationally?": "Yes, we offer international shipping with extra charges.",
    "How do I cancel my order?": "Cancel orders from your order history page before it's shipped.",
    "How long does delivery take?": "Delivery usually takes 3-5 business days.",
    "Do you have a customer support number?": "Yes, call us at 1800-123-4567 from 9AM to 6PM.",
    "Can I change my shipping address?": "Yes, you can change it before the order is shipped."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# Function to clean and preprocess text
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = re.findall(r'\b\w+\b', text)  # Basic tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Preprocess all questions
cleaned_questions = [preprocess(q) for q in questions]

# Create TF-IDF vectors for all FAQ questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(cleaned_questions)

# Function to get best matching FAQ response
def get_response(user_input):
    cleaned_input = preprocess(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(input_vector, question_vectors)

    best_match_index = np.argmax(similarity)
    best_score = similarity[0][best_match_index]

    if best_score > 0.3:  # Set a threshold to filter bad matches
        return answers[best_match_index]
    else:
        return "I'm sorry, I couldn't find an answer to that. Try asking something else."

# Chat loop
print("🤖 FAQ Bot: Ask me anything about your order (type 'exit' to quit)\n")
while True:
    user_question = input("You: ")
    if user_question.lower() in ['exit', 'quit']:
        print("FAQ Bot: Goodbye! 👋")
        break
    answer = get_response(user_question)
    print("FAQ Bot:", answer)
