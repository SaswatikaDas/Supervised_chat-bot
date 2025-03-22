import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Define chatbot responses
data = {
    "hello": "Hi there! How can I assist you?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just a bot, but I'm doing great! How about you?",
    "what is your name": "I'm a chatbot created to assist you!",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! Happy to help.",
    "help": "Sure! Let me know what you need help with.",
    "what should i eat for lunch": "How about some rice and curry, or maybe a sandwich?",
    "what should i eat for dinner": "A warm bowl of soup, pasta, or a healthy salad would be great!",
    "are you hungry": "I don't eat, but I can help you pick something delicious!",
    "what’s your favorite food": "I don't eat, but I hear pizza is a popular choice!",
    "what time is it": "I'm not connected to a clock, but you can check your device!",
    "good morning": "Good morning! Hope you have a fantastic day ahead!",
    "good afternoon": "Good afternoon! How’s your day going?",
    "good evening": "Good evening! How was your day?",
    "good night": "Good night! Sleep well and take care.",
    "what’s the weather like": "I can't check the weather, but you can try a weather app!",
    "is it going to rain today": "I’m not sure, but carrying an umbrella is always a good idea!",
    "i am sad": "I’m here for you. Want to talk about it?",
    "i am happy": "That’s great to hear! Keep smiling! 😊",
    "i am bored": "How about watching a movie or reading a book?",
    "i am tired": "You should take a break and rest for a while.",
    "where should i go for vacation": "How about the beach, mountains, or a historical city?",
    "suggest a travel destination": "You could visit Paris for its beauty or Japan for its culture!",
    "how to focus on studies": "Try using the Pomodoro technique—study for 25 minutes, then take a short break!",
    "how to stop procrastinating": "Break tasks into smaller parts and start with the easiest one!",
    "recommend a movie": "How about watching 'Inception' or 'Interstellar'?",
    "recommend a song": "You might like ‘Shape of You’ or ‘Blinding Lights’!",
    "tell me a joke": "Why did the computer catch a cold? Because it left its Windows open!",
    "tell me a fact": "Did you know? The first computer was as big as a room!",
    "default": "I'm sorry, I didn't understand that. Can you rephrase?"
}

# Prepare training data
X = list(data.keys())  # Input phrases
y = list(data.values())  # Corresponding responses

# Create a TF-IDF vectorizer and model pipeline
vectorizer = TfidfVectorizer()
model = LogisticRegression(max_iter=1000)
pipeline = make_pipeline(vectorizer, model)

# Train the model
pipeline.fit(X, y)

# Function to get chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower().strip()

    # Transform user input into vector
    user_input_vectorized = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and existing keys
    X_vectorized = vectorizer.transform(X)
    similarities = cosine_similarity(user_input_vectorized, X_vectorized).flatten()

    # Get the most similar phrase
    best_match_index = np.argmax(similarities)

    # If similarity is below a threshold, return default response
    if similarities[best_match_index] < 0.3:
        return "I don’t have any relevant information for that."

    return y[best_match_index]

# Simple chat loop
print("GIET University Chatbot: Type 'bye' to exit.")
while True:
    user_message = input("You: ").strip()
    if user_message.lower() == "bye":
        print("Chatbot: Goodbye! Have a great day.")
        break
    response = chatbot_response(user_message)
    print("Chatbot:", response)
