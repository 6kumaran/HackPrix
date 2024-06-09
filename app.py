import os
from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
import requests

app = Flask(__name__)

def generate_response(message):
    # Perform sentiment analysis
    blob = TextBlob(message)
    sentiment = blob.sentiment.polarity

    # Generate response using OpenAI API
    openai_response = get_openai_response(message)

    return openai_response

def get_openai_response(message):
    OPENAI_API_KEY = os.getenv('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')
    if not OPENAI_API_KEY:
        return "OpenAI API key is missing or not set."

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}',
    }

    data = {
        'model': 'text-davinci-003',
        'prompt': message,
        'max_tokens': 150,
        'n': 1,
        'stop': None,
        'temperature': 0.7,
    }

    response = requests.post('https://api.openai.com/v1/completions', headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['text'].strip()
    else:
        return "Sorry, I couldn't process your request at the moment."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = generate_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
