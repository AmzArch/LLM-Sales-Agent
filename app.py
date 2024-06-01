from flask import Flask, request, jsonify
import sales_agent
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import os

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/send_text', methods=['POST'])
def handle_text():
    data = request.json
    transcript = data['transcript']
    response = process_transcript(transcript)
    
    # Text to Speech part
    audio_response = convert_to_speech(response['text'])
    if audio_response:
        play_audio(audio_response)
    
    return jsonify(response)

def process_transcript(text):
    if text.lower() == 'exit':
        reply = "Conversation Ended"
    else:
        # Update conversation history
        if sales_agent.config["conversation_history"] is None:
            sales_agent.config["conversation_history"] = ""
        sales_agent.config["conversation_history"] += f"\nUser: {text}"
        
        # Get response from the response engine
        reply, history = sales_agent.response_engine.reply()
    
    return {"message": "Received", "text": reply}

def convert_to_speech(text):
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    api_key = os.getenv("DG_API_KEY")
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def play_audio(audio_bytes):
    song = AudioSegment.from_file(audio_bytes, format="mp3")
    play(song)

if __name__ == '__main__':
    app.run(debug=True)