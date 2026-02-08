"""
Web Voice Bot with WebRTC Echo Cancellation + SmartTurn
========================================================
A web-based voice AI bot that uses:
- Browser WebRTC for echo cancellation (no headphones needed!)
- Deepgram Nova-2 for STT
- OpenAI GPT-4o-mini for LLM
- ElevenLabs for TTS
- SmartTurn for intelligent turn detection

Run: python web_voice_bot.py
Open: http://localhost:5002
"""

import os
import sys
import json
import base64
import time
import asyncio
import threading
from queue import Queue, Empty

import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import websocket
import ssl
import certifi
import requests

# Fix SSL on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

load_dotenv(override=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'voice-bot-secret')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# API Keys
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# Store active sessions
active_sessions = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============ Voice Options ============
VOICES = {
    'neha': {'id': 'Zs2gGSc3xT4kRfIqS9R3', 'name': 'Neha (Indian Female)', 'description': 'Warm, friendly Indian English'},
    'george': {'id': 'JBFqnCBsd6RMkjVDRZzb', 'name': 'George (US Male)', 'description': 'Deep, authoritative'},
    'jessica': {'id': 'cgSgspJ2msm6clMCkdW9', 'name': 'Jessica (US Female)', 'description': 'Clear, professional'},
    'charlie': {'id': 'IKne3meq5aSn9XLyUdCD', 'name': 'Charlie (Energetic)', 'description': 'Upbeat, enthusiastic'},
    'sarah': {'id': 'EXAVITQu4vr4xnSDxMaL', 'name': 'Sarah (Confident)', 'description': 'Confident, warm'},
}


# ============ Deepgram STT ============
def get_deepgram_url():
    return (
        'wss://api.deepgram.com/v1/listen?'
        'model=nova-2&'
        'language=en&'
        'punctuate=true&'
        'interim_results=true&'
        'endpointing=300&'
        'encoding=linear16&'
        'sample_rate=16000&'
        'channels=1'
    )


# ============ OpenAI LLM ============
def generate_llm_response(user_text: str, conversation_history: list) -> str:
    """Generate response using OpenAI GPT-4o-mini"""
    start_time = time.time()

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_text})

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": conversation_history,
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            assistant_text = result['choices'][0]['message']['content']

            # Add assistant message to history
            conversation_history.append({"role": "assistant", "content": assistant_text})

            llm_time = (time.time() - start_time) * 1000
            log(f"LLM response in {llm_time:.0f}ms: {assistant_text[:50]}...")

            return assistant_text, llm_time
        else:
            log(f"LLM error: {response.status_code} - {response.text}")
            return "I'm having trouble thinking right now. Please try again.", 0
    except Exception as e:
        log(f"LLM exception: {e}")
        return "Sorry, I encountered an error. Please try again.", 0


# ============ ElevenLabs TTS ============
def generate_tts(text: str, voice_id: str = 'Zs2gGSc3xT4kRfIqS9R3') -> tuple:
    """Generate TTS audio using ElevenLabs streaming API"""
    start_time = time.time()
    first_chunk_time = None

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream'

    headers = {
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg'
    }

    payload = {
        'text': text,
        'model_id': 'eleven_turbo_v2',
        'voice_settings': {
            'stability': 0.5,
            'similarity_boost': 0.75,
            'style': 0.0,
            'use_speaker_boost': True
        },
        'optimize_streaming_latency': 3
    }

    try:
        audio_chunks = []
        response = requests.post(url, json=payload, headers=headers, stream=True)

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                audio_chunks.append(chunk)

        audio_data = b''.join(audio_chunks)

        ttfa = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0
        total_time = (time.time() - start_time) * 1000

        log(f"TTS generated in {total_time:.0f}ms (TTFA: {ttfa:.0f}ms)")

        return audio_data, ttfa, total_time
    except Exception as e:
        log(f"TTS error: {e}")
        return None, 0, 0


# ============ Voice Bot Session ============
class VoiceBotSession:
    """Manages a voice conversation session"""

    def __init__(self, sid):
        self.sid = sid
        self.audio_queue = Queue()
        self.ws = None
        self.running = False
        self.voice_id = VOICES['neha']['id']  # Default to Neha

        # Conversation state
        self.conversation_history = [
            {
                "role": "system",
                "content": """You are a friendly, helpful voice assistant named Neha.
Keep your responses concise - 1-2 sentences maximum for natural conversation.
Be warm and conversational, like talking to a friend.
If asked about yourself, you're an AI assistant powered by advanced language models."""
            }
        ]

        # SmartTurn state - accumulates interim transcriptions
        self.current_utterance = ""
        self.last_interim_time = None
        self.silence_threshold = 1.5  # seconds of silence to trigger response
        self.utterance_buffer = []

        # Timing
        self.user_started_speaking = None
        self.transcription_time = None

    def set_voice(self, voice_key):
        if voice_key in VOICES:
            self.voice_id = VOICES[voice_key]['id']
            log(f"[{self.sid[:8]}] Voice set to {voice_key}")

    def start(self):
        self.running = True
        url = get_deepgram_url()
        log(f"[{self.sid[:8]}] Connecting to Deepgram...")

        try:
            self.ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
            self.ws.connect(url, header=[f'Authorization: Token {DEEPGRAM_API_KEY}'])
            log(f"[{self.sid[:8]}] Connected to Deepgram!")

            # Start sender thread
            eventlet.spawn(self._sender_loop)

            # Run receiver in main greenlet
            self._receiver_loop()

        except Exception as e:
            log(f"[{self.sid[:8]}] Deepgram error: {e}")
            socketio.emit('error', {'message': str(e)}, to=self.sid)
            self.running = False

    def _sender_loop(self):
        """Send audio to Deepgram"""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is None:
                    break
                if self.ws and self.running:
                    self.ws.send_binary(audio_data)
            except Empty:
                eventlet.sleep(0)
            except Exception as e:
                log(f"[{self.sid[:8]}] Sender error: {e}")
                break

    def _receiver_loop(self):
        """Receive transcriptions from Deepgram and process with SmartTurn logic"""
        while self.running:
            try:
                if not self.ws:
                    break

                message = self.ws.recv()
                if not message:
                    eventlet.sleep(0)
                    continue

                result = json.loads(message)

                if result.get('type') == 'Results':
                    channel = result.get('channel', {})
                    alternatives = channel.get('alternatives', [])

                    if alternatives:
                        transcript = alternatives[0].get('transcript', '')
                        is_final = result.get('is_final', False)
                        speech_final = result.get('speech_final', False)

                        if transcript:
                            self.last_interim_time = time.time()

                            # Emit interim transcription to UI
                            socketio.emit('transcription', {
                                'transcript': transcript,
                                'is_final': is_final,
                                'speech_final': speech_final
                            }, to=self.sid)

                            # SmartTurn logic: accumulate until speech_final
                            if is_final:
                                self.utterance_buffer.append(transcript)

                                if speech_final:
                                    # User finished speaking - process complete utterance
                                    full_utterance = ' '.join(self.utterance_buffer)
                                    self.utterance_buffer = []

                                    if full_utterance.strip():
                                        log(f"[{self.sid[:8]}] USER: {full_utterance}")
                                        self._process_user_turn(full_utterance)

            except websocket.WebSocketConnectionClosedException:
                log(f"[{self.sid[:8]}] WebSocket closed")
                break
            except Exception as e:
                if self.running:
                    log(f"[{self.sid[:8]}] Receiver error: {e}")
                break

    def _process_user_turn(self, user_text: str):
        """Process a complete user turn through LLM and TTS"""
        e2e_start = time.time()

        # Notify UI that we're processing
        socketio.emit('bot_thinking', {'status': 'thinking'}, to=self.sid)

        # 1. Generate LLM response
        llm_start = time.time()
        response_text, llm_time = generate_llm_response(user_text, self.conversation_history)

        # 2. Generate TTS audio
        tts_start = time.time()
        audio_data, ttfa, tts_time = generate_tts(response_text, self.voice_id)

        # Calculate total latency
        e2e_time = (time.time() - e2e_start) * 1000

        # Determine latency status
        if e2e_time < 1000:
            status = 'excellent'
        elif e2e_time < 1500:
            status = 'good'
        elif e2e_time < 2000:
            status = 'acceptable'
        else:
            status = 'slow'

        # Send response to UI
        if audio_data:
            socketio.emit('bot_response', {
                'text': response_text,
                'audio': base64.b64encode(audio_data).decode('utf-8'),
                'latency': {
                    'llm_ms': round(llm_time),
                    'tts_ttfa_ms': round(ttfa),
                    'tts_total_ms': round(tts_time),
                    'e2e_ms': round(e2e_time),
                    'status': status
                }
            }, to=self.sid)

            log(f"[{self.sid[:8]}] BOT: {response_text[:50]}... (E2E: {e2e_time:.0f}ms)")
        else:
            socketio.emit('bot_response', {
                'text': response_text,
                'audio': None,
                'error': 'TTS failed'
            }, to=self.sid)

    def send_audio(self, audio_bytes):
        if self.running:
            self.audio_queue.put(audio_bytes)

    def stop(self):
        log(f"[{self.sid[:8]}] Stopping session...")
        self.running = False
        self.audio_queue.put(None)
        if self.ws:
            try:
                self.ws.close()
            except:
                pass


# ============ Routes ============
@app.route('/')
def index():
    return render_template('voice_bot.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files with correct MIME types"""
    from flask import send_from_directory, Response
    if filename.endswith('.js'):
        # Serve JS files with correct MIME type for AudioWorklet
        return send_from_directory('static', filename, mimetype='application/javascript')
    return send_from_directory('static', filename)


@app.route('/api/voices')
def get_voices():
    return jsonify(VOICES)


# ============ Socket.IO Handlers ============
@socketio.on('connect')
def handle_connect():
    log(f"=== CLIENT CONNECTED: {request.sid[:8]} ===")
    emit('connected', {'status': 'Connected to Voice Bot'})


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    log(f"=== CLIENT DISCONNECTED: {sid[:8]} ===")
    if sid in active_sessions:
        active_sessions[sid].stop()
        del active_sessions[sid]


@socketio.on('start_conversation')
def handle_start_conversation(data=None):
    sid = request.sid
    log(f"=== START CONVERSATION: {sid[:8]} ===")

    # Stop existing session if any
    if sid in active_sessions:
        active_sessions[sid].stop()

    # Create new session
    session = VoiceBotSession(sid)

    # Set voice if specified
    if data and 'voice' in data:
        session.set_voice(data['voice'])

    active_sessions[sid] = session

    # Start session in background
    eventlet.spawn(session.start)
    emit('conversation_started', {'status': 'Listening...'})


@socketio.on('audio_data')
def handle_audio_data(data):
    sid = request.sid
    if sid in active_sessions:
        try:
            audio_bytes = base64.b64decode(data['audio'])
            active_sessions[sid].send_audio(audio_bytes)
        except Exception as e:
            log(f"Audio error: {e}")


@socketio.on('set_voice')
def handle_set_voice(data):
    sid = request.sid
    if sid in active_sessions and 'voice' in data:
        active_sessions[sid].set_voice(data['voice'])
        emit('voice_changed', {'voice': data['voice']})


@socketio.on('stop_conversation')
def handle_stop_conversation():
    sid = request.sid
    log(f"=== STOP CONVERSATION: {sid[:8]} ===")
    if sid in active_sessions:
        active_sessions[sid].stop()
        del active_sessions[sid]
    emit('conversation_stopped', {'status': 'Stopped'})


# ============ Main ============
if __name__ == '__main__':
    print("=" * 60)
    print("Web Voice Bot with WebRTC Echo Cancellation + SmartTurn")
    print("=" * 60)
    print()
    print("Features:")
    print("  - Browser WebRTC echo cancellation (no headphones needed!)")
    print("  - Deepgram Nova-2 STT")
    print("  - OpenAI GPT-4o-mini LLM")
    print("  - ElevenLabs TTS (multiple voices)")
    print("  - SmartTurn for natural turn-taking")
    print("  - Real-time latency tracking")
    print()
    print("Open http://localhost:5002")
    print("=" * 60)

    socketio.run(app, host='0.0.0.0', port=5002, debug=False)
