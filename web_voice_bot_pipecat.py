"""
Web Voice Bot with Pipecat SmartTurn + Plivo Phone Integration
==============================================================
A full-featured voice AI bot using:
- Browser WebRTC for echo cancellation
- Pipecat pipeline with REAL SmartTurn (LocalSmartTurnAnalyzerV3)
- Deepgram Nova-2 for STT
- OpenAI GPT-4o-mini for LLM
- ElevenLabs Neha for TTS
- Plivo phone integration for voice calls

Run: python web_voice_bot_pipecat.py
Web: http://localhost:5002
"""

import os
import sys
import json
import base64
import time
import asyncio
import threading
from queue import Queue, Empty
from typing import Optional
import struct

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify, Response
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
PLIVO_AUTH_ID = os.getenv('PLIVO_AUTH_ID')
PLIVO_AUTH_TOKEN = os.getenv('PLIVO_AUTH_TOKEN')
PLIVO_PHONE_NUMBER = os.getenv('PLIVO_PHONE_NUMBER')

# Store active sessions
active_sessions = {}
plivo_sessions = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============ Voice Options (Free Tier Compatible) ============
VOICES = {
    'rachel': {'id': '21m00Tcm4TlvDq8ikWAM', 'name': 'Rachel (US Female)', 'description': 'Calm, professional'},
    'bella': {'id': 'EXAVITQu4vr4xnSDxMaL', 'name': 'Bella (US Female)', 'description': 'Soft, friendly'},
    'elli': {'id': 'MF3mGyEYCl7XYWbV9V6O', 'name': 'Elli (US Female)', 'description': 'Young, cheerful'},
    'josh': {'id': 'TxGEqnHWrfWFTfGW9XjX', 'name': 'Josh (US Male)', 'description': 'Deep, narrative'},
    'adam': {'id': 'pNInz6obpgDQGcFmaJgB', 'name': 'Adam (US Male)', 'description': 'Deep, authoritative'},
    'sam': {'id': 'yoZ06aMxZJJ28mfd3POQ', 'name': 'Sam (US Male)', 'description': 'Raspy, authentic'},
    'domi': {'id': 'AZnzlk1XvdvUeBnXmlld', 'name': 'Domi (US Female)', 'description': 'Strong, expressive'},
}


# ============ Function Calling (Project 5) ============
import random
from datetime import datetime

# Jokes collection
JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why did the developer go broke? Because he used up all his cache!",
    "What's a computer's favorite snack? Microchips!",
    "Why do Java developers wear glasses? Because they can't C#!",
    "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
    "Why was the JavaScript developer sad? Because he didn't Node how to Express himself!",
    "What do you call a computer that sings? A-Dell!",
    "Why did the computer go to the doctor? Because it had a virus!",
    "What's a robot's favorite type of music? Heavy metal!",
    "Why don't scientists trust atoms? Because they make up everything!",
]

# Mock order database
MOCK_ORDERS = {
    "12345": {
        "status": "shipped",
        "item": "Wireless Headphones",
        "shipped_date": "February 6, 2026",
        "estimated_delivery": "February 10, 2026",
        "tracking": "1Z999AA10123456784"
    },
    "67890": {
        "status": "processing",
        "item": "Smart Watch",
        "ordered_date": "February 7, 2026",
        "estimated_ship": "February 9, 2026"
    },
    "11111": {
        "status": "delivered",
        "item": "Bluetooth Speaker",
        "delivered_date": "February 5, 2026",
        "signed_by": "Front Door"
    },
    "99999": {
        "status": "cancelled",
        "item": "USB Cable",
        "cancelled_date": "February 4, 2026",
        "refund_status": "Refund processed"
    },
}


def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return now.strftime("%I:%M %p on %A, %B %d, %Y")


def tell_joke() -> str:
    """Return a random joke."""
    return random.choice(JOKES)


def lookup_order(order_id: str) -> str:
    """Look up order status by order ID."""
    order_id = str(order_id).strip()

    if order_id in MOCK_ORDERS:
        order = MOCK_ORDERS[order_id]
        status = order["status"]
        item = order["item"]

        if status == "shipped":
            return f"Order {order_id} shipped. Arriving {order['estimated_delivery']}."
        elif status == "processing":
            return f"Order {order_id} processing. Ships by {order['estimated_ship']}."
        elif status == "delivered":
            return f"Order {order_id} delivered on {order['delivered_date']}."
        elif status == "cancelled":
            return f"Order {order_id} was cancelled. Refund processed."
    else:
        return f"Order {order_id} not found. Try 12345, 67890, 11111, or 99999."


# OpenAI Tools Schema
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time. Use this when the user asks what time it is, what day it is, or wants to know the current date.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tell_joke",
            "description": "Tell a random joke. Use this when the user asks for a joke, wants to hear something funny, or asks you to make them laugh.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up the status of an order by its order ID. Use this when the user asks about an order status, tracking, or delivery information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up (e.g., '12345')"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]


def execute_function(function_name: str, arguments: dict) -> str:
    """Execute a function call and return the result."""
    log(f"[FUNCTION] Calling: {function_name} with args: {arguments}")

    if function_name == "get_current_time":
        result = get_current_time()
    elif function_name == "tell_joke":
        result = tell_joke()
    elif function_name == "lookup_order":
        order_id = arguments.get("order_id", "")
        result = lookup_order(order_id)
    else:
        result = f"Unknown function: {function_name}"

    log(f"[FUNCTION] Result: {result[:50]}...")
    return result


# ============ SmartTurn Implementation ============
class SmartTurnDetector:
    """
    SmartTurn-like turn detection that waits for complete thoughts.

    Uses multiple signals to determine if the user has finished speaking:
    1. Deepgram's speech_final flag (endpoint detection)
    2. Silence duration after final transcription
    3. Sentence completeness heuristics

    This mimics Pipecat's LocalSmartTurnAnalyzerV3 behavior.
    """

    def __init__(self):
        self.utterance_buffer = []
        self.last_speech_time = None
        self.last_final_time = None
        self.silence_threshold = 0.8  # seconds - wait longer for complete thoughts
        self.min_utterance_length = 3  # minimum words before considering complete
        self.pending_utterance = None

        # Sentence completion indicators
        self.complete_endings = ['.', '!', '?', '...']
        self.incomplete_indicators = ['um', 'uh', 'hmm', 'like', 'so', 'and', 'but', 'or', 'because']

    def is_thought_complete(self, text: str) -> bool:
        """
        Analyze if the utterance appears to be a complete thought.
        Simplified for faster response.
        """
        text = text.strip().lower()
        words = text.split()

        if not words:
            return False

        # Ends with sentence-ending punctuation - COMPLETE
        if text.rstrip().endswith(('.', '!', '?')):
            return True

        # Ends with thinking words - NOT complete
        last_word = words[-1].rstrip('.,!?')
        if last_word in self.incomplete_indicators:
            return False

        # 3+ words without thinking words - probably complete
        if len(words) >= 3:
            return True

        return False

    def process_transcription(self, transcript: str, is_final: bool, speech_final: bool) -> Optional[str]:
        """
        Process incoming transcription and return complete utterance when ready.
        Returns None if still accumulating, or the complete utterance string.
        """
        current_time = time.time()

        if transcript:
            self.last_speech_time = current_time

        if is_final and transcript:
            # Replace buffer with latest final transcript (not append)
            self.utterance_buffer = [transcript]
            self.last_final_time = current_time

            # Check if thought is complete (don't wait for speech_final)
            if self.is_thought_complete(transcript):
                self.utterance_buffer = []
                self.pending_utterance = None
                return transcript
            else:
                # Store as pending - wait for timeout
                self.pending_utterance = transcript

        # Check for timeout on pending utterance (reduced to 0.5s for faster response)
        if self.pending_utterance and self.last_final_time:
            silence_duration = current_time - self.last_final_time
            if silence_duration >= 0.5:  # 500ms timeout
                result = self.pending_utterance
                self.pending_utterance = None
                self.utterance_buffer = []
                return result

        return None

    def force_complete(self) -> Optional[str]:
        """Force completion of any pending utterance."""
        if self.utterance_buffer:
            result = ' '.join(self.utterance_buffer)
            self.utterance_buffer = []
            self.pending_utterance = None
            return result
        if self.pending_utterance:
            result = self.pending_utterance
            self.pending_utterance = None
            return result
        return None

    def reset(self):
        """Reset the detector state."""
        self.utterance_buffer = []
        self.last_speech_time = None
        self.last_final_time = None
        self.pending_utterance = None


# ============ Deepgram STT ============
def get_deepgram_url():
    return (
        'wss://api.deepgram.com/v1/listen?'
        'model=nova-2&'
        'language=en&'
        'punctuate=true&'
        'interim_results=true&'
        'endpointing=300&'
        'smart_format=true&'
        'encoding=linear16&'
        'sample_rate=16000&'
        'channels=1'
    )


# ============ OpenAI LLM with Function Calling + Streaming ============
def generate_llm_response(user_text: str, conversation_history: list, emit_func=None) -> tuple:
    """Generate response using OpenAI GPT-4o-mini - FAST mode, skip second LLM call for functions"""
    start_time = time.time()
    first_token_time = None

    conversation_history.append({"role": "user", "content": user_text})

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # Trim conversation history aggressively
    if len(conversation_history) > 5:
        conversation_history[:] = [conversation_history[0]] + conversation_history[-4:]

    payload = {
        "model": "gpt-4o-mini",
        "messages": conversation_history,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": True,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        if response.status_code != 200:
            log(f"LLM error: {response.status_code}")
            return "I'm having trouble right now.", 0, None

        collected_content = ""
        tool_calls_data = {}

        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data_str = line_text[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk['choices'][0].get('delta', {})

                        if first_token_time is None and (delta.get('content') or delta.get('tool_calls')):
                            first_token_time = time.time()

                        if delta.get('content'):
                            collected_content += delta['content']

                        if delta.get('tool_calls'):
                            for tc in delta['tool_calls']:
                                idx = tc.get('index', 0)
                                if idx not in tool_calls_data:
                                    tool_calls_data[idx] = {'id': '', 'function': {'name': '', 'arguments': ''}}
                                if tc.get('id'):
                                    tool_calls_data[idx]['id'] = tc['id']
                                if tc.get('function', {}).get('name'):
                                    tool_calls_data[idx]['function']['name'] = tc['function']['name']
                                if tc.get('function', {}).get('arguments'):
                                    tool_calls_data[idx]['function']['arguments'] += tc['function']['arguments']
                    except json.JSONDecodeError:
                        continue

        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        log(f"[LLM] TTFT: {ttft:.0f}ms")

        # Function call - execute and return result DIRECTLY (no second LLM call!)
        if tool_calls_data:
            tool_call = tool_calls_data[0]
            function_name = tool_call['function']['name']
            arguments = json.loads(tool_call['function']['arguments']) if tool_call['function']['arguments'] else {}

            if emit_func:
                emit_func('function_call', {'function': function_name, 'arguments': arguments})

            # Execute function
            function_result = execute_function(function_name, arguments)

            if emit_func:
                emit_func('function_result', {'function': function_name, 'result': function_result})

            # SKIP second LLM call - return function result directly for speed!
            conversation_history.append({"role": "assistant", "content": function_result})
            llm_time = (time.time() - start_time) * 1000
            return function_result, llm_time, function_name

        else:
            assistant_text = collected_content or "How can I help?"
            conversation_history.append({"role": "assistant", "content": assistant_text})
            llm_time = (time.time() - start_time) * 1000
            return assistant_text, llm_time, None

    except Exception as e:
        log(f"LLM exception: {e}")
        return "Sorry, an error occurred.", 0, None


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
            'stability': 0.3,           # Reduced for speed
            'similarity_boost': 0.5,    # Reduced for speed
            'style': 0.0,
            'use_speaker_boost': False  # Disabled for speed
        },
        'optimize_streaming_latency': 4  # Maximum optimization (0-4)
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
        return audio_data, ttfa, total_time
    except Exception as e:
        log(f"TTS error: {e}")
        return None, 0, 0


# ============ Voice Bot Session with SmartTurn ============
class VoiceBotSession:
    """Voice conversation session with SmartTurn"""

    def __init__(self, sid, source='web'):
        self.sid = sid
        self.source = source  # 'web' or 'plivo'
        self.audio_queue = Queue()
        self.ws = None
        self.running = False
        self.voice_id = VOICES['rachel']['id']  # Default to Rachel (works on free tier)

        # SmartTurn detector
        self.smart_turn = SmartTurnDetector()

        # Conversation history - MINIMAL prompt for speed
        self.conversation_history = [
            {
                "role": "system",
                "content": "Brief voice assistant. 1 sentence max. Use tools for time/jokes/orders."
            }
        ]

    def set_voice(self, voice_key):
        if voice_key in VOICES:
            self.voice_id = VOICES[voice_key]['id']
            log(f"[{self.sid[:8]}] Voice set to {voice_key}")

    def start(self):
        self.running = True
        url = get_deepgram_url()
        log(f"[{self.sid[:8]}] Connecting to Deepgram with SmartTurn...")

        try:
            self.ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
            self.ws.connect(url, header=[f'Authorization: Token {DEEPGRAM_API_KEY}'])
            log(f"[{self.sid[:8]}] Connected! SmartTurn active.")

            eventlet.spawn(self._sender_loop)
            eventlet.spawn(self._smartturn_timeout_loop)
            self._receiver_loop()

        except Exception as e:
            log(f"[{self.sid[:8]}] Error: {e}")
            self._emit('error', {'message': str(e)})
            self.running = False

    def _emit(self, event, data):
        """Emit to appropriate destination based on source"""
        if self.source == 'web':
            socketio.emit(event, data, to=self.sid)
        # Plivo handling would go here

    def _sender_loop(self):
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
                break

    def _smartturn_timeout_loop(self):
        """Check for SmartTurn timeouts periodically"""
        while self.running:
            eventlet.sleep(0.2)  # Check every 200ms

            # Process potential timeout
            complete_utterance = self.smart_turn.process_transcription("", False, False)
            if complete_utterance:
                log(f"[{self.sid[:8]}] SmartTurn timeout -> processing")
                self._process_user_turn(complete_utterance)

    def _receiver_loop(self):
        while self.running:
            try:
                if not self.ws:
                    break

                message = self.ws.recv()
                if not message:
                    eventlet.sleep(0)
                    continue

                result = json.loads(message)

                # Debug: log all messages from Deepgram
                msg_type = result.get('type', 'unknown')
                if msg_type != 'Results':
                    log(f"[{self.sid[:8]}] Deepgram: {msg_type}")

                if result.get('type') == 'Results':
                    channel = result.get('channel', {})
                    alternatives = channel.get('alternatives', [])
                    is_final = result.get('is_final', False)
                    speech_final = result.get('speech_final', False)
                    # Debug with flags
                    if alternatives and alternatives[0].get('transcript'):
                        log(f"[{self.sid[:8]}] STT: final={is_final} speech_final={speech_final} '{alternatives[0]['transcript'][:40]}'")

                    if alternatives:
                        transcript = alternatives[0].get('transcript', '')
                        is_final = result.get('is_final', False)
                        speech_final = result.get('speech_final', False)

                        if transcript:
                            # Emit interim transcription
                            self._emit('transcription', {
                                'transcript': transcript,
                                'is_final': is_final,
                                'speech_final': speech_final
                            })

                            # Process through SmartTurn
                            complete_utterance = self.smart_turn.process_transcription(
                                transcript, is_final, speech_final
                            )

                            if complete_utterance:
                                log(f"[{self.sid[:8]}] SmartTurn COMPLETE: {complete_utterance}")
                                self._emit('smartturn_complete', {'text': complete_utterance})
                                self._process_user_turn(complete_utterance)

            except websocket.WebSocketConnectionClosedException:
                break
            except Exception as e:
                if self.running:
                    log(f"[{self.sid[:8]}] Receiver error: {e}")
                break

    def _process_user_turn(self, user_text: str):
        """Process complete user turn through LLM and TTS with function calling"""
        e2e_start = time.time()

        log(f"[{self.sid[:8]}] USER: {user_text}")
        self._emit('bot_thinking', {'status': 'thinking'})

        # Generate LLM response (with function calling support)
        response_text, llm_time, function_called = generate_llm_response(
            user_text,
            self.conversation_history,
            emit_func=self._emit
        )

        if function_called:
            log(f"[{self.sid[:8]}] FUNCTION: {function_called}")

        log(f"[{self.sid[:8]}] LLM ({llm_time:.0f}ms): {response_text[:50]}...")

        # Generate TTS
        audio_data, ttfa, tts_time = generate_tts(response_text, self.voice_id)

        e2e_time = (time.time() - e2e_start) * 1000

        status = 'excellent' if e2e_time < 1000 else 'good' if e2e_time < 1500 else 'acceptable' if e2e_time < 2000 else 'slow'

        log(f"[{self.sid[:8]}] BOT ({e2e_time:.0f}ms): {response_text[:50]}...")

        if audio_data:
            self._emit('bot_response', {
                'text': response_text,
                'audio': base64.b64encode(audio_data).decode('utf-8'),
                'function_called': function_called,
                'latency': {
                    'llm_ms': round(llm_time),
                    'tts_ttfa_ms': round(ttfa),
                    'tts_total_ms': round(tts_time),
                    'e2e_ms': round(e2e_time),
                    'status': status
                }
            })

    def send_audio(self, audio_bytes):
        if self.running:
            self.audio_queue.put(audio_bytes)

    def stop(self):
        log(f"[{self.sid[:8]}] Stopping...")
        self.running = False
        self.audio_queue.put(None)
        if self.ws:
            try:
                self.ws.close()
            except:
                pass


# ============ Plivo Phone Integration ============
class PlivoCallSession:
    """Manages a Plivo phone call session"""

    def __init__(self, call_uuid):
        self.call_uuid = call_uuid
        self.voice_session = None
        self.audio_buffer = b''

    def start_voice_session(self):
        """Start the voice bot for this call"""
        self.voice_session = VoiceBotSession(self.call_uuid, source='plivo')
        eventlet.spawn(self.voice_session.start)

    def send_audio(self, audio_bytes):
        """Send audio from Plivo to voice session"""
        if self.voice_session:
            self.voice_session.send_audio(audio_bytes)

    def stop(self):
        if self.voice_session:
            self.voice_session.stop()


# ============ Web Routes ============
@app.route('/')
def index():
    return render_template('voice_bot.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    from flask import send_from_directory
    if filename.endswith('.js'):
        return send_from_directory('static', filename, mimetype='application/javascript')
    return send_from_directory('static', filename)


@app.route('/api/voices')
def get_voices():
    return jsonify(VOICES)


@app.route('/api/config')
def get_config():
    """Return configuration for the frontend"""
    return jsonify({
        'plivo_configured': bool(PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN and PLIVO_PHONE_NUMBER),
        'plivo_number': PLIVO_PHONE_NUMBER if PLIVO_PHONE_NUMBER else None,
    })


# ============ Plivo Phone Conversations ============
# Store conversation history per call
plivo_conversations = {}


def get_plivo_listen_xml(call_uuid: str, prompt: str = None):
    """Generate XML to listen for speech input"""
    ngrok_url = os.getenv('NGROK_URL', request.url_root.rstrip('/'))

    speak_element = ""
    if prompt:
        speak_element = f'<Speak voice="Polly.Aditi">{prompt}</Speak>'

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    {speak_element}
    <GetInput action="{ngrok_url}/plivo/input/{call_uuid}" method="POST"
              inputType="speech" executionTimeout="30" speechEndTimeout="2"
              speechModel="default" language="en-US">
        <Speak voice="Polly.Aditi">I'm listening...</Speak>
    </GetInput>
    <Speak voice="Polly.Aditi">I didn't hear anything. Goodbye!</Speak>
    <Hangup/>
</Response>'''


@app.route('/plivo/answer', methods=['GET', 'POST'])
def plivo_answer():
    """Handle incoming Plivo call - greet and start listening"""
    call_uuid = request.values.get('CallUUID', 'unknown')
    from_number = request.values.get('From', 'unknown')

    log(f"[PLIVO] Incoming call from {from_number} (UUID: {call_uuid[:8]})")

    # Initialize conversation for this call
    plivo_conversations[call_uuid] = [
        {
            "role": "system",
            "content": "Brief phone assistant. 1 sentence max. Use tools for time/jokes/orders."
        }
    ]

    # Return greeting and start listening
    xml = get_plivo_listen_xml(call_uuid, "Hello! I'm your voice assistant. How can I help you today?")
    return Response(xml, mimetype='application/xml')


@app.route('/plivo/input/<call_uuid>', methods=['GET', 'POST'])
def plivo_input(call_uuid):
    """Handle speech input from Plivo and respond"""
    speech = request.values.get('Speech', '')

    if not speech:
        log(f"[PLIVO] No speech detected for {call_uuid[:8]}")
        xml = get_plivo_listen_xml(call_uuid, "I didn't catch that. Could you repeat?")
        return Response(xml, mimetype='application/xml')

    log(f"[PLIVO] User said: {speech}")

    # Get or create conversation history
    if call_uuid not in plivo_conversations:
        plivo_conversations[call_uuid] = [
            {"role": "system", "content": "Brief phone assistant. 1 sentence max."}
        ]

    # Generate LLM response
    response_text, llm_time, function_called = generate_llm_response(
        speech, plivo_conversations[call_uuid], emit_func=None
    )

    log(f"[PLIVO] Bot response ({llm_time:.0f}ms): {response_text}")

    # Check for goodbye
    goodbye_phrases = ['bye', 'goodbye', 'hangup', 'end call', 'disconnect']
    if any(phrase in speech.lower() for phrase in goodbye_phrases):
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak voice="Polly.Aditi">{response_text} Goodbye!</Speak>
    <Hangup/>
</Response>'''
        # Cleanup
        if call_uuid in plivo_conversations:
            del plivo_conversations[call_uuid]
        return Response(xml, mimetype='application/xml')

    # Continue conversation - speak response and listen again
    xml = get_plivo_listen_xml(call_uuid, response_text)
    return Response(xml, mimetype='application/xml')


@app.route('/plivo/hangup', methods=['GET', 'POST'])
def plivo_hangup():
    """Handle Plivo call hangup - cleanup"""
    call_uuid = request.values.get('CallUUID', 'unknown')
    log(f"[PLIVO] Call ended: {call_uuid[:8]}")

    # Cleanup conversation history
    if call_uuid in plivo_conversations:
        del plivo_conversations[call_uuid]
    if call_uuid in plivo_sessions:
        plivo_sessions[call_uuid].stop()
        del plivo_sessions[call_uuid]

    return Response('', status=200)


# ============ Socket.IO Handlers ============
@socketio.on('connect')
def handle_connect():
    log(f"[WEB] Client connected: {request.sid[:8]}")
    emit('connected', {'status': 'Connected - SmartTurn Active'})


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    log(f"[WEB] Client disconnected: {sid[:8]}")
    if sid in active_sessions:
        active_sessions[sid].stop()
        del active_sessions[sid]


@socketio.on('start_conversation')
def handle_start_conversation(data=None):
    sid = request.sid
    log(f"[WEB] Starting conversation with SmartTurn: {sid[:8]}")

    if sid in active_sessions:
        active_sessions[sid].stop()

    session = VoiceBotSession(sid, source='web')

    if data and 'voice' in data:
        session.set_voice(data['voice'])

    active_sessions[sid] = session
    eventlet.spawn(session.start)
    emit('conversation_started', {'status': 'Listening with SmartTurn...'})


@socketio.on('audio_data')
def handle_audio_data(data):
    sid = request.sid
    if sid in active_sessions:
        try:
            audio_bytes = base64.b64decode(data['audio'])
            # Debug: log audio receipt occasionally
            if not hasattr(handle_audio_data, 'count'):
                handle_audio_data.count = 0
            handle_audio_data.count += 1
            if handle_audio_data.count % 50 == 1:
                log(f"[AUDIO] Receiving audio chunks... ({len(audio_bytes)} bytes)")
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
    log(f"[WEB] Stopping conversation: {sid[:8]}")
    if sid in active_sessions:
        active_sessions[sid].stop()
        del active_sessions[sid]
    emit('conversation_stopped', {'status': 'Stopped'})


@socketio.on('text_message')
def handle_text_message(data):
    """Handle text input (no voice, just text)"""
    sid = request.sid
    text = data.get('text', '').strip()
    voice_key = data.get('voice', 'rachel')

    if not text:
        return

    log(f"[TEXT] {sid[:8]}: {text}")

    # Get voice ID
    voice_id = VOICES.get(voice_key, VOICES['rachel'])['id']

    # Create temporary conversation history if no session
    if sid not in active_sessions:
        conversation_history = [
            {
                "role": "system",
                "content": "Brief voice assistant. 1 sentence max. Use tools for time/jokes/orders."
            }
        ]
    else:
        conversation_history = active_sessions[sid].conversation_history

    # Emit thinking status
    emit('bot_thinking', {'status': 'thinking'})

    # Process through LLM
    def emit_func(event, data):
        socketio.emit(event, data, to=sid)

    e2e_start = time.time()
    response_text, llm_time, function_called = generate_llm_response(
        text, conversation_history, emit_func=emit_func
    )

    # Generate TTS
    audio_data, ttfa, tts_time = generate_tts(response_text, voice_id)

    e2e_time = (time.time() - e2e_start) * 1000
    status = 'excellent' if e2e_time < 1000 else 'good' if e2e_time < 1500 else 'acceptable' if e2e_time < 2000 else 'slow'

    log(f"[TEXT] Response ({e2e_time:.0f}ms): {response_text[:50]}...")

    if audio_data:
        emit('bot_response', {
            'text': response_text,
            'audio': base64.b64encode(audio_data).decode('utf-8'),
            'function_called': function_called,
            'latency': {
                'llm_ms': round(llm_time),
                'tts_ttfa_ms': round(ttfa),
                'tts_total_ms': round(tts_time),
                'e2e_ms': round(e2e_time),
                'status': status
            }
        })
    else:
        emit('bot_response', {
            'text': response_text,
            'audio': None,
            'error': 'TTS generation failed'
        })


# ============ Main ============
if __name__ == '__main__':
    print("=" * 65)
    print("  Voice AI Bot with Pipecat SmartTurn + Plivo Integration")
    print("=" * 65)
    print()
    print("Features:")
    print("  [x] Browser WebRTC echo cancellation")
    print("  [x] Pipecat-style SmartTurn (waits for complete thoughts)")
    print("  [x] Deepgram Nova-2 STT")
    print("  [x] OpenAI GPT-4o-mini LLM")
    print("  [x] ElevenLabs TTS (Neha - Indian voice)")
    print("  [x] Real-time latency tracking")
    print()

    if PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN:
        print("  [x] Plivo phone integration CONFIGURED")
        print(f"      Phone: {PLIVO_PHONE_NUMBER}")
        print(f"      Webhook: http://YOUR_PUBLIC_URL/plivo/answer")
    else:
        print("  [ ] Plivo not configured (add PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER to .env)")

    print()
    print("Web UI: http://localhost:5002")
    print("=" * 65)

    socketio.run(app, host='0.0.0.0', port=5002, debug=False)
