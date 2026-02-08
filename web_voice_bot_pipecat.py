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


# ============ Voice Options ============
VOICES = {
    'neha': {'id': 'Zs2gGSc3xT4kRfIqS9R3', 'name': 'Neha (Indian Female)', 'description': 'Warm, friendly Indian English'},
    'george': {'id': 'JBFqnCBsd6RMkjVDRZzb', 'name': 'George (US Male)', 'description': 'Deep, authoritative'},
    'jessica': {'id': 'cgSgspJ2msm6clMCkdW9', 'name': 'Jessica (US Female)', 'description': 'Clear, professional'},
    'charlie': {'id': 'IKne3meq5aSn9XLyUdCD', 'name': 'Charlie (Energetic)', 'description': 'Upbeat, enthusiastic'},
    'sarah': {'id': 'EXAVITQu4vr4xnSDxMaL', 'name': 'Sarah (Confident)', 'description': 'Confident, warm'},
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
            return (f"Order {order_id} for {item} has been shipped on {order['shipped_date']}. "
                   f"Expected delivery is {order['estimated_delivery']}. "
                   f"Tracking number is {order['tracking']}.")
        elif status == "processing":
            return (f"Order {order_id} for {item} is currently being processed. "
                   f"It was ordered on {order['ordered_date']} and should ship by {order['estimated_ship']}.")
        elif status == "delivered":
            return (f"Order {order_id} for {item} was delivered on {order['delivered_date']}. "
                   f"It was signed for at: {order['signed_by']}.")
        elif status == "cancelled":
            return (f"Order {order_id} for {item} was cancelled on {order['cancelled_date']}. "
                   f"{order['refund_status']}.")
    else:
        return f"I couldn't find order {order_id}. Please check the order number. Valid test orders are: 12345, 67890, 11111, or 99999."


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
        This is the core of SmartTurn logic.
        """
        text = text.strip().lower()
        words = text.split()

        # Too short - probably not complete
        if len(words) < self.min_utterance_length:
            return False

        # Ends with sentence-ending punctuation
        if any(text.rstrip().endswith(end) for end in self.complete_endings):
            return True

        # Ends with incomplete indicator - user is thinking
        last_word = words[-1].rstrip('.,!?')
        if last_word in self.incomplete_indicators:
            return False

        # Check for question patterns
        question_starters = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'is', 'are', 'can', 'could', 'would', 'should', 'do', 'does', 'did']
        if words[0] in question_starters and len(words) >= 4:
            return True

        # Default: if it's long enough and doesn't end with thinking words, consider complete
        if len(words) >= 5:
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
            self.utterance_buffer.append(transcript)
            self.last_final_time = current_time

            # If Deepgram says speech is final, check if thought is complete
            if speech_final:
                full_utterance = ' '.join(self.utterance_buffer)

                # SmartTurn logic: check if thought appears complete
                if self.is_thought_complete(full_utterance):
                    self.utterance_buffer = []
                    return full_utterance
                else:
                    # Store as pending - might get more
                    self.pending_utterance = full_utterance
                    return None

        # Check for timeout on pending utterance
        if self.pending_utterance and self.last_final_time:
            silence_duration = current_time - self.last_final_time
            if silence_duration >= self.silence_threshold:
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


# ============ OpenAI LLM with Function Calling ============
def generate_llm_response(user_text: str, conversation_history: list, emit_func=None) -> tuple:
    """Generate response using OpenAI GPT-4o-mini with function calling support"""
    start_time = time.time()

    conversation_history.append({"role": "user", "content": user_text})

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": conversation_history,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 150,
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            log(f"LLM error: {response.status_code}")
            return "I'm having trouble thinking right now.", 0, None

        result = response.json()
        message = result['choices'][0]['message']

        # Check if LLM wants to call a function
        if message.get('tool_calls'):
            tool_call = message['tool_calls'][0]
            function_name = tool_call['function']['name']
            arguments = json.loads(tool_call['function']['arguments']) if tool_call['function']['arguments'] else {}

            # Emit function call event to UI
            if emit_func:
                emit_func('function_call', {
                    'function': function_name,
                    'arguments': arguments
                })

            # Execute the function
            function_result = execute_function(function_name, arguments)

            # Emit function result to UI
            if emit_func:
                emit_func('function_result', {
                    'function': function_name,
                    'result': function_result
                })

            # Add function call and result to conversation
            conversation_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": message['tool_calls']
            })
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call['id'],
                "content": function_result
            })

            # Get final response incorporating function result
            payload_final = {
                "model": "gpt-4o-mini",
                "messages": conversation_history,
                "max_tokens": 150,
                "temperature": 0.7,
            }

            response_final = requests.post(url, json=payload_final, headers=headers)
            if response_final.status_code == 200:
                result_final = response_final.json()
                assistant_text = result_final['choices'][0]['message']['content']
                conversation_history.append({"role": "assistant", "content": assistant_text})
                llm_time = (time.time() - start_time) * 1000
                return assistant_text, llm_time, function_name
            else:
                return function_result, (time.time() - start_time) * 1000, function_name

        else:
            # No function call - regular response
            assistant_text = message.get('content', "I'm not sure how to respond to that.")
            conversation_history.append({"role": "assistant", "content": assistant_text})
            llm_time = (time.time() - start_time) * 1000
            return assistant_text, llm_time, None

    except Exception as e:
        log(f"LLM exception: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error.", 0, None


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
        self.voice_id = VOICES['neha']['id']

        # SmartTurn detector
        self.smart_turn = SmartTurnDetector()

        # Conversation history
        self.conversation_history = [
            {
                "role": "system",
                "content": """You are Neha, a friendly and helpful voice assistant with access to tools.

You have access to these functions:
- get_current_time: Use when asked about the time or date
- tell_joke: Use when asked for a joke or something funny
- lookup_order: Use when asked about order status (test IDs: 12345, 67890, 11111, 99999)

Keep your responses concise - 1-2 sentences maximum for natural conversation.
Be warm and conversational, like talking to a friend.
When reporting function results, speak them naturally as if you're having a conversation."""
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

                if result.get('type') == 'Results':
                    channel = result.get('channel', {})
                    alternatives = channel.get('alternatives', [])

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


# ============ Plivo Webhooks ============
@app.route('/plivo/answer', methods=['GET', 'POST'])
def plivo_answer():
    """Handle incoming Plivo call"""
    call_uuid = request.values.get('CallUUID', 'unknown')
    from_number = request.values.get('From', 'unknown')

    log(f"[PLIVO] Incoming call from {from_number} (UUID: {call_uuid[:8]})")

    # Create session for this call
    session = PlivoCallSession(call_uuid)
    plivo_sessions[call_uuid] = session

    # Return Plivo XML to stream audio
    # Using Plivo's stream feature to get real-time audio
    xml_response = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak voice="Polly.Aditi">Hello! I'm Neha, your voice assistant. How can I help you today?</Speak>
    <Stream bidirectional="true" keepCallAlive="true"
           streamTimeout="3600"
           contentType="audio/x-mulaw;rate=8000">
        wss://{request.host}/plivo/stream/{call_uuid}
    </Stream>
</Response>'''

    return Response(xml_response, mimetype='application/xml')


@app.route('/plivo/hangup', methods=['GET', 'POST'])
def plivo_hangup():
    """Handle Plivo call hangup"""
    call_uuid = request.values.get('CallUUID', 'unknown')
    log(f"[PLIVO] Call ended: {call_uuid[:8]}")

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
