"""
Voice Bot using Real Pipecat Framework
=======================================
- VAD (Voice Activity Detection) for silence start/stop
- SmartTurn (LocalSmartTurnAnalyzerV3) for additional check after VAD stop
- Deepgram Nova 3 for STT (final transcript only, not streaming)
- OpenAI GPT-4o-mini for LLM with function calling
- ElevenLabs for TTS
- Proper latency tracking: VAD/SmartTurn + LLM TTFT + TTS TTFS

Run: python pipecat_voice_bot.py
Web: http://localhost:5002
"""

import os
import sys
import json
import base64
import time
import asyncio
from datetime import datetime
from typing import Optional
import random

from dotenv import load_dotenv
load_dotenv(override=True)

# FastAPI and WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Pipecat imports
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import (
    Frame, AudioRawFrame, TextFrame, TranscriptionFrame,
    LLMMessagesFrame, TTSAudioRawFrame, EndFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# Logging
from loguru import logger

# API Keys
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# FastAPI app
app = FastAPI(title="Pipecat Voice Bot")

# ============ Function Calling Tools ============
JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why did the developer go broke? Because he used up all his cache!",
    "What's a computer's favorite snack? Microchips!",
]

MOCK_ORDERS = {
    "12345": {"status": "shipped", "item": "Wireless Headphones", "delivery": "February 10, 2026"},
    "67890": {"status": "processing", "item": "Smart Watch", "ship_by": "February 9, 2026"},
    "11111": {"status": "delivered", "item": "Bluetooth Speaker", "delivered": "February 5, 2026"},
}


def get_current_time() -> str:
    return datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")


def tell_joke() -> str:
    return random.choice(JOKES)


def lookup_order(order_id: str) -> str:
    order_id = str(order_id).strip()
    if order_id in MOCK_ORDERS:
        o = MOCK_ORDERS[order_id]
        return f"Order {order_id}: {o['item']} - {o['status']}"
    return f"Order {order_id} not found. Try: 12345, 67890, 11111"


TOOLS = [
    {"type": "function", "function": {"name": "get_current_time", "description": "Get current date/time", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "tell_joke", "description": "Tell a joke", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "lookup_order", "description": "Look up order status", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
]


def execute_function(name: str, args: dict) -> str:
    if name == "get_current_time":
        return get_current_time()
    elif name == "tell_joke":
        return tell_joke()
    elif name == "lookup_order":
        return lookup_order(args.get("order_id", ""))
    return f"Unknown function: {name}"


# ============ Latency Tracker ============
class LatencyTracker:
    """Track latency at each stage of the pipeline"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.vad_stop_time = None          # When VAD detected silence
        self.smart_turn_time = None        # When SmartTurn confirmed end
        self.transcript_time = None        # When final transcript received
        self.llm_start_time = None         # When LLM request started
        self.llm_ttft_time = None          # When first LLM token received
        self.tts_start_time = None         # When TTS request started
        self.tts_ttfs_time = None          # When first TTS audio received (Time To First Speech)

    def mark_vad_stop(self):
        self.vad_stop_time = time.time()
        logger.info(f"[LATENCY] VAD stop detected")

    def mark_smart_turn(self):
        self.smart_turn_time = time.time()
        if self.vad_stop_time:
            delta = (self.smart_turn_time - self.vad_stop_time) * 1000
            logger.info(f"[LATENCY] SmartTurn confirmed: +{delta:.0f}ms after VAD")

    def mark_transcript(self):
        self.transcript_time = time.time()
        logger.info(f"[LATENCY] Final transcript received")

    def mark_llm_start(self):
        self.llm_start_time = time.time()

    def mark_llm_ttft(self):
        self.llm_ttft_time = time.time()
        if self.llm_start_time:
            ttft = (self.llm_ttft_time - self.llm_start_time) * 1000
            logger.info(f"[LATENCY] LLM TTFT: {ttft:.0f}ms")

    def mark_tts_start(self):
        self.tts_start_time = time.time()

    def mark_tts_ttfs(self):
        self.tts_ttfs_time = time.time()
        if self.tts_start_time:
            ttfs = (self.tts_ttfs_time - self.tts_start_time) * 1000
            logger.info(f"[LATENCY] TTS TTFS: {ttfs:.0f}ms")

    def get_summary(self) -> dict:
        """Get latency summary"""
        result = {}

        # VAD/SmartTurn to transcript
        if self.smart_turn_time and self.vad_stop_time:
            result['vad_to_smartturn_ms'] = int((self.smart_turn_time - self.vad_stop_time) * 1000)

        # LLM TTFT
        if self.llm_ttft_time and self.llm_start_time:
            result['llm_ttft_ms'] = int((self.llm_ttft_time - self.llm_start_time) * 1000)

        # TTS TTFS
        if self.tts_ttfs_time and self.tts_start_time:
            result['tts_ttfs_ms'] = int((self.tts_ttfs_time - self.tts_start_time) * 1000)

        # End-to-end: VAD/SmartTurn + LLM TTFT + TTS TTFS
        start = self.smart_turn_time or self.vad_stop_time
        end = self.tts_ttfs_time
        if start and end:
            result['e2e_ms'] = int((end - start) * 1000)

        return result


# ============ Custom Processors for Latency Tracking ============
class LatencyTrackingSTT(FrameProcessor):
    """Wraps STT to track when final transcript is received"""

    def __init__(self, tracker: LatencyTracker, websocket: WebSocket):
        super().__init__()
        self.tracker = tracker
        self.websocket = websocket
        self.accumulated_transcript = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Final transcript received
            self.tracker.mark_transcript()
            logger.info(f"[STT] Final transcript: {frame.text}")

            # Send to websocket
            await self.websocket.send_json({
                "type": "transcript",
                "text": frame.text,
                "is_final": True
            })

        await self.push_frame(frame, direction)


class LatencyTrackingLLM(FrameProcessor):
    """Wraps LLM to track TTFT"""

    def __init__(self, tracker: LatencyTracker, websocket: WebSocket):
        super().__init__()
        self.tracker = tracker
        self.websocket = websocket
        self.first_token_received = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMMessagesFrame):
            # LLM request starting
            self.tracker.mark_llm_start()
            self.first_token_received = False
            logger.info(f"[LLM] Processing messages...")

        if isinstance(frame, TextFrame) and not self.first_token_received:
            # First token received
            self.tracker.mark_llm_ttft()
            self.first_token_received = True

            await self.websocket.send_json({
                "type": "llm_response",
                "text": frame.text
            })

        await self.push_frame(frame, direction)


class LatencyTrackingTTS(FrameProcessor):
    """Wraps TTS to track TTFS (Time To First Speech)"""

    def __init__(self, tracker: LatencyTracker, websocket: WebSocket):
        super().__init__()
        self.tracker = tracker
        self.websocket = websocket
        self.first_audio_received = False
        self.audio_buffer = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            # TTS request starting
            self.tracker.mark_tts_start()
            self.first_audio_received = False
            self.audio_buffer = bytearray()

        if isinstance(frame, TTSAudioRawFrame):
            if not self.first_audio_received:
                # First audio chunk received
                self.tracker.mark_tts_ttfs()
                self.first_audio_received = True

            # Accumulate audio
            self.audio_buffer.extend(frame.audio)

        if isinstance(frame, EndFrame) and self.audio_buffer:
            # Send accumulated audio to websocket
            latency = self.tracker.get_summary()
            await self.websocket.send_json({
                "type": "audio",
                "audio": base64.b64encode(bytes(self.audio_buffer)).decode(),
                "latency": latency
            })
            logger.info(f"[LATENCY] Summary: {latency}")
            self.audio_buffer = bytearray()

        await self.push_frame(frame, direction)


# ============ Voice Bot Session ============
class VoiceBotSession:
    """Manages a single voice conversation using Pipecat"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.tracker = LatencyTracker()
        self.pipeline = None
        self.task = None
        self.running = False

        # Conversation history
        self.messages = [
            {"role": "system", "content": "You are a helpful voice assistant. Keep responses to 1-2 sentences. Use tools when appropriate."}
        ]

    async def start(self):
        """Start the Pipecat pipeline"""
        self.running = True

        # VAD with Silero (for silence detection)
        vad = SileroVADAnalyzer(
            sample_rate=16000,
            num_channels=1,
            params=SileroVADAnalyzer.VADParams(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=300,  # 300ms silence triggers VAD stop
            )
        )

        # SmartTurn analyzer (additional check after VAD stop)
        smart_turn = LocalSmartTurnAnalyzerV3(
            sample_rate=16000,
            num_channels=1,
        )

        # Deepgram STT - Nova 3 (NOT Nova 2!)
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            model="nova-3",  # Using Nova 3 as requested
            language="en",
            punctuate=True,
            interim_results=False,  # Only final transcripts, not streaming
            endpointing=300,
            smart_format=True,
        )

        # OpenAI LLM with function calling
        llm = OpenAILLMService(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            params=OpenAILLMService.LLMParams(
                max_tokens=100,
                temperature=0.3,
            ),
            tools=TOOLS,
        )

        # ElevenLabs TTS
        tts = ElevenLabsTTSService(
            api_key=ELEVENLABS_API_KEY,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
            model="eleven_turbo_v2",
            params=ElevenLabsTTSService.TTSParams(
                optimize_streaming_latency=4,
            )
        )

        # Latency tracking processors
        stt_tracker = LatencyTrackingSTT(self.tracker, self.websocket)
        llm_tracker = LatencyTrackingLLM(self.tracker, self.websocket)
        tts_tracker = LatencyTrackingTTS(self.tracker, self.websocket)

        # Build pipeline: VAD -> SmartTurn -> STT -> LLM -> TTS
        self.pipeline = Pipeline([
            vad,
            smart_turn,
            stt,
            stt_tracker,
            llm,
            llm_tracker,
            tts,
            tts_tracker,
        ])

        # Create task
        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )

        logger.info("[PIPECAT] Pipeline started with VAD + SmartTurn + Nova-3")

        await self.websocket.send_json({
            "type": "status",
            "status": "Pipeline started with VAD + SmartTurn + Deepgram Nova-3"
        })

    async def process_audio(self, audio_bytes: bytes):
        """Process incoming audio through the pipeline"""
        if not self.running or not self.task:
            return

        # Create audio frame and push to pipeline
        frame = AudioRawFrame(
            audio=audio_bytes,
            sample_rate=16000,
            num_channels=1
        )

        await self.task.queue_frame(frame)

    async def stop(self):
        """Stop the pipeline"""
        self.running = False
        if self.task:
            await self.task.queue_frame(EndFrame())
        logger.info("[PIPECAT] Pipeline stopped")


# ============ WebSocket Handler ============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Client connected")

    session = VoiceBotSession(websocket)

    try:
        await session.start()

        while True:
            data = await websocket.receive_json()

            if data.get("type") == "audio":
                # Decode and process audio
                audio_bytes = base64.b64decode(data["audio"])
                await session.process_audio(audio_bytes)

            elif data.get("type") == "stop":
                break

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
    finally:
        await session.stop()


# ============ HTML Page ============
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pipecat Voice Bot</title>
    <style>
        body { font-family: Arial; background: #1a1a2e; color: white; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #667eea; }
        .status { padding: 10px; background: #333; border-radius: 8px; margin: 10px 0; }
        .btn { padding: 15px 30px; font-size: 18px; border: none; border-radius: 8px; cursor: pointer; margin: 5px; }
        .btn-start { background: #4ade80; color: black; }
        .btn-stop { background: #f87171; color: white; }
        .latency { background: #333; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .latency-item { display: inline-block; margin: 10px; text-align: center; }
        .latency-value { font-size: 24px; font-weight: bold; color: #4ade80; }
        .transcript { background: #222; padding: 15px; border-radius: 8px; min-height: 100px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pipecat Voice Bot</h1>
        <p>Using: VAD + SmartTurn + Deepgram Nova-3 + GPT-4o-mini + ElevenLabs</p>

        <div class="status" id="status">Disconnected</div>

        <button class="btn btn-start" onclick="start()">Start</button>
        <button class="btn btn-stop" onclick="stop()">Stop</button>

        <div class="transcript" id="transcript">Transcripts will appear here...</div>

        <div class="latency" id="latency">
            <h3>Latency Breakdown</h3>
            <div class="latency-item">
                <div class="latency-value" id="vad-smartturn">-</div>
                <div>VAD/SmartTurn</div>
            </div>
            <div class="latency-item">
                <div class="latency-value" id="llm-ttft">-</div>
                <div>LLM TTFT</div>
            </div>
            <div class="latency-item">
                <div class="latency-value" id="tts-ttfs">-</div>
                <div>TTS TTFS</div>
            </div>
            <div class="latency-item">
                <div class="latency-value" id="e2e" style="color: #fbbf24;">-</div>
                <div>End-to-End</div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let mediaStream = null;
        let audioContext = null;
        let processor = null;

        function start() {
            ws = new WebSocket('ws://localhost:5002/ws');

            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected - Starting mic...';
                startMic();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'status') {
                    document.getElementById('status').textContent = data.status;
                }
                else if (data.type === 'transcript') {
                    document.getElementById('transcript').innerHTML += '<div>You: ' + data.text + '</div>';
                }
                else if (data.type === 'llm_response') {
                    document.getElementById('transcript').innerHTML += '<div style="color: #667eea;">Bot: ' + data.text + '</div>';
                }
                else if (data.type === 'audio') {
                    // Play audio
                    const audio = new Audio('data:audio/mp3;base64,' + data.audio);
                    audio.play();

                    // Update latency display
                    if (data.latency) {
                        document.getElementById('vad-smartturn').textContent = (data.latency.vad_to_smartturn_ms || '-') + 'ms';
                        document.getElementById('llm-ttft').textContent = (data.latency.llm_ttft_ms || '-') + 'ms';
                        document.getElementById('tts-ttfs').textContent = (data.latency.tts_ttfs_ms || '-') + 'ms';
                        document.getElementById('e2e').textContent = (data.latency.e2e_ms || '-') + 'ms';
                    }
                }
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                stopMic();
            };
        }

        async function startMic() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: true, sampleRate: 16000 }
                });

                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const float32 = e.inputBuffer.getChannelData(0);
                        const int16 = new Int16Array(float32.length);
                        for (let i = 0; i < float32.length; i++) {
                            int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
                        }
                        ws.send(JSON.stringify({
                            type: 'audio',
                            audio: btoa(String.fromCharCode(...new Uint8Array(int16.buffer)))
                        }));
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

                document.getElementById('status').textContent = 'Listening...';
            } catch (err) {
                console.error('Mic error:', err);
            }
        }

        function stopMic() {
            if (processor) processor.disconnect();
            if (audioContext) audioContext.close();
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        }

        function stop() {
            if (ws) {
                ws.send(JSON.stringify({ type: 'stop' }));
                ws.close();
            }
            stopMic();
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def get_index():
    return HTMLResponse(HTML_PAGE)


# ============ Main ============
if __name__ == "__main__":
    print("=" * 60)
    print("  Pipecat Voice Bot")
    print("=" * 60)
    print()
    print("Features:")
    print("  [x] VAD (Silero) for silence detection")
    print("  [x] SmartTurn (LocalSmartTurnAnalyzerV3) for turn detection")
    print("  [x] Deepgram Nova-3 STT (final transcripts only)")
    print("  [x] OpenAI GPT-4o-mini with function calling")
    print("  [x] ElevenLabs TTS")
    print()
    print("Latency Tracking:")
    print("  - VAD timeout / SmartTurn detect")
    print("  - LLM TTFT (Time To First Token)")
    print("  - TTS TTFS (Time To First Speech)")
    print("  - End-to-End = VAD/SmartTurn + LLM TTFT + TTS TTFS")
    print()
    print("Web UI: http://localhost:5002")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=5002)
