"""
Real Pipecat Voice Bot
======================
Using ACTUAL Pipecat library with:
1. VAD (Silero) - Detects silence start/stop
2. SmartTurn (LocalSmartTurnAnalyzerV3) - Additional check after VAD stop
3. Deepgram Nova-3 - Final transcripts only (not streaming interim)
4. Correct latency: VAD timeout/SmartTurn + LLM TTFT + TTS TTFS

WHY 800ms WAIT?
---------------
The 800ms (or configurable stop_secs) is the SmartTurn timeout. After VAD detects
silence, SmartTurn analyzes the audio with an ML model to predict if the user is
truly done speaking. If the model is uncertain, it waits up to stop_secs (800ms-3s)
for either:
  a) More speech (user continues)
  b) Confident "end of turn" prediction from the ML model
  c) Timeout (fallback to end turn)

This prevents interrupting users who pause to think mid-sentence.

Run: python real_pipecat_bot.py
Web: http://localhost:5002
"""

import os
import sys
import json
import base64
import time
import asyncio
from datetime import datetime
from typing import Optional, List
import random

from dotenv import load_dotenv
load_dotenv(override=True)

# Pipecat imports
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    EndFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

# Deepgram STT
from pipecat.services.deepgram.stt import DeepgramSTTService
from deepgram.clients.listen.v1.websocket.options import LiveOptions

# OpenAI LLM
from pipecat.services.openai.llm import OpenAILLMService

# ElevenLabs TTS
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

# VAD
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# SmartTurn
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

# FastAPI for WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# API Keys
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

app = FastAPI()

# ============ Function Calling ============
JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "What's a computer's favorite snack? Microchips!",
    "Why did the developer go broke? Because he used up all his cache!",
]

MOCK_ORDERS = {
    "12345": {"status": "shipped", "delivery": "February 10, 2026"},
    "67890": {"status": "processing", "ship_by": "February 9, 2026"},
}

def get_current_time():
    return datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")

def tell_joke():
    return random.choice(JOKES)

def lookup_order(order_id):
    if order_id in MOCK_ORDERS:
        o = MOCK_ORDERS[order_id]
        return f"Order {order_id} is {o['status']}"
    return f"Order {order_id} not found"


# ============ Latency Tracking Processor ============
class LatencyTracker(FrameProcessor):
    """
    Tracks latency at each stage:
    - VAD stop time (when silence detected)
    - SmartTurn end time (when turn confirmed)
    - LLM TTFT (time to first token)
    - TTS TTFS (time to first speech)

    End-to-end = (SmartTurn end OR VAD timeout) to TTS first audio
    """

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self.reset()

    def reset(self):
        self.vad_stop_time = None
        self.transcript_time = None
        self.llm_start_time = None
        self.llm_first_token_time = None
        self.tts_start_time = None
        self.tts_first_audio_time = None
        self.current_transcript = ""
        self.current_response = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # User stopped speaking (VAD detected silence + SmartTurn confirmed)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self.vad_stop_time = time.time()
            logger.info(f"[LATENCY] VAD/SmartTurn stop detected")

        # Final transcript received (NOT interim)
        if isinstance(frame, TranscriptionFrame) and not isinstance(frame, InterimTranscriptionFrame):
            self.transcript_time = time.time()
            self.current_transcript = frame.text
            logger.info(f"[LATENCY] Final transcript: {frame.text}")

            # Send transcript to UI
            try:
                await self.websocket.send_json({
                    "type": "transcript",
                    "text": frame.text
                })
            except:
                pass

        # LLM started
        if isinstance(frame, LLMFullResponseStartFrame):
            self.llm_start_time = time.time()
            logger.info(f"[LATENCY] LLM started")

        # LLM text (first token)
        if isinstance(frame, TextFrame):
            if self.llm_first_token_time is None and self.llm_start_time:
                self.llm_first_token_time = time.time()
                ttft = (self.llm_first_token_time - self.llm_start_time) * 1000
                logger.info(f"[LATENCY] LLM TTFT: {ttft:.0f}ms")
            self.current_response += frame.text

        # LLM finished
        if isinstance(frame, LLMFullResponseEndFrame):
            logger.info(f"[LATENCY] LLM response: {self.current_response[:50]}...")
            try:
                await self.websocket.send_json({
                    "type": "response",
                    "text": self.current_response
                })
            except:
                pass

        # TTS started
        if isinstance(frame, TTSStartedFrame):
            self.tts_start_time = time.time()
            logger.info(f"[LATENCY] TTS started")

        # TTS audio (first chunk)
        if isinstance(frame, TTSAudioRawFrame):
            if self.tts_first_audio_time is None and self.tts_start_time:
                self.tts_first_audio_time = time.time()
                ttfs = (self.tts_first_audio_time - self.tts_start_time) * 1000
                logger.info(f"[LATENCY] TTS TTFS: {ttfs:.0f}ms")

        # TTS finished - calculate full latency
        if isinstance(frame, TTSStoppedFrame):
            await self._send_latency_summary()
            self.reset()

        await self.push_frame(frame, direction)

    async def _send_latency_summary(self):
        """Calculate and send latency breakdown"""
        latency = {}

        # VAD/SmartTurn to transcript
        if self.vad_stop_time and self.transcript_time:
            latency['vad_to_transcript_ms'] = int((self.transcript_time - self.vad_stop_time) * 1000)

        # LLM TTFT
        if self.llm_start_time and self.llm_first_token_time:
            latency['llm_ttft_ms'] = int((self.llm_first_token_time - self.llm_start_time) * 1000)

        # TTS TTFS
        if self.tts_start_time and self.tts_first_audio_time:
            latency['tts_ttfs_ms'] = int((self.tts_first_audio_time - self.tts_start_time) * 1000)

        # End-to-end: VAD/SmartTurn stop to TTS first audio
        # Formula: VAD timeout/SmartTurn + LLM TTFT + TTS TTFS
        if self.vad_stop_time and self.tts_first_audio_time:
            latency['e2e_ms'] = int((self.tts_first_audio_time - self.vad_stop_time) * 1000)

        logger.info(f"[LATENCY] Summary: {latency}")

        try:
            await self.websocket.send_json({
                "type": "latency",
                "latency": latency
            })
        except:
            pass


# ============ Audio Output Handler ============
class AudioOutputHandler(FrameProcessor):
    """Sends audio to websocket for playback"""

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self.audio_buffer = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            self.audio_buffer.extend(frame.audio)

        if isinstance(frame, TTSStoppedFrame) and self.audio_buffer:
            # Send accumulated audio
            try:
                await self.websocket.send_json({
                    "type": "audio",
                    "audio": base64.b64encode(bytes(self.audio_buffer)).decode(),
                    "sample_rate": 24000  # ElevenLabs default
                })
            except:
                pass
            self.audio_buffer = bytearray()

        await self.push_frame(frame, direction)


# ============ Main Pipeline Builder ============
async def create_pipeline(websocket: WebSocket):
    """
    Create the Pipecat pipeline:

    Audio -> VAD -> SmartTurn -> Deepgram STT -> OpenAI LLM -> ElevenLabs TTS -> Audio Out

    VAD: Detects when user starts/stops speaking based on audio energy
    SmartTurn: After VAD stop, uses ML model to confirm end-of-turn

    The 800ms (stop_secs) is SmartTurn's timeout - how long to wait for:
    - More speech from user
    - Confident end-of-turn prediction
    Before forcing turn end
    """

    # 1. VAD - Voice Activity Detection (Silero)
    # Detects silence/speech transitions
    vad = SileroVADAnalyzer(
        params=VADParams(
            threshold=0.5,              # Confidence threshold for speech detection
            min_speech_duration_ms=250, # Minimum speech to trigger "started"
            min_silence_duration_ms=300,# 300ms silence triggers "stopped"
        )
    )

    # 2. SmartTurn - ML-based end-of-turn detection
    # WHY 800ms? This is the fallback timeout after VAD detects silence.
    # SmartTurn analyzes audio features to predict if user is truly done.
    # If uncertain, waits up to stop_secs before forcing turn end.
    smart_turn = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(
            stop_secs=0.8,      # 800ms timeout - wait for confident prediction
            pre_speech_ms=500,  # Include 500ms audio before speech started
        )
    )

    # 3. Deepgram STT - Speech to Text
    # Using Nova-3 (NOT Nova-2!) with final transcripts only
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            model="nova-3",          # Nova-3 (latest) not Nova-2!
            language="en",
            punctuate=True,
            interim_results=False,   # ONLY final transcripts, not streaming
            endpointing=300,         # 300ms silence = utterance boundary
            smart_format=True,
            vad_events=True,         # Get VAD events from Deepgram too
        )
    )

    # 4. OpenAI LLM
    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
    )

    # 5. ElevenLabs TTS
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
        model="eleven_turbo_v2",
    )

    # 6. Latency tracker
    latency_tracker = LatencyTracker(websocket)

    # 7. Audio output handler
    audio_output = AudioOutputHandler(websocket)

    # Build pipeline
    pipeline = Pipeline([
        vad,            # Detect speech/silence
        smart_turn,     # Confirm end-of-turn with ML
        stt,            # Transcribe
        latency_tracker,# Track latency
        llm,            # Generate response
        tts,            # Synthesize speech
        audio_output,   # Send to browser
    ])

    return pipeline


# ============ WebSocket Handler ============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Client connected")

    try:
        # Create pipeline
        pipeline = await create_pipeline(websocket)

        # Create task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )

        # Create runner
        runner = PipelineRunner()

        # Start pipeline in background
        pipeline_task = asyncio.create_task(runner.run(task))

        await websocket.send_json({
            "type": "status",
            "message": "Pipeline started: VAD + SmartTurn(800ms) + Nova-3 + GPT-4o-mini + ElevenLabs"
        })

        # Process incoming audio
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)

                if data.get("type") == "audio":
                    audio_bytes = base64.b64decode(data["audio"])
                    frame = AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=16000,
                        num_channels=1
                    )
                    await task.queue_frame(frame)

                elif data.get("type") == "stop":
                    break

            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"[WS] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("[WS] Client disconnected")


# ============ HTML Page ============
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Real Pipecat Voice Bot</title>
    <style>
        body { font-family: system-ui; background: #1a1a2e; color: #fff; padding: 20px; max-width: 900px; margin: 0 auto; }
        h1 { color: #667eea; }
        .info { background: #2d2d44; padding: 15px; border-radius: 8px; margin: 15px 0; font-size: 14px; line-height: 1.6; }
        .info h3 { color: #4ade80; margin-top: 0; }
        .btn { padding: 15px 30px; font-size: 16px; border: none; border-radius: 8px; cursor: pointer; margin: 5px; }
        .btn-start { background: #4ade80; color: #000; }
        .btn-stop { background: #f87171; color: #fff; }
        .status { padding: 15px; background: #333; border-radius: 8px; margin: 15px 0; }
        .conversation { background: #222; padding: 15px; border-radius: 8px; min-height: 150px; margin: 15px 0; }
        .user { color: #60a5fa; margin: 10px 0; }
        .bot { color: #4ade80; margin: 10px 0; }
        .latency { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0; }
        .latency-item { background: #333; padding: 15px; border-radius: 8px; text-align: center; }
        .latency-value { font-size: 28px; font-weight: bold; color: #4ade80; }
        .latency-label { font-size: 12px; color: #888; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Real Pipecat Voice Bot</h1>

    <div class="info">
        <h3>How VAD + SmartTurn Work Together</h3>
        <p><strong>VAD (Voice Activity Detection):</strong> Monitors audio energy. When you stop speaking for 300ms, VAD signals "user stopped".</p>
        <p><strong>SmartTurn (800ms timeout):</strong> After VAD stop, SmartTurn analyzes the audio with an ML model to predict if you're truly done. Why 800ms? It's the fallback timeout - SmartTurn waits up to 800ms for either:</p>
        <ul>
            <li>More speech from you (you're just pausing to think)</li>
            <li>Confident "end of turn" prediction from the ML model</li>
            <li>Timeout reached (force end turn)</li>
        </ul>
        <p>This prevents the bot from interrupting you mid-thought when you say "um..." or pause briefly.</p>
    </div>

    <div class="info">
        <h3>Pipeline Components</h3>
        <p><strong>Deepgram Nova-3</strong> (NOT Nova-2) with <code>interim_results=False</code> - only final transcripts</p>
        <p><strong>Latency Formula:</strong> E2E = VAD/SmartTurn + LLM TTFT + TTS TTFS</p>
    </div>

    <button class="btn btn-start" onclick="start()">Start Conversation</button>
    <button class="btn btn-stop" onclick="stop()">Stop</button>

    <div class="status" id="status">Click Start to begin</div>

    <div class="conversation" id="conversation"></div>

    <div class="latency">
        <div class="latency-item">
            <div class="latency-value" id="vad">-</div>
            <div class="latency-label">VAD/SmartTurn → Transcript</div>
        </div>
        <div class="latency-item">
            <div class="latency-value" id="llm">-</div>
            <div class="latency-label">LLM TTFT</div>
        </div>
        <div class="latency-item">
            <div class="latency-value" id="tts">-</div>
            <div class="latency-label">TTS TTFS</div>
        </div>
        <div class="latency-item">
            <div class="latency-value" id="e2e" style="color: #fbbf24;">-</div>
            <div class="latency-label">End-to-End</div>
        </div>
    </div>

    <script>
        let ws, mediaStream, audioContext, processor;

        async function start() {
            document.getElementById('status').textContent = 'Connecting...';
            document.getElementById('conversation').innerHTML = '';

            ws = new WebSocket('ws://localhost:5002/ws');

            ws.onopen = async () => {
                document.getElementById('status').textContent = 'Connected. Starting mic...';
                await startMic();
            };

            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);

                if (data.type === 'status') {
                    document.getElementById('status').textContent = data.message;
                }
                else if (data.type === 'transcript') {
                    document.getElementById('conversation').innerHTML +=
                        '<div class="user">You: ' + data.text + '</div>';
                }
                else if (data.type === 'response') {
                    document.getElementById('conversation').innerHTML +=
                        '<div class="bot">Bot: ' + data.text + '</div>';
                }
                else if (data.type === 'audio') {
                    playAudio(data.audio, data.sample_rate);
                }
                else if (data.type === 'latency') {
                    const l = data.latency;
                    document.getElementById('vad').textContent = (l.vad_to_transcript_ms || '-') + 'ms';
                    document.getElementById('llm').textContent = (l.llm_ttft_ms || '-') + 'ms';
                    document.getElementById('tts').textContent = (l.tts_ttfs_ms || '-') + 'ms';
                    document.getElementById('e2e').textContent = (l.e2e_ms || '-') + 'ms';
                }
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                stopMic();
            };
        }

        async function startMic() {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, sampleRate: 16000, channelCount: 1 }
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
                    const b64 = btoa(String.fromCharCode(...new Uint8Array(int16.buffer)));
                    ws.send(JSON.stringify({ type: 'audio', audio: b64 }));
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);
            document.getElementById('status').textContent = 'Listening... (speak now)';
        }

        function playAudio(b64, sampleRate) {
            const bytes = atob(b64);
            const arr = new Uint8Array(bytes.length);
            for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);

            const audioCtx = new AudioContext({ sampleRate: sampleRate || 24000 });
            audioCtx.decodeAudioData(arr.buffer.slice(0), (buffer) => {
                const source = audioCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(audioCtx.destination);
                source.start();
            }).catch(e => {
                // Fallback: try as raw PCM
                console.log('Trying raw PCM playback');
            });
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
async def index():
    return HTMLResponse(HTML)


# ============ Main ============
if __name__ == "__main__":
    print("=" * 70)
    print("  Real Pipecat Voice Bot")
    print("=" * 70)
    print()
    print("Using ACTUAL Pipecat library with:")
    print("  1. VAD (Silero) - 300ms silence detection")
    print("  2. SmartTurn (LocalSmartTurnAnalyzerV3) - ML end-of-turn detection")
    print("  3. Deepgram Nova-3 - Final transcripts only (NOT interim)")
    print("  4. OpenAI GPT-4o-mini - LLM")
    print("  5. ElevenLabs Turbo v2 - TTS")
    print()
    print("WHY 800ms WAIT?")
    print("  After VAD detects silence, SmartTurn uses an ML model to predict")
    print("  if the user is truly done speaking. The 800ms is the fallback")
    print("  timeout - it waits for confident prediction or more speech.")
    print()
    print("LATENCY FORMULA:")
    print("  End-to-End = VAD/SmartTurn + LLM TTFT + TTS TTFS")
    print()
    print("Web UI: http://localhost:5002")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=5002, log_level="info")
