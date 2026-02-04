"""
Project 3: Voice Bot with SmartTurn Turn Detection
==================================================
Enhanced voice bot using SmartTurn for better turn detection.

SmartTurn uses AI to detect when you've finished speaking, rather than
just relying on silence. This means it won't interrupt you during:
- Thinking pauses ("I want to order... hmm... maybe a pizza")
- Natural speech hesitations
- Mid-sentence breaths

Pipeline: Mic → VAD → SmartTurn → Deepgram STT → GPT-4o-mini → ElevenLabs → Speaker

Run: python 05_smartturn_voice_bot.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import os
import sys
import time
import ssl
import certifi

# Fix SSL certificates on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="INFO")


class SmartTurnLogger(FrameProcessor):
    """Logs conversation events with SmartTurn-specific detection."""

    def __init__(self):
        super().__init__()
        self.turn_count = 0
        self.response_start_time = None
        self.user_speaking = False
        self.speech_start_time = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Log when user starts speaking
        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_speaking = True
            self.speech_start_time = time.time()
            print(f"\n[VAD] User started speaking...")

        # Log when user stops speaking (VAD silence detected)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self.speech_start_time:
                duration = time.time() - self.speech_start_time
                print(f"[VAD] Silence detected after {duration:.1f}s of speech")
                print(f"[SMARTTURN] Analyzing if user finished their thought...")
            self.user_speaking = False

        # Log transcriptions (final - SmartTurn determined user is done)
        elif isinstance(frame, TranscriptionFrame):
            self.turn_count += 1
            print(f"\n{'='*60}")
            print(f"[TURN {self.turn_count}] SMARTTURN: User turn complete!")
            print(f"[USER] {frame.text}")
            print(f"{'='*60}")
            self.response_start_time = time.time()

        # Log interim transcriptions
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"[LISTENING] {frame.text}", end="\r", flush=True)

        # Log LLM response
        elif isinstance(frame, LLMFullResponseStartFrame):
            print("[LLM] Generating response...")

        # Log assistant text
        elif isinstance(frame, TextFrame) and frame.text:
            if self.response_start_time:
                latency = time.time() - self.response_start_time
                print(f"[ASSISTANT] {frame.text}")
                print(f"[LATENCY] Time to first response: {latency:.2f}s")
                self.response_start_time = None

        # Log TTS events
        elif isinstance(frame, TTSStartedFrame):
            print("[TTS] Speaking...")

        elif isinstance(frame, TTSStoppedFrame):
            print("[TTS] Finished speaking\n")

        await self.push_frame(frame, direction)


async def main():
    print("\n" + "=" * 70, flush=True)
    print("PROJECT 3: VOICE BOT WITH SMARTTURN", flush=True)
    print("=" * 70, flush=True)
    print("Pipeline: Mic → VAD → SmartTurn → Deepgram → GPT-4o-mini → ElevenLabs → Speaker", flush=True)
    print(flush=True)
    print("SmartTurn Enhancement:", flush=True)
    print("  - Uses AI to detect when you've finished speaking", flush=True)
    print("  - Won't interrupt during thinking pauses", flush=True)
    print("  - Waits for complete thoughts, not just silence", flush=True)
    print(flush=True)
    print("Try saying: 'I want to order... hmm... maybe a pizza'", flush=True)
    print("The bot should wait for you to finish, not interrupt at 'hmm'", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Configure local audio transport
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # Speech-to-Text: Deepgram Nova-2
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
    )

    # LLM: OpenAI GPT-4o-mini
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    # Text-to-Speech: ElevenLabs
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # George voice
        model="eleven_turbo_v2",
    )

    # Conversation logger
    conversation_logger = SmartTurnLogger()

    # System prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses under 2 sentences.",
        },
    ]

    # Create context with SmartTurn
    context = LLMContext(messages=messages)

    # Configure SmartTurn with VAD
    # Key: VAD stop_secs=0.2 (short) so SmartTurn can take over
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context=context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.2,  # Short silence threshold, SmartTurn handles the rest
                    min_volume=0.6,
                )
            ),
        ),
    )

    # Build pipeline with SmartTurn
    pipeline = Pipeline(
        [
            transport.input(),       # Microphone input
            stt,                     # Deepgram STT
            user_aggregator,         # SmartTurn + VAD user turn detection
            llm,                     # OpenAI GPT-4o-mini
            conversation_logger,     # Log events
            tts,                     # ElevenLabs TTS
            transport.output(),      # Speaker output
            assistant_aggregator,    # Assistant context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("Loading SmartTurn model (first run may take a moment)...", flush=True)
    print("Bot initialized! Speak into your microphone.", flush=True)
    print("Press Ctrl+C to exit.\n", flush=True)

    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
