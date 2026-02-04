"""
Project 2: Local Voice Bot (Microphone/Speaker)
================================================
Pipeline: Microphone → SileroVAD → Deepgram STT → OpenAI GPT-4o-mini → ElevenLabs TTS → Speaker

Features:
- Local audio I/O via PyAudio
- Voice Activity Detection (SileroVAD)
- Interruption handling (speak while AI is talking to stop it)
- Conversation context maintained across turns

Run: python 04_local_voice_bot.py
"""

import warnings
# Suppress deprecation warnings for cleaner output - must be before imports
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

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

# Configure logging - set to INFO for cleaner output
logger.remove(0)
logger.add(sys.stderr, level="INFO")


class ConversationLogger(FrameProcessor):
    """Logs conversation events for verification."""

    def __init__(self):
        super().__init__()
        self.turn_count = 0
        self.response_start_time = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Log transcriptions (what user said)
        if isinstance(frame, TranscriptionFrame):
            self.turn_count += 1
            print(f"\n{'='*50}")
            print(f"[TURN {self.turn_count}] USER: {frame.text}")
            print(f"{'='*50}")
            self.response_start_time = time.time()

        # Log interim transcriptions
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"[LISTENING] {frame.text}", end="\r")

        # Log when LLM starts responding
        elif isinstance(frame, LLMFullResponseStartFrame):
            print("[LLM] Generating response...")

        # Log LLM text output
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
            print("[TTS] Finished speaking")

        # Log interruption events
        elif isinstance(frame, UserStartedSpeakingFrame):
            print("\n[INTERRUPT] User started speaking!")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            print("[VAD] User stopped speaking")

        await self.push_frame(frame, direction)


async def main():
    print("\n" + "=" * 60)
    print("PROJECT 2: LOCAL VOICE BOT")
    print("=" * 60)
    print("Pipeline: Mic → VAD → Deepgram STT → GPT-4o-mini → ElevenLabs → Speaker")
    print("Interruption handling: ENABLED")
    print("=" * 60 + "\n")

    # Configure local audio with VAD for natural conversation
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
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

    # Text-to-Speech: ElevenLabs with turbo model for low latency
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # George - deep US male
        model="eleven_turbo_v2",
    )

    # Conversation logger for verification
    conversation_logger = ConversationLogger()

    # System prompt as specified
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses under 2 sentences.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Build the pipeline
    # Mic → VAD → STT → Context → LLM → Logger → TTS → Speaker
    pipeline = Pipeline(
        [
            transport.input(),              # Microphone input with VAD
            stt,                            # Deepgram speech-to-text
            context_aggregator.user(),      # Accumulate user messages
            llm,                            # OpenAI GPT-4o-mini
            conversation_logger,            # Log conversation events
            tts,                            # ElevenLabs text-to-speech
            transport.output(),             # Speaker output
            context_aggregator.assistant(), # Accumulate assistant messages
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,       # Enable interruption handling
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("Bot initialized! Speak into your microphone.")
    print("Try interrupting while the bot is speaking - it should stop!")
    print("Press Ctrl+C to exit.\n")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
