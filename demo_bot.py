#!/usr/bin/env python3
"""
DEMO VOICE BOT
==============
IMPORTANT: Use HEADPHONES to prevent echo feedback!

Run: python demo_bot.py
"""

# Silence EVERYTHING before any imports
import sys
import os
import warnings
import logging

# Nuclear option: silence all warnings and logging
warnings.filterwarnings("ignore")
os.environ["LOGURU_LEVEL"] = "CRITICAL"
logging.disable(logging.CRITICAL)

# Patch warnings.warn
_orig_warn = warnings.warn
def _no_warn(*a, **k): pass
warnings.warn = _no_warn

# Patch print from other modules
class QuietStderr:
    def write(self, msg):
        if "deprecated" in msg.lower() or "warning" in msg.lower() or "debug" in msg.lower() or "failed" in msg.lower():
            return
        sys.__stderr__.write(msg)
    def flush(self):
        sys.__stderr__.flush()

# Don't redirect stderr yet - we need to see our own output

import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Now import pipecat (will generate warnings we ignore)
import asyncio
import time
from dotenv import load_dotenv
load_dotenv(override=True)

# Silence loguru specifically
from loguru import logger
logger.remove()
logger.add(lambda msg: None)  # Send all logs to nowhere

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame, TranscriptionFrame, TextFrame,
    LLMFullResponseStartFrame, TTSAudioRawFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams


class DemoTracker(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.t0 = None
        self.t1 = None
        self.t2 = None
        self._said = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.time()

        if isinstance(frame, UserStartedSpeakingFrame):
            print("\n🎤 Listening...", flush=True)
            self.t0 = self.t1 = self.t2 = None
            self._said = False

        elif isinstance(frame, TranscriptionFrame):
            self.t0 = now
            print(f"\n👤 You: {frame.text}", flush=True)

        elif isinstance(frame, LLMFullResponseStartFrame):
            if not self.t1:
                self.t1 = now

        elif isinstance(frame, TextFrame) and frame.text and not self._said:
            print(f"🤖 Bot: {frame.text}", flush=True)
            self._said = True

        elif isinstance(frame, TTSAudioRawFrame):
            if not self.t2 and self.t0:
                self.t2 = now
                e2e = (self.t2 - self.t0) * 1000
                icon = "🟢" if e2e < 800 else "🟡" if e2e < 1200 else "🔴"
                print(f"{icon} {e2e:.0f}ms", flush=True)

        await self.push_frame(frame, direction)


async def main():
    print("\n" + "="*50)
    print("    🎙️  VOICE AI DEMO  🎙️")
    print("="*50)
    print()
    print("⚠️  USE HEADPHONES to prevent echo!")
    print()
    print("🟢 <800ms | 🟡 800-1200ms | 🔴 >1200ms")
    print("="*50)
    print("\nSpeak after you see 🎤. Press Ctrl+C to exit.\n")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6)),
            vad_audio_passthrough=True,
        )
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
    )

    tracker = DemoTracker()

    messages = [{
        "role": "system",
        "content": "You are a helpful voice assistant. Give brief 1-2 sentence responses."
    }]

    context = OpenAILLMContext(messages)
    agg = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        tracker,
        agg.user(),
        llm,
        tts,
        transport.output(),
        agg.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
