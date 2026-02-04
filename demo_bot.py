#!/usr/bin/env python3
"""
DEMO VOICE BOT - Ready for presentation
=======================================
Clean, stable voice bot with latency tracking.

Run: python demo_bot.py
"""

import warnings
warnings.filterwarnings("ignore")

_original_warn = warnings.warn
def _quiet_warn(*args, **kwargs): pass
warnings.warn = _quiet_warn

import asyncio
import os
import sys
import time
import ssl
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import logging
logging.getLogger().setLevel(logging.CRITICAL)
for name in ['pipecat', 'deepgram', 'websockets', 'httpx', 'httpcore']:
    logging.getLogger(name).setLevel(logging.CRITICAL)

from dotenv import load_dotenv
load_dotenv(override=True)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame, TranscriptionFrame, TextFrame,
    LLMFullResponseStartFrame, LLMFullResponseEndFrame,
    TTSAudioRawFrame, TTSStoppedFrame,
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
    """Simple tracker for demo."""
    def __init__(self):
        super().__init__()
        self.transcription_time = None
        self.llm_time = None
        self.tts_time = None
        self._logged = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.time()

        if isinstance(frame, UserStartedSpeakingFrame):
            print("\n[Listening...]", flush=True)
            self.transcription_time = None
            self.llm_time = None
            self.tts_time = None
            self._logged = False

        elif isinstance(frame, TranscriptionFrame):
            self.transcription_time = now
            print(f"\n[You]: {frame.text}", flush=True)

        elif isinstance(frame, LLMFullResponseStartFrame):
            if self.llm_time is None:
                self.llm_time = now

        elif isinstance(frame, TextFrame) and frame.text and not self._logged:
            print(f"[Bot]: {frame.text}", flush=True)
            self._logged = True

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._logged = False

        elif isinstance(frame, TTSAudioRawFrame):
            if self.tts_time is None and self.transcription_time:
                self.tts_time = now
                e2e = self.tts_time - self.transcription_time
                llm_lat = (self.llm_time - self.transcription_time) if self.llm_time else 0
                tts_lat = (self.tts_time - self.llm_time) if self.llm_time else 0

                status = "🟢" if e2e < 0.8 else "🟡" if e2e < 1.2 else "🟠" if e2e < 1.8 else "🔴"
                print(f"\n{status} Latency: {e2e*1000:.0f}ms (LLM: {llm_lat*1000:.0f}ms, TTS: {tts_lat*1000:.0f}ms)", flush=True)

        await self.push_frame(frame, direction)


async def main():
    print("\n" + "="*50, flush=True)
    print("       VOICE AI DEMO", flush=True)
    print("="*50, flush=True)
    print("Using: GPT-4o-mini + Cartesia TTS", flush=True)
    print("="*50, flush=True)
    print("\nSpeak naturally. Press Ctrl+C to exit.\n", flush=True)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
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

    # Cartesia TTS - much faster than ElevenLabs (~40ms vs ~1500ms)
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    )

    tracker = DemoTracker()

    messages = [{
        "role": "system",
        "content": "You are a friendly voice assistant. Keep responses brief - 1-2 sentences max. Be helpful and conversational."
    }]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        tracker,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner(handle_sigint=True)

    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo ended. Goodbye!", flush=True)
    except Exception as e:
        print(f"\nError: {e}", flush=True)
