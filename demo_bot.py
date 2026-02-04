#!/usr/bin/env python3
"""
DEMO VOICE BOT - Use HEADPHONES!
"""

import sys
import os
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["LOGURU_LEVEL"] = "CRITICAL"
logging.disable(logging.CRITICAL)

import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import asyncio
import time
from dotenv import load_dotenv
load_dotenv(override=True)

from loguru import logger
logger.remove()

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame, TranscriptionFrame, TextFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
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


# Global timing
g_start_time = None
g_printed_user = False


class InputTracker(FrameProcessor):
    """Track user input - place after STT"""
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        global g_start_time, g_printed_user
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            g_start_time = None
            g_printed_user = False
            print("\n🎤 Listening...", flush=True)

        elif isinstance(frame, TranscriptionFrame) and frame.text:
            g_start_time = time.time()
            if not g_printed_user:
                print(f"👤 You: {frame.text}", flush=True)
                g_printed_user = True

        await self.push_frame(frame, direction)


class OutputTracker(FrameProcessor):
    """Track bot output - place after TTS"""
    def __init__(self):
        super().__init__()
        self.bot_response = ""
        self.printed = False
        self.first_text_time = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        global g_start_time
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self.bot_response = ""
            self.printed = False
            self.first_text_time = None

        elif isinstance(frame, TextFrame) and frame.text:
            if not self.first_text_time:
                self.first_text_time = time.time()
            self.bot_response += frame.text

            # Print when we have a reasonable response
            if not self.printed and len(self.bot_response) > 10:
                print(f"🤖 Bot: {self.bot_response}", flush=True)
                self.printed = True

                # Show latency
                if g_start_time and self.first_text_time:
                    latency = (self.first_text_time - g_start_time) * 1000
                    icon = "🟢" if latency < 800 else "🟡" if latency < 1200 else "🔴"
                    print(f"{icon} Response time: {latency:.0f}ms\n", flush=True)

        await self.push_frame(frame, direction)


async def main():
    print("\n" + "="*50)
    print("    🎙️  VOICE AI DEMO  🎙️")
    print("="*50)
    print("\n⚠️  USE HEADPHONES to prevent echo!\n")
    print("🟢 <800ms | 🟡 800-1200ms | 🔴 >1200ms")
    print("="*50)

    # Check API keys
    missing = []
    if not os.getenv("DEEPGRAM_API_KEY"): missing.append("DEEPGRAM_API_KEY")
    if not os.getenv("OPENAI_API_KEY"): missing.append("OPENAI_API_KEY")
    if not os.getenv("CARTESIA_API_KEY"): missing.append("CARTESIA_API_KEY")

    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        return

    print("\n✅ Starting...\n")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.7)),
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

    input_tracker = InputTracker()
    output_tracker = OutputTracker()

    messages = [{
        "role": "system",
        "content": "You are a helpful voice assistant. Keep responses brief - 1 to 2 sentences maximum."
    }]

    context = OpenAILLMContext(messages)
    agg = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        input_tracker,   # Track transcription
        agg.user(),
        llm,
        output_tracker,  # Track LLM output
        tts,
        transport.output(),
        agg.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner(handle_sigint=True)

    print("🎤 Ready! Start speaking...\n")
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo ended!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
