"""
Project 4: Latency-Optimized Voice Bot
======================================
Voice bot with detailed latency tracking and optimization.

Tracks:
- STT: Time to get transcription
- LLM: Time to first token (TTFT)
- TTS: Time to first audio byte (TTFA)
- E2E: Total end-to-end latency

Run: python 06_latency_optimized_bot.py [--model gpt-4o-mini] [--tts cartesia]
"""

import warnings
# Suppress ALL deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

import argparse
import asyncio
import os
import sys
import time
import ssl
import certifi
from collections import deque
from typing import Optional

# Fix SSL certificates on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warnings from imports too
import logging
logging.getLogger("pipecat").setLevel(logging.ERROR)

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

# Configure logging - only errors
logger.remove(0)
logger.add(sys.stderr, level="ERROR")


class LatencyState:
    """Shared state for tracking latencies."""
    def __init__(self, model_name: str, tts_name: str):
        self.model_name = model_name
        self.tts_name = tts_name
        self.turn_count = 0
        self.latency_history = deque(maxlen=20)
        self.reset_turn()

    def reset_turn(self):
        # Use transcription time as the anchor point (most reliable)
        self.transcription_time: Optional[float] = None
        self.llm_first_token_time: Optional[float] = None
        self.tts_first_audio_time: Optional[float] = None
        self.user_text: str = ""

    def llm_latency(self) -> Optional[float]:
        if self.transcription_time and self.llm_first_token_time:
            return self.llm_first_token_time - self.transcription_time
        return None

    def tts_latency(self) -> Optional[float]:
        if self.llm_first_token_time and self.tts_first_audio_time:
            return self.tts_first_audio_time - self.llm_first_token_time
        return None

    def e2e_latency(self) -> Optional[float]:
        if self.transcription_time and self.tts_first_audio_time:
            return self.tts_first_audio_time - self.transcription_time
        return None

    def save_turn(self):
        self.latency_history.append({
            'llm': self.llm_latency(),
            'tts': self.tts_latency(),
            'e2e': self.e2e_latency(),
        })
        self.turn_count += 1


class InputTracker(FrameProcessor):
    """Tracks input events. Place after STT."""
    def __init__(self, state: LatencyState):
        super().__init__()
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            print(f"\n[MIC] Listening...", flush=True)

        elif isinstance(frame, TranscriptionFrame):
            self.state.reset_turn()
            self.state.transcription_time = time.time()
            self.state.user_text = frame.text
            print(f"\n{'='*55}", flush=True)
            print(f"[YOU] {frame.text}", flush=True)
            print(f"{'='*55}", flush=True)

        elif isinstance(frame, InterimTranscriptionFrame):
            if frame.text:
                print(f"\r[...] {frame.text[:50]:<50}", end="", flush=True)

        await self.push_frame(frame, direction)


class OutputTracker(FrameProcessor):
    """Tracks output events. Place after TTS."""
    def __init__(self, state: LatencyState):
        super().__init__()
        self.state = state
        self._logged = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            if self.state.llm_first_token_time is None and self.state.transcription_time:
                self.state.llm_first_token_time = time.time()
                llm_lat = self.state.llm_latency()
                if llm_lat:
                    print(f"[LLM] {llm_lat*1000:.0f}ms to first token", flush=True)

        elif isinstance(frame, TextFrame) and frame.text:
            if not self._logged:
                print(f"[BOT] {frame.text}", flush=True)
                self._logged = True

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._logged = False

        elif isinstance(frame, TTSAudioRawFrame):
            if self.state.tts_first_audio_time is None and self.state.llm_first_token_time:
                self.state.tts_first_audio_time = time.time()
                self._print_latency()
                self.state.save_turn()

        elif isinstance(frame, TTSStoppedFrame):
            if self.state.turn_count > 0 and self.state.turn_count % 3 == 0:
                self._print_averages()

        await self.push_frame(frame, direction)

    def _print_latency(self):
        llm = self.state.llm_latency() or 0
        tts = self.state.tts_latency() or 0
        e2e = self.state.e2e_latency() or 0

        # Status
        if e2e < 1.0:
            status = "🟢"
        elif e2e < 1.5:
            status = "🟡"
        elif e2e < 2.0:
            status = "🟠"
        else:
            status = "🔴"

        print(f"\n[LATENCY] {status} E2E: {e2e*1000:.0f}ms (LLM: {llm*1000:.0f}ms + TTS: {tts*1000:.0f}ms)", flush=True)

        # Bar chart
        max_lat = max(llm, tts, 0.1)
        def bar(v): return "█" * int((v / max_lat) * 25)
        print(f"  LLM ({self.state.model_name}): {llm*1000:4.0f}ms {bar(llm)}", flush=True)
        print(f"  TTS ({self.state.tts_name}):  {tts*1000:4.0f}ms {bar(tts)}", flush=True)

    def _print_averages(self):
        h = self.state.latency_history
        if len(h) < 2:
            return
        def avg(k):
            v = [x[k] for x in h if x[k]]
            return sum(v)/len(v) if v else 0
        print(f"\n[AVG over {len(h)} turns] E2E: {avg('e2e')*1000:.0f}ms | LLM: {avg('llm')*1000:.0f}ms | TTS: {avg('tts')*1000:.0f}ms", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o"])
    parser.add_argument("--tts", default="elevenlabs", choices=["elevenlabs", "cartesia"])
    args = parser.parse_args()

    print("\n" + "=" * 55, flush=True)
    print("LATENCY TEST BOT", flush=True)
    print(f"LLM: {args.model} | TTS: {args.tts}", flush=True)
    print("=" * 55, flush=True)
    print("Target: <1000ms 🟢 | <1500ms 🟡 | <2000ms 🟠 | >2000ms 🔴", flush=True)
    print("=" * 55 + "\n", flush=True)

    # Shared state
    state = LatencyState(args.model, args.tts)

    # Transport
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
            vad_audio_passthrough=True,
        )
    )

    # STT
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
    )

    # LLM
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=args.model,
    )

    # TTS - choose based on argument
    if args.tts == "cartesia":
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )
    else:
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model="eleven_turbo_v2",
            optimize_streaming_latency=4,  # Max optimization
        )

    # Trackers
    input_tracker = InputTracker(state)
    output_tracker = OutputTracker(state)

    # Context
    messages = [{"role": "system", "content": "You are helpful. Keep responses under 2 sentences."}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        input_tracker,
        context_aggregator.user(),
        llm,
        tts,
        output_tracker,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner(handle_sigint=True)

    print("Speak now! Press Ctrl+C to exit.\n", flush=True)
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
