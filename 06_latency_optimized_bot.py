"""
Project 4: Latency-Optimized Voice Bot
======================================
Voice bot with detailed latency tracking and optimization.

Tracks:
- VAD: When user stops speaking
- STT: Time to get transcription
- LLM: Time to first token
- TTS: Time to first audio byte
- E2E: Total end-to-end latency

Pipeline: Mic → VAD → Deepgram STT → GPT-4o-mini → ElevenLabs TTS → Speaker

Run: python 06_latency_optimized_bot.py [--model gpt-4o]
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import asyncio
import os
import sys
import time
import ssl
import certifi
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# Fix SSL certificates on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# Shared state for latency tracking across processors
class LatencyState:
    """Shared state for tracking latencies across multiple processors."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.turn_count = 0
        self.latency_history = deque(maxlen=20)

        # Current turn timestamps
        self.vad_end_time: Optional[float] = None
        self.stt_end_time: Optional[float] = None
        self.llm_first_token_time: Optional[float] = None
        self.tts_first_audio_time: Optional[float] = None
        self.user_text: str = ""

    def reset_turn(self):
        """Reset for a new turn."""
        self.vad_end_time = None
        self.stt_end_time = None
        self.llm_first_token_time = None
        self.tts_first_audio_time = None
        self.user_text = ""

    def stt_latency(self) -> Optional[float]:
        if self.vad_end_time and self.stt_end_time:
            return self.stt_end_time - self.vad_end_time
        return None

    def llm_latency(self) -> Optional[float]:
        if self.stt_end_time and self.llm_first_token_time:
            return self.llm_first_token_time - self.stt_end_time
        return None

    def tts_latency(self) -> Optional[float]:
        if self.llm_first_token_time and self.tts_first_audio_time:
            return self.tts_first_audio_time - self.llm_first_token_time
        return None

    def e2e_latency(self) -> Optional[float]:
        if self.vad_end_time and self.tts_first_audio_time:
            return self.tts_first_audio_time - self.vad_end_time
        return None

    def save_turn(self):
        """Save current turn metrics to history."""
        self.latency_history.append({
            'stt': self.stt_latency(),
            'llm': self.llm_latency(),
            'tts': self.tts_latency(),
            'e2e': self.e2e_latency(),
        })
        self.turn_count += 1


class InputLatencyTracker(FrameProcessor):
    """Tracks input latencies (VAD, STT). Place after STT in pipeline."""

    def __init__(self, state: LatencyState):
        super().__init__()
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.time()

        # VAD: User started speaking
        if isinstance(frame, UserStartedSpeakingFrame):
            print(f"\n[VAD] User speaking...", flush=True)

        # VAD: User stopped speaking
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.state.reset_turn()
            self.state.vad_end_time = now
            print(f"[VAD] User stopped @ {now:.3f}", flush=True)

        # STT: Transcription received
        elif isinstance(frame, TranscriptionFrame):
            self.state.stt_end_time = now
            self.state.user_text = frame.text
            stt_lat = self.state.stt_latency()

            lat_str = f" (STT: {stt_lat*1000:.0f}ms)" if stt_lat else ""
            print(f"[STT] Transcription{lat_str}", flush=True)
            print(f"[USER] {frame.text}", flush=True)

        # Interim transcription
        elif isinstance(frame, InterimTranscriptionFrame):
            text = frame.text[:60] if frame.text else ""
            print(f"\r[...] {text}", end="", flush=True)

        await self.push_frame(frame, direction)


class OutputLatencyTracker(FrameProcessor):
    """Tracks output latencies (LLM, TTS). Place after TTS in pipeline."""

    def __init__(self, state: LatencyState):
        super().__init__()
        self.state = state
        self._response_logged = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.time()

        # LLM: First token
        if isinstance(frame, LLMFullResponseStartFrame):
            if self.state.llm_first_token_time is None:
                self.state.llm_first_token_time = now
                llm_lat = self.state.llm_latency()
                lat_str = f" (LLM TTFT: {llm_lat*1000:.0f}ms)" if llm_lat else ""
                print(f"[LLM] First token{lat_str}", flush=True)

        # LLM: Response text
        elif isinstance(frame, TextFrame) and frame.text:
            if not self._response_logged:
                print(f"[ASSISTANT] {frame.text}", flush=True)
                self._response_logged = True

        # LLM: Finished
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._response_logged = False

        # TTS: First audio byte
        elif isinstance(frame, TTSAudioRawFrame):
            if self.state.tts_first_audio_time is None:
                self.state.tts_first_audio_time = now
                tts_lat = self.state.tts_latency()
                e2e_lat = self.state.e2e_latency()

                lat_str = f" (TTS TTFA: {tts_lat*1000:.0f}ms)" if tts_lat else ""
                print(f"[TTS] First audio{lat_str}", flush=True)

                if e2e_lat:
                    self._print_latency_breakdown(e2e_lat)
                    self.state.save_turn()

        # TTS: Finished
        elif isinstance(frame, TTSStoppedFrame):
            if self.state.turn_count > 0 and self.state.turn_count % 3 == 0:
                self._print_averages()

        await self.push_frame(frame, direction)

    def _print_latency_breakdown(self, e2e: float):
        """Print visual latency breakdown."""
        stt = self.state.stt_latency() or 0
        llm = self.state.llm_latency() or 0
        tts = self.state.tts_latency() or 0

        print(f"\n{'='*60}", flush=True)
        print(f"[E2E LATENCY] {e2e*1000:.0f}ms total", flush=True)

        # Visual bar chart
        max_lat = max(stt, llm, tts, 0.001)
        scale = 40

        def bar(val):
            width = int((val / max_lat) * scale) if max_lat > 0 else 0
            return "█" * width

        print(f"  STT:  {stt*1000:6.0f}ms {bar(stt)}", flush=True)
        print(f"  LLM:  {llm*1000:6.0f}ms {bar(llm)} ({self.state.model_name})", flush=True)
        print(f"  TTS:  {tts*1000:6.0f}ms {bar(tts)}", flush=True)

        # Status indicator
        if e2e < 1.0:
            status = "🟢 EXCELLENT"
        elif e2e < 1.5:
            status = "🟡 GOOD"
        elif e2e < 2.0:
            status = "🟠 ACCEPTABLE"
        else:
            status = "🔴 NEEDS OPTIMIZATION"
        print(f"  Status: {status} (target: <1500ms)", flush=True)
        print(f"{'='*60}\n", flush=True)

    def _print_averages(self):
        """Print average latencies over recent turns."""
        history = self.state.latency_history
        if len(history) < 2:
            return

        def avg(key):
            vals = [h[key] for h in history if h[key] is not None]
            return sum(vals) / len(vals) if vals else 0

        print(f"\n{'─'*60}", flush=True)
        print(f"[AVERAGE] Over {len(history)} turns:", flush=True)
        print(f"  STT: {avg('stt')*1000:.0f}ms | LLM: {avg('llm')*1000:.0f}ms | TTS: {avg('tts')*1000:.0f}ms", flush=True)
        print(f"  E2E Average: {avg('e2e')*1000:.0f}ms", flush=True)
        print(f"{'─'*60}\n", flush=True)


async def main():
    parser = argparse.ArgumentParser(description="Latency-optimized voice bot")
    parser.add_argument("--model", default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                        help="OpenAI model to use")
    parser.add_argument("--voice", default="JBFqnCBsd6RMkjVDRZzb",
                        help="ElevenLabs voice ID")
    args = parser.parse_args()

    print("\n" + "=" * 70, flush=True)
    print("PROJECT 4: LATENCY-OPTIMIZED VOICE BOT", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Target E2E Latency: <1500ms", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Latency Breakdown:", flush=True)
    print("  STT  = Time from silence to transcription", flush=True)
    print("  LLM  = Time from transcription to first LLM token (TTFT)", flush=True)
    print("  TTS  = Time from LLM token to first audio byte (TTFA)", flush=True)
    print("  E2E  = Total time from silence to hearing response", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Shared latency state
    latency_state = LatencyState(model_name=args.model)

    # Configure local audio with optimized VAD
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5,
                    min_volume=0.6,
                )
            ),
            vad_audio_passthrough=True,
        )
    )

    # Deepgram STT
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
    )

    # LLM
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=args.model,
    )

    # ElevenLabs TTS
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=args.voice,
        model="eleven_turbo_v2",
        optimize_streaming_latency=3,
    )

    # Latency trackers
    input_tracker = InputLatencyTracker(latency_state)
    output_tracker = OutputLatencyTracker(latency_state)

    # System prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses under 2 sentences. Be concise.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Build pipeline with trackers in correct positions
    # Input tracker AFTER stt to see TranscriptionFrame
    # Output tracker AFTER tts to see TTSAudioRawFrame
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            input_tracker,              # Tracks VAD and STT
            context_aggregator.user(),
            llm,
            tts,
            output_tracker,             # Tracks LLM and TTS
            transport.output(),
            context_aggregator.assistant(),
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

    print("Bot initialized! Speak to test latency.", flush=True)
    print("Average latencies shown every 3 turns.", flush=True)
    print("Press Ctrl+C to exit.\n", flush=True)

    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
