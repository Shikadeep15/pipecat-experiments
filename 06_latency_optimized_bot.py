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


@dataclass
class LatencyMetrics:
    """Tracks latency metrics for a single turn."""
    vad_end_time: Optional[float] = None      # When user stopped speaking
    stt_end_time: Optional[float] = None      # When transcription received
    llm_first_token_time: Optional[float] = None  # When LLM started generating
    llm_end_time: Optional[float] = None      # When LLM finished
    tts_first_audio_time: Optional[float] = None  # When first TTS audio received
    tts_end_time: Optional[float] = None      # When TTS finished

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


class LatencyTracker(FrameProcessor):
    """Tracks and logs detailed latency metrics."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__()
        self.model_name = model_name
        self.current_metrics: Optional[LatencyMetrics] = None
        self.turn_count = 0
        self.latency_history: deque = deque(maxlen=20)  # Keep last 20 turns
        self.user_text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.time()

        # VAD: User stopped speaking
        if isinstance(frame, UserStoppedSpeakingFrame):
            self.current_metrics = LatencyMetrics(vad_end_time=now)
            print(f"\n[VAD] User stopped speaking @ {now:.3f}", flush=True)

        # STT: Transcription received
        elif isinstance(frame, TranscriptionFrame):
            # Create metrics if we don't have them (VAD event might have been missed)
            if not self.current_metrics:
                self.current_metrics = LatencyMetrics(vad_end_time=now)

            self.current_metrics.stt_end_time = now
            self.user_text = frame.text
            stt_lat = self.current_metrics.stt_latency()
            if stt_lat is not None and stt_lat > 0:
                print(f"[STT] Transcription received @ {now:.3f} (STT: {stt_lat*1000:.0f}ms)", flush=True)
            else:
                print(f"[STT] Transcription received @ {now:.3f}", flush=True)
            print(f"[USER] {frame.text}", flush=True)

        # Interim transcription (for feedback)
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"\r[...] {frame.text[:50]}", end="", flush=True)

        # LLM: First token
        elif isinstance(frame, LLMFullResponseStartFrame):
            if self.current_metrics:
                self.current_metrics.llm_first_token_time = now
                llm_lat = self.current_metrics.llm_latency()
                if llm_lat is not None:
                    print(f"[LLM] First token @ {now:.3f} (LLM TTFT: {llm_lat*1000:.0f}ms)", flush=True)
                else:
                    print(f"[LLM] First token @ {now:.3f}", flush=True)

        # LLM: Response text
        elif isinstance(frame, TextFrame) and frame.text and self.current_metrics:
            # Only log once per turn
            if not hasattr(self, '_logged_response') or not self._logged_response:
                print(f"[ASSISTANT] {frame.text}", flush=True)
                self._logged_response = True

        # LLM: Finished
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self.current_metrics:
                self.current_metrics.llm_end_time = now
            self._logged_response = False

        # TTS: First audio byte
        elif isinstance(frame, TTSAudioRawFrame):
            if self.current_metrics and self.current_metrics.tts_first_audio_time is None:
                self.current_metrics.tts_first_audio_time = now
                tts_lat = self.current_metrics.tts_latency()
                e2e_lat = self.current_metrics.e2e_latency()

                if tts_lat is not None:
                    print(f"[TTS] First audio @ {now:.3f} (TTS TTFA: {tts_lat*1000:.0f}ms)", flush=True)
                else:
                    print(f"[TTS] First audio @ {now:.3f}", flush=True)

                if e2e_lat is not None:
                    print(f"\n{'='*60}", flush=True)
                    print(f"[E2E LATENCY] {e2e_lat*1000:.0f}ms total", flush=True)
                    self._print_latency_breakdown()
                    print(f"{'='*60}\n", flush=True)

                # Store for averaging
                self.latency_history.append(self.current_metrics)
                self.turn_count += 1

        # TTS: Finished speaking
        elif isinstance(frame, TTSStoppedFrame):
            if self.current_metrics:
                self.current_metrics.tts_end_time = now
                if self.turn_count > 0 and self.turn_count % 3 == 0:
                    self._print_average_latencies()

        # User started speaking (potential interruption)
        elif isinstance(frame, UserStartedSpeakingFrame):
            print(f"\n[VAD] User started speaking...", flush=True)

        await self.push_frame(frame, direction)

    def _print_latency_breakdown(self):
        if not self.current_metrics:
            return

        m = self.current_metrics
        stt = m.stt_latency() or 0
        llm = m.llm_latency() or 0
        tts = m.tts_latency() or 0
        e2e = m.e2e_latency() or 0

        # Visual bar chart
        scale = 50  # characters for max bar
        max_lat = max(stt, llm, tts, 0.001)

        def bar(val):
            width = int((val / max_lat) * scale) if max_lat > 0 else 0
            return "█" * width

        print(f"  STT:  {stt*1000:6.0f}ms {bar(stt)}", flush=True)
        print(f"  LLM:  {llm*1000:6.0f}ms {bar(llm)} ({self.model_name})", flush=True)
        print(f"  TTS:  {tts*1000:6.0f}ms {bar(tts)}", flush=True)

        # Target indicator
        if e2e < 1.0:
            status = "🟢 EXCELLENT"
        elif e2e < 1.5:
            status = "🟡 GOOD"
        elif e2e < 2.0:
            status = "🟠 ACCEPTABLE"
        else:
            status = "🔴 NEEDS OPTIMIZATION"
        print(f"  Status: {status} (target: <1500ms)", flush=True)

    def _print_average_latencies(self):
        if len(self.latency_history) < 2:
            return

        avg_stt = sum(m.stt_latency() or 0 for m in self.latency_history) / len(self.latency_history)
        avg_llm = sum(m.llm_latency() or 0 for m in self.latency_history) / len(self.latency_history)
        avg_tts = sum(m.tts_latency() or 0 for m in self.latency_history) / len(self.latency_history)
        avg_e2e = sum(m.e2e_latency() or 0 for m in self.latency_history) / len(self.latency_history)

        print(f"\n{'─'*60}", flush=True)
        print(f"[AVERAGE LATENCY] Over {len(self.latency_history)} turns:", flush=True)
        print(f"  STT: {avg_stt*1000:.0f}ms | LLM: {avg_llm*1000:.0f}ms | TTS: {avg_tts*1000:.0f}ms", flush=True)
        print(f"  E2E Average: {avg_e2e*1000:.0f}ms", flush=True)
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
    print("  STT  = Time from VAD silence to transcription", flush=True)
    print("  LLM  = Time from transcription to first LLM token", flush=True)
    print("  TTS  = Time from first LLM token to first audio byte", flush=True)
    print("  E2E  = Total time from user stops speaking to hearing response", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Configure local audio with optimized VAD
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5,  # Faster response
                    min_volume=0.6,
                )
            ),
            vad_audio_passthrough=True,
        )
    )

    # Deepgram STT - Nova-2 is fast and accurate
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
    )

    # LLM - configurable model
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=args.model,
    )

    # ElevenLabs TTS - turbo model for speed
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=args.voice,
        model="eleven_turbo_v2",  # Fastest model
        optimize_streaming_latency=3,  # 0-4, higher = more optimized
    )

    # Latency tracker
    latency_tracker = LatencyTracker(model_name=args.model)

    # System prompt optimized for short responses
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses under 2 sentences. Be concise.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            latency_tracker,  # Track latencies here
            tts,
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
