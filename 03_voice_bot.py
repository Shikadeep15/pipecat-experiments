"""
Voice Bot Example - Full Conversational AI
==========================================
Complete voice pipeline: Mic -> STT -> LLM -> TTS -> Speaker

This example demonstrates:
- Deepgram for speech-to-text
- OpenAI GPT for conversation
- ElevenLabs for text-to-speech
- Local audio I/O via PyAudio

Run: python 03_voice_bot.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    # Configure local audio with VAD for natural conversation
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # Speech-to-Text: Deepgram Nova
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
    )

    # LLM: OpenAI GPT-4
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

    # Set up conversation context
    messages = [
        {
            "role": "system",
            "content": """You are a helpful voice assistant. Keep your responses brief and conversational
            since they will be spoken aloud. Be friendly and engaging. Your name is Aria.""",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Build the pipeline: Audio In -> STT -> LLM -> TTS -> Audio Out
    pipeline = Pipeline([
        transport.input(),           # Microphone input
        stt,                         # Speech-to-text
        context_aggregator.user(),   # Accumulate user messages
        llm,                         # Language model
        tts,                         # Text-to-speech
        transport.output(),          # Speaker output
        context_aggregator.assistant(),  # Accumulate assistant messages
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        # Greet the user when they connect
        await task.queue_frames([
            LLMMessagesFrame([
                {"role": "system", "content": "Greet the user warmly and ask how you can help them today."}
            ])
        ])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("\n" + "="*50)
    print("Voice Bot Started!")
    print("Speak into your microphone to chat.")
    print("Press Ctrl+C to exit.")
    print("="*50 + "\n")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
