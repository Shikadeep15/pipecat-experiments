"""
ElevenLabs TTS Example - Pipecat Local Audio
=============================================
Uses ElevenLabs for high-quality text-to-speech.

Run: python 02_elevenlabs_tts.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    # Set up local audio output
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))

    # Initialize ElevenLabs TTS service
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # George - deep US male voice
        model="eleven_turbo_v2",  # Fastest model
    )

    # Create pipeline: TTS -> Audio Output
    pipeline = Pipeline([tts, transport.output()])
    task = PipelineTask(pipeline)

    async def say_something():
        await asyncio.sleep(1)
        await task.queue_frames([
            TTSSpeakFrame("Hello! This is ElevenLabs text to speech running through Pipecat. Pretty cool, right?"),
            EndFrame()
        ])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)
    await asyncio.gather(runner.run(task), say_something())


if __name__ == "__main__":
    asyncio.run(main())
