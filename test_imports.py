"""
Test Pipecat Imports
====================
Simple script to verify Pipecat is installed correctly.

Run: python test_imports.py
"""

import sys

def test_imports():
    print("Testing Pipecat imports...\n")

    tests = []

    # Core imports
    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        tests.append(("Core Pipeline", True, None))
    except ImportError as e:
        tests.append(("Core Pipeline", False, str(e)))

    # Frames
    try:
        from pipecat.frames.frames import EndFrame, TTSSpeakFrame, LLMMessagesFrame
        tests.append(("Frames", True, None))
    except ImportError as e:
        tests.append(("Frames", False, str(e)))

    # Local Transport
    try:
        from pipecat.transports.local.audio import LocalAudioTransport
        tests.append(("Local Audio Transport", True, None))
    except ImportError as e:
        tests.append(("Local Audio Transport", False, str(e)))

    # Deepgram STT
    try:
        from pipecat.services.deepgram.stt import DeepgramSTTService
        tests.append(("Deepgram STT", True, None))
    except ImportError as e:
        tests.append(("Deepgram STT", False, str(e)))

    # OpenAI LLM
    try:
        from pipecat.services.openai.llm import OpenAILLMService
        tests.append(("OpenAI LLM", True, None))
    except ImportError as e:
        tests.append(("OpenAI LLM", False, str(e)))

    # ElevenLabs TTS
    try:
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
        tests.append(("ElevenLabs TTS", True, None))
    except ImportError as e:
        tests.append(("ElevenLabs TTS", False, str(e)))

    # Cartesia TTS
    try:
        from pipecat.services.cartesia.tts import CartesiaTTSService
        tests.append(("Cartesia TTS", True, None))
    except ImportError as e:
        tests.append(("Cartesia TTS", False, str(e)))

    # Silero VAD
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        tests.append(("Silero VAD", True, None))
    except ImportError as e:
        tests.append(("Silero VAD", False, str(e)))

    # Print results
    print("=" * 50)
    print("Import Test Results")
    print("=" * 50)

    all_passed = True
    for name, passed, error in tests:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        if error:
            print(f"       Error: {error}")
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("\nAll imports successful! Pipecat is ready to use.")
        return 0
    else:
        print("\nSome imports failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())
