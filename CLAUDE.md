# CLAUDE.md - Pipecat Experiments

## Project Overview

**Pipecat Experiments** - A sandbox for learning and experimenting with Pipecat, an open-source framework for building voice and multimodal conversational AI applications.

## What is Pipecat?

Pipecat is a framework that makes it easy to build real-time voice AI pipelines by connecting:
- **STT (Speech-to-Text)**: Deepgram, Whisper, AssemblyAI
- **LLM (Language Models)**: OpenAI, Anthropic, local models
- **TTS (Text-to-Speech)**: ElevenLabs, Cartesia, PlayHT
- **Transports**: Local audio, WebRTC, Twilio, Daily

## Quick Start

```bash
# Install dependencies (requires Python 3.10+)
pip install -r requirements.txt

# On macOS, also install portaudio for PyAudio
brew install portaudio

# Test imports
python test_imports.py

# Run simple TTS example
python 01_simple_tts.py

# Run ElevenLabs TTS example
python 02_elevenlabs_tts.py

# Run full voice bot (STT + LLM + TTS)
python 03_voice_bot.py
```

## Environment Variables

Required in `.env`:

| Variable | Description | Get From |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT | https://platform.openai.com/api-keys |
| `DEEPGRAM_API_KEY` | Deepgram API key for STT | https://console.deepgram.com |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for TTS | https://elevenlabs.io |
| `CARTESIA_API_KEY` | Cartesia API key for low-latency TTS | https://cartesia.ai |

## Examples

### 01_simple_tts.py
Basic text-to-speech using Cartesia. Demonstrates:
- Local audio output transport
- TTS service initialization
- Pipeline construction
- Frame queuing

### 02_elevenlabs_tts.py
TTS using ElevenLabs with the turbo model. Demonstrates:
- ElevenLabs integration
- Voice selection
- Model configuration

### 03_voice_bot.py
Full conversational voice bot. Demonstrates:
- Microphone input with VAD (Voice Activity Detection)
- Speech-to-text with Deepgram
- LLM conversation with OpenAI
- Text-to-speech with ElevenLabs
- Bidirectional audio pipeline
- Interruption handling

### 04_local_voice_bot.py (Project 2)
**Local Voice Bot with full verification logging.**

Pipeline: `Microphone → SileroVAD → Deepgram STT → GPT-4o-mini → ElevenLabs TTS → Speaker`

Features:
- Voice Activity Detection (SileroVAD) for natural turn-taking
- Interruption handling (speak while bot talks to stop it)
- Conversation logging with latency metrics
- System prompt: "You are a helpful assistant. Keep responses under 2 sentences."

```bash
python 04_local_voice_bot.py
```

**Verification:**
- Bot starts and listens
- Transcription works (logged as `[TURN N] USER: ...`)
- LLM generates response (logged as `[ASSISTANT] ...`)
- TTS plays through speakers (logged as `[TTS] Speaking...`)
- Interruption handling works (logged as `[INTERRUPT]`)
- Latency displayed (`[LATENCY] Time to first response: X.XXs`)

## Pipecat Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Pipeline                         │
│                                                     │
│  Input → STT → Context → LLM → TTS → Output        │
│    │      │       │       │     │      │           │
│  [Mic]  [Deepgram] [Memory] [GPT] [11Labs] [Speaker]│
└─────────────────────────────────────────────────────┘
```

### Key Concepts

- **Frames**: Data units flowing through the pipeline (audio, text, control)
- **Processors**: Transform or react to frames
- **Services**: External API integrations (STT, LLM, TTS)
- **Transports**: Handle I/O (audio devices, WebRTC, telephony)
- **Pipeline**: Connects processors in sequence
- **PipelineTask**: Manages a running pipeline

## Common Frame Types

| Frame | Purpose |
|-------|---------|
| `AudioRawFrame` | Raw audio data |
| `TranscriptionFrame` | STT output text |
| `TextFrame` | Generic text |
| `LLMMessagesFrame` | Messages for LLM |
| `TTSSpeakFrame` | Text to synthesize |
| `EndFrame` | Signals pipeline completion |

## Voice Options

### ElevenLabs Voices
| Name | Voice ID | Description |
|------|----------|-------------|
| George | JBFqnCBsd6RMkjVDRZzb | US male (deep) |
| Jessica | cgSgspJ2msm6clMCkdW9 | US female |
| Riya | Zs2gGSc3xT4kRfIqS9R3 | Indian English female |

### Cartesia Voices
| Name | Voice ID | Description |
|------|----------|-------------|
| British Lady | 71a7ad14-091c-4e8e-a314-022ece01c121 | British accent |
| Friendly Australian | a38e4e85-e815-4c3c-9b7c-a0b08d0e0b74 | Australian accent |

## Troubleshooting

### PyAudio Installation Issues
```bash
# macOS
brew install portaudio
pip install pyaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### No Audio Output
- Check system volume and output device
- Verify API keys are set in `.env`
- Run with DEBUG logging to see errors

### High Latency
- Use Cartesia TTS (40ms TTFA) instead of ElevenLabs
- Use `eleven_turbo_v2` model for ElevenLabs
- Enable VAD to reduce processing of silence

## Verification Checklist

### Project 1: Install Pipecat
- [x] Pipecat installed successfully (v0.0.101)
- [x] All dependencies resolved
- [x] .env file configured with API keys
- [x] Can import pipecat without errors (all 8 tests pass)

### Project 2: Local Voice Bot
- [x] Bot starts and listens
- [x] Deepgram STT connection works
- [x] ElevenLabs TTS connection works
- [x] Pipeline linked correctly
- [x] VAD (Silero) loaded
- [x] Interruption handling enabled (`allow_interruptions=True`)
- [ ] 5-turn conversation completed
- [ ] Latency under 2 seconds

## Resources

- [Pipecat Documentation](https://docs.pipecat.ai)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat/tree/main/examples)
- [ElevenLabs Voices](https://elevenlabs.io/voice-library)
- [Deepgram Nova Models](https://developers.deepgram.com/docs/models)
