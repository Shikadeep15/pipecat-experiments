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

### 05_smartturn_voice_bot.py (Project 3)
**Voice Bot with SmartTurn for intelligent turn detection.**

Pipeline: `Mic → VAD → SmartTurn → Deepgram STT → GPT-4o-mini → ElevenLabs TTS → Speaker`

SmartTurn uses AI to detect when you've actually finished speaking, not just
when you pause. This prevents the bot from interrupting during:
- Thinking pauses ("I want to order... hmm... maybe a pizza")
- Natural speech hesitations
- Mid-sentence breaths

```bash
python 05_smartturn_voice_bot.py
```

**Key Configuration:**
- VAD `stop_secs=0.2` (short, so SmartTurn can take over)
- SmartTurn analyzes ~65ms of audio to determine if user is done
- Works best with 16kHz mono PCM audio (up to 8 seconds)

**Verification:**
- Test with: "I want to order... hmm... maybe a pizza"
- Bot should wait for complete thought, not respond at "hmm"
- Logs show `[SMARTTURN] Analyzing if user finished their thought...`

### 06_latency_optimized_bot.py (Project 4)
**Latency-optimized voice bot with detailed timing breakdown.**

Tracks every stage of the pipeline:
- **STT**: Time from VAD silence to transcription
- **LLM**: Time from transcription to first token (TTFT)
- **TTS**: Time from first LLM token to first audio byte (TTFA)
- **E2E**: Total end-to-end latency

```bash
# Default (GPT-4o-mini - fastest)
python 06_latency_optimized_bot.py

# Compare with GPT-4o
python 06_latency_optimized_bot.py --model gpt-4o

# Compare with GPT-4-turbo
python 06_latency_optimized_bot.py --model gpt-4-turbo
```

**Sample Output:**
```
[VAD] User stopped speaking @ 1234567890.123
[STT] Transcription received @ 1234567890.456 (STT: 333ms)
[USER] What's the weather like today?
[LLM] First token @ 1234567890.789 (LLM TTFT: 333ms)
[ASSISTANT] I don't have access to weather data, but you can check...
[TTS] First audio @ 1234567891.012 (TTS TTFA: 223ms)

============================================================
[E2E LATENCY] 889ms total
  STT:    333ms ████████████████
  LLM:    333ms ████████████████ (gpt-4o-mini)
  TTS:    223ms ███████████
  Status: 🟢 EXCELLENT (target: <1500ms)
============================================================
```

**Latency Targets:**
| Status | E2E Latency | Description |
|--------|-------------|-------------|
| 🟢 EXCELLENT | <1000ms | Feels instant |
| 🟡 GOOD | 1000-1500ms | Natural conversation |
| 🟠 ACCEPTABLE | 1500-2000ms | Noticeable but usable |
| 🔴 NEEDS WORK | >2000ms | Feels slow |

### 07_function_calling_bot.py (Project 5)
**Voice bot with function calling / tool use.**

The LLM can call functions based on user queries:

| Function | Trigger | Example |
|----------|---------|---------|
| `get_current_time` | "What time is it?" | Returns current date/time |
| `tell_joke` | "Tell me a joke" | Returns random programmer joke |
| `lookup_order` | "Order status for 12345?" | Returns mock order info |

```bash
python 07_function_calling_bot.py
```

**Test Order IDs:**
- `12345` - Shipped (with tracking)
- `67890` - Processing
- `11111` - Delivered
- `99999` - Cancelled

**Sample Output:**
```
[TURN 1] USER: What time is it?
============================================================
[FUNCTION CALL] get_current_time
[FUNCTION RESULT] 08:45 AM on Tuesday, February 04, 2026
[LLM] Generating response...
[ASSISTANT] It's currently 8:45 AM on Tuesday, February 4th, 2026.
[TTS] Speaking...
```

**How Function Calling Works:**
1. User asks a question via voice
2. Deepgram transcribes the speech
3. GPT-4o-mini decides if a function is needed
4. If yes, function executes and returns result
5. LLM incorporates result into natural response
6. ElevenLabs speaks the response

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

### Project 3: SmartTurn Integration
- [x] SmartTurn model loaded successfully
- [x] Integrated with VAD (stop_secs=0.2)
- [x] Pipeline configured correctly
- [ ] Bot waits for complete thoughts
- [ ] Doesn't interrupt during thinking pauses
- [ ] Feels more natural than VAD alone

### Project 4: Latency Measurement & Optimization
- [x] Latency logging implemented (STT, LLM, TTS, E2E)
- [x] Visual breakdown with bar chart
- [x] Model comparison support (--model flag)
- [x] Average latency tracking (every 3 turns)
- [ ] Measured average latency under 1.5 seconds
- [ ] Compared GPT-4o-mini vs GPT-4o latency

### Project 5: Function Calling
- [x] get_current_time function implemented
- [x] tell_joke function implemented
- [x] lookup_order function implemented (mock data)
- [x] Functions registered with LLM
- [ ] Time function works via voice
- [ ] Joke function works via voice
- [ ] Order lookup works via voice
- [ ] General questions work without functions

## Resources

- [Pipecat Documentation](https://docs.pipecat.ai)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat/tree/main/examples)
- [ElevenLabs Voices](https://elevenlabs.io/voice-library)
- [Deepgram Nova Models](https://developers.deepgram.com/docs/models)
