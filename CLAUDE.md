# CLAUDE.md - Pipecat Voice Bot Project

## Project Overview

**Pipecat Voice Bot** - A real-time voice AI application using the actual Pipecat framework with:
- VAD (Voice Activity Detection) + SmartTurn for intelligent turn detection
- Deepgram Nova-3 for Speech-to-Text (NOT Nova-2!)
- OpenAI GPT-4o-mini for LLM with function calling
- ElevenLabs for Text-to-Speech
- WebRTC echo cancellation in browser

## Quick Start

```bash
# Install dependencies
pip install pipecat-ai[local-smart-turn-v3,websocket] fastapi uvicorn

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the Pipecat voice bot
python real_pipecat_bot.py
# Open http://localhost:5002
```

## Key Concepts

### VAD vs SmartTurn

**VAD (Voice Activity Detection):**
- Monitors audio energy/amplitude
- Detects when user starts/stops speaking based on silence
- Simple threshold-based: 300ms silence = user stopped
- Problem: Interrupts users who pause to think

**SmartTurn (LocalSmartTurnAnalyzerV3):**
- ML-based end-of-turn detection
- Runs AFTER VAD detects silence
- Analyzes audio features to predict if user is truly done
- 800ms timeout: waits for confident prediction or more speech
- Prevents interrupting mid-thought pauses like "um..." or thinking

**How they work together:**
```
User speaks → VAD detects speech
User pauses → VAD detects 300ms silence → triggers "stop"
           → SmartTurn analyzes: Is this a real stop?
              - If confident end-of-turn → proceed to STT
              - If uncertain → wait up to 800ms for more speech
              - If timeout → force end turn
```

### Why 800ms Wait?

The 800ms (`stop_secs=0.8`) is SmartTurn's fallback timeout. After VAD detects silence:
1. SmartTurn's ML model analyzes the audio
2. If model is confident user is done → proceed immediately
3. If model is uncertain → wait for either:
   - More speech from user (they were just pausing)
   - Confident prediction from model
   - Timeout reached (800ms) → force proceed

This prevents the bot from interrupting when users say "I want to order... hmm... maybe a pizza" - it waits for the complete thought.

## Environment Variables

Required in `.env`:
```
DEEPGRAM_API_KEY=your_key      # Deepgram API key (Nova-3 STT)
OPENAI_API_KEY=your_key        # OpenAI API key (GPT-4o-mini)
ELEVENLABS_API_KEY=your_key    # ElevenLabs API key (TTS)
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPECAT VOICE PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

Browser (WebRTC)          Server (Pipecat)              APIs
      │                         │                         │
      │ getUserMedia()          │                         │
      │ echoCancellation:true   │                         │
      │ sampleRate:16000        │                         │
      ▼                         │                         │
┌──────────┐  WebSocket    ┌─────────┐                    │
│ Mic Audio│ ───────────►  │   VAD   │ (Silero)           │
└──────────┘               │ 300ms   │                    │
                           └────┬────┘                    │
                                │ User stopped speaking   │
                                ▼                         │
                           ┌─────────┐                    │
                           │SmartTurn│ (ML model)         │
                           │ 800ms   │                    │
                           └────┬────┘                    │
                                │ Confirmed end-of-turn   │
                                ▼                         │
                           ┌─────────┐              ┌──────────┐
                           │   STT   │ ──────────►  │ Deepgram │
                           │         │              │ Nova-3   │
                           └────┬────┘              └──────────┘
                                │ Final transcript        │
                                ▼                         │
                           ┌─────────┐              ┌──────────┐
                           │   LLM   │ ──────────►  │  OpenAI  │
                           │         │  streaming   │GPT-4o-mini│
                           └────┬────┘              └──────────┘
                                │ Response text           │
                                ▼                         │
                           ┌─────────┐              ┌──────────┐
                           │   TTS   │ ──────────►  │ElevenLabs│
                           │         │  streaming   │ Turbo v2 │
                           └────┬────┘              └──────────┘
                                │ Audio chunks            │
      ┌──────────┐              │                         │
      │ Speaker  │ ◄────────────┘                         │
      └──────────┘                                        │
```

## Latency Breakdown

**Formula:** `End-to-End = VAD/SmartTurn + LLM TTFT + TTS TTFS`

| Component | Metric | Expected |
|-----------|--------|----------|
| VAD/SmartTurn | Time from silence to transcript | 100-800ms |
| LLM | TTFT (Time To First Token) | 300-600ms |
| TTS | TTFS (Time To First Speech) | 500-600ms |
| **Total E2E** | VAD stop → First audio | 900-2000ms |

**Note:** TTS TTFS is typically 500-600ms, not 200ms as sometimes quoted.

## Key Files

```
pipecat-experiments/
├── real_pipecat_bot.py       # Main Pipecat voice bot (USE THIS)
├── web_voice_bot_pipecat.py  # Legacy custom implementation
├── CLAUDE.md                 # This file
├── .env                      # API keys (not committed)
└── .env.example              # Template for API keys
```

## Pipecat Components Used

| Component | Pipecat Class | Purpose |
|-----------|---------------|---------|
| VAD | `SileroVADAnalyzer` | Detect speech/silence |
| SmartTurn | `LocalSmartTurnAnalyzerV3` | ML end-of-turn detection |
| STT | `DeepgramSTTService` | Speech to text (Nova-3) |
| LLM | `OpenAILLMService` | Generate responses |
| TTS | `ElevenLabsTTSService` | Text to speech |
| Pipeline | `Pipeline` | Connect processors |
| Task | `PipelineTask` | Run pipeline |

## Configuration

### VAD Settings
```python
VADParams(
    threshold=0.5,              # Speech detection confidence
    min_speech_duration_ms=250, # Min speech to trigger start
    min_silence_duration_ms=300 # Silence to trigger stop
)
```

### SmartTurn Settings
```python
SmartTurnParams(
    stop_secs=0.8,      # 800ms timeout after VAD stop
    pre_speech_ms=500   # Audio before speech to analyze
)
```

### Deepgram Settings
```python
LiveOptions(
    model="nova-3",          # Latest model (NOT nova-2!)
    interim_results=False,   # Only final transcripts
    endpointing=300,         # 300ms utterance boundary
    punctuate=True,
    smart_format=True
)
```

## Function Calling

The bot supports these functions:
- `get_current_time()` - Returns current date/time
- `tell_joke()` - Returns a random joke
- `lookup_order(order_id)` - Looks up order status (test IDs: 12345, 67890)

Example queries:
- "What time is it?"
- "Tell me a joke"
- "What's the status of order 12345?"

## Implementation Details

### Non-Streaming Input
Input is **NOT streaming** - the bot only triggers on the **final transcript** from Deepgram, not interim results:
```python
LiveOptions(
    interim_results=False,  # Only final transcripts
)
```

### Correct Latency Tracking
```python
# E2E = VAD/SmartTurn stop → TTS first audio
latency = {
    'vad_to_transcript_ms': transcript_time - vad_stop_time,
    'llm_ttft_ms': llm_first_token - llm_start,
    'tts_ttfs_ms': tts_first_audio - tts_start,
    'e2e_ms': tts_first_audio - vad_stop_time
}
```

## Troubleshooting

1. **No transcription:** Check DEEPGRAM_API_KEY is valid
2. **SmartTurn not loading:** Run `pip install pipecat-ai[local-smart-turn-v3]`
3. **High latency:** LLM is usually the bottleneck; consider response length
4. **Echo issues:** Ensure browser echoCancellation is enabled
5. **TTS fails:** Check ELEVENLABS_API_KEY quota

## References

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Deepgram Nova-3](https://deepgram.com/product/nova)
- [ElevenLabs API](https://elevenlabs.io/docs)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
