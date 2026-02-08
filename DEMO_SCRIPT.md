# Voice AI Bot Demo Script

## Introduction (30 seconds)

> "Hi, I'm going to walk you through a real-time Voice AI Bot I built using Pipecat concepts. This bot has three key features that make it production-ready:
>
> 1. **WebRTC Echo Cancellation** - No headphones needed
> 2. **SmartTurn** - Waits for complete thoughts, not just silence
> 3. **Plivo Phone Integration** - Can receive actual phone calls
>
> Let me show you how each component works."

---

## Part 1: The Pipeline Architecture (1 minute)

> "Here's the complete voice AI pipeline:"

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VOICE AI PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Browser Microphone                                                │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────────┐                                               │
│   │  WebRTC Echo    │  ← Chrome's built-in echo cancellation        │
│   │  Cancellation   │    Filters out bot's voice from speakers      │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                               │
│   │  Deepgram STT   │  ← Nova-2 model, 16kHz, real-time streaming   │
│   │  (Speech→Text)  │    Returns interim + final transcriptions     │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                               │
│   │   SmartTurn     │  ← Waits for COMPLETE thoughts                │
│   │   Detector      │    Not just silence pauses                    │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                               │
│   │  OpenAI LLM     │  ← GPT-4o-mini for fast responses             │
│   │  (GPT-4o-mini)  │    ~300-500ms latency                         │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                               │
│   │  ElevenLabs TTS │  ← Neha voice (Indian English)                │
│   │  (Text→Speech)  │    eleven_turbo_v2, streaming mode            │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│      Browser Audio                                                  │
│      (Speakers)                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> "Each component is optimized for low latency. Let me explain each one."

---

## Part 2: WebRTC Echo Cancellation (1 minute)

> "The first problem with voice bots is echo. When the bot speaks through your speakers, your microphone picks it up, creating a feedback loop.
>
> Traditional solution: Use headphones.
>
> My solution: Use Chrome's WebRTC echo cancellation."

**Show the code:**
```javascript
navigator.mediaDevices.getUserMedia({
    audio: {
        echoCancellation: true,   // KEY FEATURE
        noiseSuppression: true,   // Reduces background noise
        autoGainControl: true,    // Normalizes volume
        sampleRate: 16000         // Matches Deepgram
    }
})
```

> "When `echoCancellation: true` is set, Chrome's WebRTC audio processing:
> 1. Listens to what's playing through speakers
> 2. Subtracts that audio from the microphone input
> 3. Only sends YOUR voice to the server
>
> This is the same technology used in Google Meet and Zoom."

---

## Part 3: SmartTurn - Intelligent Turn Detection (2 minutes)

> "The second problem is knowing WHEN the user has finished speaking.
>
> **Traditional VAD (Voice Activity Detection):**
> - Detects silence (e.g., 0.5 seconds of no sound)
> - Problem: Interrupts during thinking pauses
>
> **SmartTurn (What I implemented):**
> - Analyzes the CONTENT of speech
> - Waits for complete thoughts, not just silence"

**Show the SmartTurn logic:**
```python
class SmartTurnDetector:
    def is_thought_complete(self, text):
        # 1. Check for sentence-ending punctuation
        if text.endswith('.') or text.endswith('!') or text.endswith('?'):
            return True

        # 2. Check for incomplete indicators (user is still thinking)
        incomplete_words = ['um', 'uh', 'hmm', 'like', 'so', 'and', 'but', 'or']
        if text.split()[-1] in incomplete_words:
            return False  # WAIT for more

        # 3. Check for question patterns
        if starts_with_question_word and len(words) >= 4:
            return True

        return len(words) >= 5  # Default: complete if 5+ words
```

**Demo script:**
> "Let me demonstrate. I'll say a sentence with a thinking pause:"
>
> *Say:* "I want to order... hmm... maybe a pizza"
>
> "Notice how the bot WAITED until I finished the complete thought. It didn't interrupt at 'hmm'. That's SmartTurn in action."

**Compare with traditional VAD:**
| Scenario | Traditional VAD | SmartTurn |
|----------|-----------------|-----------|
| "I want to order... hmm... pizza" | Responds at "order..." | Waits for "pizza" |
| "What is... let me think... 2+2?" | Responds at "is..." | Waits for "2+2?" |
| "Hello!" | Responds immediately | Responds immediately |

---

## Part 4: Deepgram STT Configuration (1 minute)

> "For speech-to-text, I'm using Deepgram Nova-2, their most accurate model."

**Show the configuration:**
```python
# Deepgram WebSocket URL
'wss://api.deepgram.com/v1/listen?'
'model=nova-2&'           # Latest, most accurate model
'language=en&'            # English
'punctuate=true&'         # Add punctuation (helps SmartTurn)
'interim_results=true&'   # Show text as user speaks
'endpointing=300&'        # 300ms silence = end of utterance
'smart_format=true&'      # Format numbers, dates nicely
'encoding=linear16&'      # PCM audio format
'sample_rate=16000'       # 16kHz sample rate
```

> "Key features I'm using:
> 1. **interim_results** - Shows transcription in real-time as you speak
> 2. **punctuation** - Adds periods and question marks (helps SmartTurn)
> 3. **endpointing** - Deepgram's built-in silence detection
> 4. **speech_final** flag - Tells us when an utterance is complete"

---

## Part 5: OpenAI LLM Configuration (1 minute)

> "For the brain of the bot, I'm using GPT-4o-mini - it's fast and cost-effective."

**Show the configuration:**
```python
payload = {
    "model": "gpt-4o-mini",    # Fastest GPT-4 variant
    "messages": conversation_history,
    "max_tokens": 150,         # Keep responses short for voice
    "temperature": 0.7,        # Balanced creativity
}
```

**System prompt:**
```python
"You are Neha, a friendly and helpful voice assistant.
Keep your responses concise - 1-2 sentences maximum.
Be warm and conversational, like talking to a friend."
```

> "For voice AI, short responses are crucial. Nobody wants to listen to a 5-paragraph answer. 1-2 sentences is ideal."

---

## Part 6: ElevenLabs TTS Configuration (1 minute)

> "For text-to-speech, I'm using ElevenLabs with the Neha voice - a natural Indian English voice."

**Show the configuration:**
```python
payload = {
    'text': response_text,
    'model_id': 'eleven_turbo_v2',  # Optimized for speed
    'voice_settings': {
        'stability': 0.5,           # Balance consistency/expressiveness
        'similarity_boost': 0.75,   # Sound like the original voice
        'style': 0.0,               # Neutral style
        'use_speaker_boost': True   # Enhance clarity
    },
    'optimize_streaming_latency': 3  # Level 3 = aggressive optimization
}
```

> "Key optimization: `optimize_streaming_latency: 3`
> This tells ElevenLabs to start sending audio BEFORE the full synthesis is complete.
> Result: ~200-400ms time-to-first-audio instead of ~800-1500ms."

---

## Part 7: Latency Breakdown (1 minute)

> "Let me show you the latency metrics. Every response shows a breakdown:"

```
┌─────────────────────────────────────────┐
│         LATENCY BREAKDOWN               │
├─────────────────────────────────────────┤
│  LLM:        450ms   ████████           │
│  TTS TTFA:   280ms   █████              │
│  TTS Total:  890ms   ████████████████   │
│  End-to-End: 730ms   ████████████       │
├─────────────────────────────────────────┤
│  Status: EXCELLENT (<1 second)          │
└─────────────────────────────────────────┘
```

**Latency targets:**
| Status | E2E Latency | User Experience |
|--------|-------------|-----------------|
| Excellent | <1000ms | Feels instant |
| Good | 1000-1500ms | Natural conversation |
| Acceptable | 1500-2000ms | Noticeable but OK |
| Slow | >2000ms | Feels laggy |

> "We're consistently hitting under 1 second end-to-end, which feels like a natural conversation."

---

## Part 8: Plivo Phone Integration (1 minute)

> "Finally, this bot can receive actual phone calls through Plivo."

**Architecture:**
```
Phone Call → Plivo → Webhook → My Server → Voice Bot Pipeline
                                    ↓
                              Audio Stream
                                    ↓
                    Deepgram → LLM → ElevenLabs
                                    ↓
                              Audio Response
                                    ↓
                              Plivo → Phone
```

**Webhook endpoints:**
```python
@app.route('/plivo/answer')   # Handles incoming calls
@app.route('/plivo/hangup')   # Handles call termination
```

**Setup required:**
1. Get a Plivo phone number
2. Use ngrok to expose localhost: `ngrok http 5002`
3. Configure Plivo webhook: `https://YOUR_NGROK_URL/plivo/answer`
4. Call your Plivo number - Neha answers!

---

## Part 9: Live Demo (2 minutes)

> "Let me do a live demo. I'll open the web interface and have a conversation."

**Demo conversation:**
1. "Hello, what's your name?"
2. "I want to order... hmm... maybe a pizza" (test SmartTurn)
3. "Tell me a joke"
4. "What's the weather like?" (bot admits limitations)

**Point out during demo:**
- Real-time transcription appearing
- "(SmartTurn)" label when thought is complete
- Latency metrics after each response
- No echo even without headphones

---

## Conclusion (30 seconds)

> "To summarize, this Voice AI Bot has:
>
> 1. **WebRTC Echo Cancellation** - Works without headphones
> 2. **SmartTurn** - Waits for complete thoughts
> 3. **Sub-second Latency** - Feels like natural conversation
> 4. **Plivo Integration** - Can receive real phone calls
>
> The code is available at: github.com/Shikadeep15/pipecat-experiments
>
> Thanks for watching!"

---

## Technical Summary

| Component | Technology | Configuration |
|-----------|------------|---------------|
| Audio Capture | WebRTC | echoCancellation: true |
| STT | Deepgram Nova-2 | 16kHz, interim_results, punctuate |
| Turn Detection | SmartTurn | Custom implementation |
| LLM | GPT-4o-mini | max_tokens: 150, temp: 0.7 |
| TTS | ElevenLabs | eleven_turbo_v2, Neha voice |
| Phone | Plivo | Bidirectional streaming |

**Files:**
- `web_voice_bot_pipecat.py` - Main server
- `templates/voice_bot.html` - Web UI
- `static/audio-processor.js` - Audio capture worklet
