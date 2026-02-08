# Loom Demo Script: Pipecat Voice AI Projects

## Video Title
**"Building Production-Ready Voice AI with Pipecat: Latency Tracking & Function Calling"**

## Estimated Duration: 8-10 minutes

---

## PART 1: INTRODUCTION (1 minute)

### Opening Shot: Terminal with project folder

**[SPEAK]:**
> "Hey everyone! Today I'm going to show you two voice AI projects I built using Pipecat - an open-source framework for building real-time conversational AI.
>
> I'll demonstrate:
> 1. A **latency-optimized voice bot** with detailed performance tracking
> 2. A **function-calling voice bot** that can execute real tools
>
> Both use the same tech stack: Deepgram for speech-to-text, GPT-4o-mini for the LLM, and ElevenLabs for text-to-speech."

### Show the architecture diagram

**[SPEAK]:**
> "The pipeline flows like this: Your microphone captures audio, Silero VAD detects when you're speaking, Deepgram transcribes your speech, GPT-4o-mini generates a response, and ElevenLabs speaks it back through your speakers."

```
Mic → VAD → Deepgram STT → GPT-4o-mini → ElevenLabs TTS → Speaker
```

---

## PART 2: LATENCY-OPTIMIZED BOT DEMO (3-4 minutes)

### Show the code briefly

**[ACTION]:** Open `06_latency_optimized_bot.py` in editor

**[SPEAK]:**
> "This bot tracks every stage of the voice AI pipeline. Let me show you the key metrics we're measuring:"

**[HIGHLIGHT these sections]:**
- `LatencyMetrics` dataclass (lines 72-100)
- The latency breakdown visualization (lines 185-216)

**[SPEAK]:**
> "We track:
> - **STT latency**: Time from when you stop speaking to getting the transcription
> - **LLM TTFT**: Time to first token from the language model
> - **TTS TTFA**: Time to first audio byte from text-to-speech
> - **E2E latency**: Total end-to-end time - the most important metric"

### Run the demo

**[ACTION]:** Run the bot in terminal

```bash
cd /Users/deepshika/Desktop/pipecat-experiments && python3 06_latency_optimized_bot.py
```

**[SPEAK]:**
> "Let me start the bot... You can see it shows the target latency is under 1500 milliseconds for a natural conversation."

### Demo Conversation 1: Simple question

**[SPEAK TO BOT]:** "Hello, how are you today?"

**[WAIT for response, then explain the output]:**

**[SPEAK]:**
> "Look at the output - you can see the exact latency breakdown:
> - STT took about X milliseconds
> - LLM time to first token was X milliseconds
> - TTS time to first audio was X milliseconds
> - Total end-to-end was X milliseconds
>
> The visual bar chart shows which component is the bottleneck."

### Demo Conversation 2: Another question

**[SPEAK TO BOT]:** "What can you help me with?"

**[WAIT for response]**

**[SPEAK]:**
> "Notice the status indicator - green means excellent, under 1 second. Yellow is good, orange is acceptable, and red means we need optimization."

### Demo Conversation 3: Test interruption

**[SPEAK TO BOT]:** "Tell me a long story about..."
**[INTERRUPT while bot is speaking]:** "Actually, stop!"

**[SPEAK]:**
> "The bot handles interruptions naturally - it stops speaking when it detects you've started talking again."

### Show average latencies

**[SPEAK]:**
> "Every 3 turns, it shows average latencies so you can track performance over time."

**[ACTION]:** Press Ctrl+C to exit

---

## PART 3: FUNCTION CALLING BOT DEMO (3-4 minutes)

### Transition

**[SPEAK]:**
> "Now let's look at something more powerful - a voice bot that can actually DO things, not just chat. This bot has three functions: getting the current time, telling jokes, and looking up order status."

### Show the code briefly

**[ACTION]:** Open `07_function_calling_bot.py` in editor

**[HIGHLIGHT these sections]:**
- Function schemas (lines 159-192)
- Function handlers (lines 199-232)
- Mock order database (lines 89-115)

**[SPEAK]:**
> "Each function has a schema that tells GPT-4o-mini when to use it. The handlers execute the actual logic and return results to the LLM, which then speaks them naturally."

### Run the demo

**[ACTION]:** Run the bot

```bash
cd /Users/deepshika/Desktop/pipecat-experiments && python3 07_function_calling_bot.py
```

### Demo Function 1: Time

**[SPEAK TO BOT]:** "What time is it?"

**[WAIT for response]**

**[SPEAK]:**
> "Watch the terminal - you can see `[FUNCTION CALL] get_current_time` was triggered. The function returned the actual time, and the bot spoke it naturally."

### Demo Function 2: Joke

**[SPEAK TO BOT]:** "Tell me a joke"

**[WAIT for response]**

**[SPEAK]:**
> "The `tell_joke` function was called, it picked a random programmer joke, and the LLM delivered it conversationally."

### Demo Function 3: Order Lookup (multiple scenarios)

**[SPEAK TO BOT]:** "What's the status of order 12345?"

**[WAIT for response]**

**[SPEAK]:**
> "This is the `lookup_order` function - it found order 12345 which is shipped and gave us the tracking number and delivery date."

**[SPEAK TO BOT]:** "Can you check on order 11111?"

**[WAIT for response]**

**[SPEAK]:**
> "That one shows as delivered. Let's try an invalid order..."

**[SPEAK TO BOT]:** "What about order 55555?"

**[WAIT for response]**

**[SPEAK]:**
> "It gracefully handles the error and tells us the order wasn't found."

### Demo: General question (no function needed)

**[SPEAK TO BOT]:** "How are you doing today?"

**[WAIT for response]**

**[SPEAK]:**
> "Notice no function was called - the LLM knows when it needs tools versus when it can just answer directly."

**[ACTION]:** Press Ctrl+C to exit

---

## PART 4: WRAP-UP (1 minute)

### Summary

**[SPEAK]:**
> "So that's two practical voice AI applications built with Pipecat:
>
> 1. A **latency tracker** that helps you optimize for real-time performance - critical for production voice apps
> 2. A **function-calling bot** that can integrate with real APIs and databases
>
> The key takeaways:
> - **Sub-second E2E latency** is achievable with the right stack
> - **Function calling** lets voice bots do real work, not just chat
> - **Pipecat** makes it easy to swap components - different STT, LLM, or TTS providers
>
> All the code is in my GitHub. Thanks for watching!"

---

## TECHNICAL NOTES FOR RECORDING

### Before Recording
1. Verify `.env` has valid API keys
2. Test both bots work
3. Close unnecessary apps to reduce background noise
4. Use headphones to prevent echo

### Terminal Settings
- Font size: 14-16pt (readable on screen)
- Dark theme for better contrast
- Window size: ~120 columns wide

### Demo Order IDs for Function Bot
| Order ID | Status | Good for Demo |
|----------|--------|---------------|
| 12345 | Shipped | Yes - has tracking |
| 67890 | Processing | Yes - shows pending |
| 11111 | Delivered | Yes - shows completion |
| 99999 | Cancelled | Yes - shows refund |
| 55555 | Not found | Yes - shows error handling |

### Backup Phrases if Recognition Issues
- "What is the current time?"
- "Can you tell me a funny joke?"
- "Please check on order number one two three four five"

### Expected Latency Ranges
- **STT**: 200-400ms (Deepgram Nova-2)
- **LLM**: 200-500ms (GPT-4o-mini TTFT)
- **TTS**: 150-300ms (ElevenLabs turbo)
- **E2E**: 600-1200ms total
