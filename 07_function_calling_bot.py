"""
Project 5: Voice Bot with Function Calling
==========================================
Voice bot that can call functions based on user queries.

Functions:
- get_current_time: Returns the current time
- tell_joke: Returns a random joke
- lookup_order: Returns mock order status

Pipeline: Mic → VAD → Deepgram STT → GPT-4o-mini (with tools) → ElevenLabs TTS → Speaker

Run: python 07_function_calling_bot.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import json
import os
import sys
import time
import ssl
import certifi
import random
from datetime import datetime

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
    TranscriptionFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

# Collection of jokes
JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why did the developer go broke? Because he used up all his cache!",
    "What's a computer's favorite snack? Microchips!",
    "Why do Java developers wear glasses? Because they can't C#!",
    "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
    "Why was the JavaScript developer sad? Because he didn't Node how to Express himself!",
    "What do you call a computer that sings? A-Dell!",
    "Why did the computer go to the doctor? Because it had a virus!",
    "What's a robot's favorite type of music? Heavy metal!",
    "Why don't scientists trust atoms? Because they make up everything!",
]

# Mock order database
MOCK_ORDERS = {
    "12345": {
        "status": "shipped",
        "item": "Wireless Headphones",
        "shipped_date": "February 2, 2026",
        "estimated_delivery": "February 6, 2026",
        "tracking": "1Z999AA10123456784"
    },
    "67890": {
        "status": "processing",
        "item": "Smart Watch",
        "ordered_date": "February 3, 2026",
        "estimated_ship": "February 5, 2026"
    },
    "11111": {
        "status": "delivered",
        "item": "Bluetooth Speaker",
        "delivered_date": "January 30, 2026",
        "signed_by": "Front Door"
    },
    "99999": {
        "status": "cancelled",
        "item": "USB Cable",
        "cancelled_date": "February 1, 2026",
        "refund_status": "Refund processed"
    },
}


def get_current_time_impl() -> str:
    """Get the current time."""
    now = datetime.now()
    return now.strftime("%I:%M %p on %A, %B %d, %Y")


def tell_joke_impl() -> str:
    """Return a random joke."""
    return random.choice(JOKES)


def lookup_order_impl(order_id: str) -> str:
    """Look up order status by order ID."""
    order_id = str(order_id).strip()

    if order_id in MOCK_ORDERS:
        order = MOCK_ORDERS[order_id]
        status = order["status"]
        item = order["item"]

        if status == "shipped":
            return (f"Order {order_id} for {item} has been shipped on {order['shipped_date']}. "
                   f"Expected delivery is {order['estimated_delivery']}. "
                   f"Tracking number is {order['tracking']}.")
        elif status == "processing":
            return (f"Order {order_id} for {item} is currently being processed. "
                   f"It was ordered on {order['ordered_date']} and is expected to ship by {order['estimated_ship']}.")
        elif status == "delivered":
            return (f"Order {order_id} for {item} was delivered on {order['delivered_date']}. "
                   f"It was signed for at: {order['signed_by']}.")
        elif status == "cancelled":
            return (f"Order {order_id} for {item} was cancelled on {order['cancelled_date']}. "
                   f"{order['refund_status']}.")
    else:
        return f"I couldn't find order {order_id}. Please check the order number and try again. Valid test orders are: 12345, 67890, 11111, or 99999."


# =============================================================================
# FUNCTION SCHEMAS (Pipecat format)
# =============================================================================

get_current_time_schema = FunctionSchema(
    name="get_current_time",
    description="Get the current date and time. Use this when the user asks what time it is, what day it is, or wants to know the current date.",
    properties={},
    required=[],
)

tell_joke_schema = FunctionSchema(
    name="tell_joke",
    description="Tell a random joke. Use this when the user asks for a joke, wants to hear something funny, or asks you to make them laugh.",
    properties={},
    required=[],
)

lookup_order_schema = FunctionSchema(
    name="lookup_order",
    description="Look up the status of an order by its order ID. Use this when the user asks about an order status, tracking, or delivery information.",
    properties={
        "order_id": {
            "type": "string",
            "description": "The order ID to look up (e.g., '12345')"
        }
    },
    required=["order_id"],
)

# Create tools schema
tools = ToolsSchema(
    standard_tools=[
        get_current_time_schema,
        tell_joke_schema,
        lookup_order_schema,
    ]
)


# =============================================================================
# FUNCTION HANDLERS
# =============================================================================

async def handle_get_current_time(params: FunctionCallParams):
    """Handle get_current_time function call."""
    print(f"\n[FUNCTION CALL] get_current_time", flush=True)
    result = get_current_time_impl()
    print(f"[FUNCTION RESULT] {result}", flush=True)
    await params.result_callback(result)


async def handle_tell_joke(params: FunctionCallParams):
    """Handle tell_joke function call."""
    print(f"\n[FUNCTION CALL] tell_joke", flush=True)
    result = tell_joke_impl()
    print(f"[FUNCTION RESULT] {result}", flush=True)
    await params.result_callback(result)


async def handle_lookup_order(params: FunctionCallParams):
    """Handle lookup_order function call."""
    print(f"\n[FUNCTION CALL] lookup_order", flush=True)

    # Extract order_id from arguments
    args = params.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    order_id = args.get("order_id", "")
    print(f"[FUNCTION ARGS] order_id={order_id}", flush=True)

    result = lookup_order_impl(order_id)
    print(f"[FUNCTION RESULT] {result}", flush=True)
    await params.result_callback(result)


# =============================================================================
# CONVERSATION LOGGER
# =============================================================================

class FunctionCallLogger(FrameProcessor):
    """Logs conversation events with function call tracking."""

    def __init__(self):
        super().__init__()
        self.turn_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            print(f"\n[VAD] User speaking...", flush=True)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            print(f"[VAD] User stopped", flush=True)

        elif isinstance(frame, TranscriptionFrame):
            self.turn_count += 1
            print(f"\n{'='*60}", flush=True)
            print(f"[TURN {self.turn_count}] USER: {frame.text}", flush=True)
            print(f"{'='*60}", flush=True)

        elif isinstance(frame, LLMFullResponseStartFrame):
            print(f"[LLM] Generating response...", flush=True)

        elif isinstance(frame, TextFrame) and frame.text:
            print(f"[ASSISTANT] {frame.text}", flush=True)

        elif isinstance(frame, TTSStartedFrame):
            print(f"[TTS] Speaking...", flush=True)

        elif isinstance(frame, TTSStoppedFrame):
            print(f"[TTS] Done\n", flush=True)

        await self.push_frame(frame, direction)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    print("\n" + "=" * 70, flush=True)
    print("PROJECT 5: VOICE BOT WITH FUNCTION CALLING", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Available Functions:", flush=True)
    print("  1. get_current_time - Ask 'What time is it?'", flush=True)
    print("  2. tell_joke       - Ask 'Tell me a joke'", flush=True)
    print("  3. lookup_order    - Ask 'What's the status of order 12345?'", flush=True)
    print(flush=True)
    print("Test Order IDs: 12345, 67890, 11111, 99999", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Configure local audio
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.6, min_volume=0.6)
            ),
            vad_audio_passthrough=True,
        )
    )

    # Deepgram STT
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
    )

    # OpenAI LLM with function calling
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    # Register function handlers
    llm.register_function("get_current_time", handle_get_current_time)
    llm.register_function("tell_joke", handle_tell_joke)
    llm.register_function("lookup_order", handle_lookup_order)

    # ElevenLabs TTS
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model="eleven_turbo_v2",
    )

    # Logger
    conversation_logger = FunctionCallLogger()

    # System prompt
    messages = [
        {
            "role": "system",
            "content": """You are a helpful voice assistant with access to tools.

When a user asks for the time, use the get_current_time function.
When a user asks for a joke, use the tell_joke function.
When a user asks about an order status, use the lookup_order function.

For general questions that don't need these functions, just answer directly.
Keep your responses concise and natural for voice - under 2-3 sentences.
When reporting function results, speak them naturally as if you're having a conversation.""",
        },
    ]

    # Create context with tools
    context = OpenAILLMContext(messages=messages, tools=tools)
    context_aggregator = llm.create_context_aggregator(context)

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            conversation_logger,
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

    print("Bot initialized! Try these voice commands:", flush=True)
    print("  - 'What time is it?'", flush=True)
    print("  - 'Tell me a joke'", flush=True)
    print("  - 'What's the status of order 12345?'", flush=True)
    print("  - 'How are you?' (general question, no function)", flush=True)
    print("\nPress Ctrl+C to exit.\n", flush=True)

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
