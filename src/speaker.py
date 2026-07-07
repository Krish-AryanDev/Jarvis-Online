"""
speaker.py — JARVIS Text-to-Speech Module

Strategy: pre-synthesize every known JARVIS response at startup using
edge-tts, cache the decoded PCM audio in memory, then play from cache
instantly with sounddevice — zero network delay at runtime.

If edge-tts is unavailable, pyttsx3 is used as an offline fallback.

Public API
----------
    speak(text: str)   — enqueue text for playback (returns immediately)
    shutdown()         — graceful stop (call on app exit)
"""

import asyncio
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------

_USE_EDGE_TTS = False
try:
    import edge_tts
    import miniaudio
    _USE_EDGE_TTS = True
    print('[Speaker] edge-tts + miniaudio ready — pre-caching responses...')
except ImportError:
    print('[Speaker] edge-tts/miniaudio not available — using pyttsx3 fallback')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EDGE_VOICE  = 'en-GB-RyanNeural'   # British English neural voice
EDGE_RATE   = '+0%'                 # '+10%' faster, '-10%' slower
PYTTSX_RATE = 160                   # words per minute (pyttsx3 fallback)

# ---------------------------------------------------------------------------
# All JARVIS spoken responses — single source of truth
# Add new entries here whenever a new command is added.
# ---------------------------------------------------------------------------

RESPONSES = {
    'At your service, sir.'                             : None,
    'Initialising systems. Boot sequence engaged.'      : None,
    'Heads-up display activated.'                       : None,
    'Heads-up display disabled.'                        : None,
    'Heads-up display toggled.'                         : None,
    'Scanning face.'                                    : None,
    'Night vision toggled.'                             : None,
    'Thermal mode toggled.'                             : None,
    'Recording started.'                                : None,
    'Recording stopped.'                                : None,
    'Screenshot captured.'                              : None,
    'Face tracking toggled.'                            : None,
    'Diagnostics toggled.'                              : None,
    'Goodbye, sir.'                                     : None,
}

# ---------------------------------------------------------------------------
# Audio cache  { text -> np.ndarray (PCM int16, 1-ch) }
# ---------------------------------------------------------------------------

_audio_cache: dict[str, tuple[np.ndarray, int]] = {}   # text -> (pcm, samplerate)
_cache_ready = threading.Event()                         # set once pre-caching is done

# ---------------------------------------------------------------------------
# Speech queue & playback thread
# ---------------------------------------------------------------------------

_speech_queue: queue.Queue = queue.Queue()


# ---------------------------------------------------------------------------
# edge-tts engine
# ---------------------------------------------------------------------------

async def _synth_one(text: str) -> Optional[tuple[np.ndarray, int]]:
    """Synthesise *text* via edge-tts → decode MP3 → return PCM array."""
    import miniaudio
    audio_bytes = b''
    communicate = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE)
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            audio_bytes += chunk['data']
    if not audio_bytes:
        return None
    decoded = miniaudio.decode(
        audio_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
    )
    pcm = np.frombuffer(decoded.samples, dtype=np.int16)
    return pcm, decoded.sample_rate


async def _precache_all():
    """Synthesise every response in RESPONSES concurrently at startup."""
    tasks = {text: asyncio.create_task(_synth_one(text)) for text in RESPONSES}
    for text, task in tasks.items():
        try:
            result = await task
            if result is not None:
                _audio_cache[text] = result
        except Exception as e:
            print(f'[Speaker] Pre-cache failed for "{text[:30]}...": {e}')
    _cache_ready.set()
    print(f'[Speaker] Pre-cache complete — {len(_audio_cache)}/{len(RESPONSES)} responses cached')


def _edge_tts_worker():
    """Background thread: runs asyncio loop for pre-caching + runtime synthesis."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Pre-cache all responses immediately
    loop.run_until_complete(_precache_all())

    async def _play_or_synth(text: str):
        """Use cache if available, else synthesise on-demand."""
        if text in _audio_cache:
            pcm, sr = _audio_cache[text]
        else:
            result = await _synth_one(text)
            if result is None:
                return
            pcm, sr = result
            _audio_cache[text] = (pcm, sr)   # cache for next time

        # Play with sounddevice (blocking in this thread — fine, it's the worker)
        sd.play(pcm.astype(np.float32) / 32768.0, samplerate=sr)
        sd.wait()

    while True:
        text = _speech_queue.get()
        if text is None:    # sentinel — shutdown
            break
        loop.run_until_complete(_play_or_synth(text))
        _speech_queue.task_done()


# ---------------------------------------------------------------------------
# pyttsx3 fallback engine
# ---------------------------------------------------------------------------

def _pyttsx3_worker():
    """Offline fallback: pyttsx3 synthesis (no pre-caching, no network)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', PYTTSX_RATE)

        # Prefer British English voice if available
        for voice in engine.getProperty('voices'):
            name = voice.name.lower()
            if 'english' in name and ('united kingdom' in name or 'british' in name):
                engine.setProperty('voice', voice.id)
                print(f'[Speaker] pyttsx3 voice: {voice.name}')
                break

        _cache_ready.set()  # no pre-caching needed

        while True:
            text = _speech_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f'[Speaker] pyttsx3 error: {e}')
            _speech_queue.task_done()

    except Exception as e:
        print(f'[Speaker] pyttsx3 init failed: {e}')
        _cache_ready.set()


# ---------------------------------------------------------------------------
# Start the worker daemon thread at import time
# ---------------------------------------------------------------------------

_worker_fn = _edge_tts_worker if _USE_EDGE_TTS else _pyttsx3_worker

_worker_thread = threading.Thread(
    target=_worker_fn,
    daemon=True,
    name='JarvisSpeaker',
)
_worker_thread.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak(text: str) -> None:
    """
    Enqueue *text* for JARVIS to speak.

    Returns immediately — playback happens on a background thread so the
    camera loop, HUD, and voice recognition are never blocked.

    If pre-caching is still in progress the request is queued and plays
    as soon as the cache is ready.
    """
    if text:
        _speech_queue.put(text)


def shutdown() -> None:
    """Gracefully stop the speaker thread (send sentinel, then join)."""
    _speech_queue.put(None)
    _worker_thread.join(timeout=5)
