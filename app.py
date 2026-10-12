import asyncio
import datetime
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
import speech_recognition as sr
from dotenv import load_dotenv
from flask import Flask, Response, request, send_from_directory

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from stt import WhisperSTT

load_dotenv()

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:8080/v1")


# --- Tools ---------------------------------------------------------------------

@tool
def get_current_datetime() -> str:
    """Use this tool to get the current local date and time."""
    return datetime.datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")


@tool
def get_weather(location: str) -> str:
    """Use this tool to get the current weather for a city or location name."""
    try:
        resp = requests.get(f"https://wttr.in/{location}?format=3", timeout=10)
        return resp.text.strip() if resp.ok else f"Couldn't fetch weather for {location}."
    except requests.RequestException as e:
        return f"Weather lookup failed: {e}"


llm = ChatOpenAI(
    base_url=LM_STUDIO_URL,
    api_key="lm-studio",
    model="qwen3-0.6b",
    streaming=True,
    temperature=0.4,
)

agent = create_agent(
    model=llm,
    tools=[get_current_datetime, get_weather],
    system_prompt="Your name is FRIDAY & You are a concise, helpful voice assistant. Keep spoken replies short.",
)

# --- TTS: Kokoro-82M via audio.cpp (models/audio.cpp), same subprocess-per-call
# pattern as whisper-cli in stt.py -------------------------------------------

AUDIOCPP_CLI = os.getenv(
    "AUDIOCPP_CLI", "./models/audio.cpp/build/macos-metal-release/bin/audiocpp_cli"
)
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "models/kokoro-82m-bf16.gguf")
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "af_heart")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en-us")
TTS_BACKEND = os.getenv("TTS_BACKEND", "metal")

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# max_workers=1: sentences must be synthesized strictly in order (a single worker
# guarantees FIFO), otherwise a short later sentence could finish before a longer
# earlier one and play out of sequence. Synthesis still overlaps with playback
# since this worker runs independently of both the agent stream and the frontend.
tts_executor = ThreadPoolExecutor(max_workers=1)
_tts_cache: dict = {}
_tts_lock = threading.Lock()


def synthesize_speech(text: str) -> bytes:
    out_path = f"tts_{uuid.uuid4().hex}.wav"
    cmd = [
        AUDIOCPP_CLI, "--task", "tts", "--family", "kokoro_tts",
        "--model", KOKORO_MODEL, "--backend", TTS_BACKEND,
        "--language", TTS_LANGUAGE, "--voice-id", TTS_VOICE_ID,
        "--text", text, "--out", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def _speak(text: str, gen: int, cancel_event: threading.Event):
    text = text.strip()
    if not text or cancel_event.is_set():
        return
    try:
        audio_bytes = synthesize_speech(text)
    except Exception as e:
        print(f"TTS failed: {e}")
        return
    if cancel_event.is_set():
        return
    audio_id = uuid.uuid4().hex
    with _tts_lock:
        _tts_cache[audio_id] = audio_bytes
    push({"type": "audio", "id": audio_id, "gen": gen})


app = Flask(__name__, static_folder="static", static_url_path="")

sse_queue: "queue.Queue[str]" = queue.Queue()
state_lock = threading.Lock()
history: list = []
current = {"gen": 0, "text": "", "cancel": None, "thread": None}


def push(event: dict):
    sse_queue.put(json.dumps(event))


def handle_input(new_text: str):
    """Entry point for any newly transcribed text, from either mic mode.

    If a request is already in flight, kill it and re-send the combined text
    (old request text + the new chunk) instead of starting an unrelated turn.
    """
    new_text = new_text.strip()
    if not new_text:
        return

    with state_lock:
        if current["thread"] and current["thread"].is_alive():
            current["cancel"].set()
            combined = f"{current['text']} {new_text}".strip()
        else:
            combined = new_text

        current["gen"] += 1
        gen = current["gen"]
        cancel_event = threading.Event()
        current["text"] = combined
        current["cancel"] = cancel_event

        t = threading.Thread(target=_run_agent, args=(combined, gen, cancel_event), daemon=True)
        current["thread"] = t
        t.start()

    push({"type": "transcript", "text": combined, "gen": gen})


def _run_agent(user_text: str, gen: int, cancel_event: threading.Event):
    asyncio.run(_run_agent_async(user_text, gen, cancel_event))


async def _run_agent_async(user_text: str, gen: int, cancel_event: threading.Event):
    messages = list(history) + [HumanMessage(content=user_text)]
    final_text = ""
    speak_buffer = ""
    agen = agent.astream_events({"messages": messages}, version="v2")
    try:
        async for ev in agen:
            if cancel_event.is_set():
                break

            kind = ev["event"]
            if kind == "on_chat_model_stream":
                chunk = ev["data"]["chunk"].content
                if chunk:
                    final_text += chunk
                    push({"type": "token", "text": chunk, "gen": gen})

                    speak_buffer += chunk
                    sentences = _SENTENCE_END_RE.split(speak_buffer)
                    for sentence in sentences[:-1]:
                        tts_executor.submit(_speak, sentence, gen, cancel_event)
                    speak_buffer = sentences[-1]
            elif kind == "on_tool_start":
                push({"type": "tool", "name": ev.get("name", "tool"), "gen": gen})
    except Exception as e:
        push({"type": "error", "text": str(e), "gen": gen})
        return
    finally:
        await agen.aclose()

    if cancel_event.is_set():
        push({"type": "cancelled", "gen": gen})
        return

    tts_executor.submit(_speak, speak_buffer, gen, cancel_event)

    with state_lock:
        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=final_text))
        if current["gen"] == gen:
            current["thread"] = None

    push({"type": "done", "gen": gen})


# --- Mic input: always-listening + push-to-talk share one WhisperSTT -----------

_stt_lock = threading.Lock()
_stt_instance = None


class LiveWhisperSTT(WhisperSTT):
    """App-layer wrapper around WhisperSTT: keeps mic capture (listen_in_background)
    running continuously instead of blocking it on turn-detection/transcription.

    Captured utterances are handed off to a single worker thread so audio keeps being
    picked up while a previous utterance is still being turn-detected/transcribed (true
    parallelism, not consecutive). Turn-detection calls themselves stay strictly ordered
    on that one worker thread, since WhisperSTT's audio_past stitching assumes turns are
    processed in the order they were spoken. Does not alter any WhisperSTT recognition
    logic (turn_detection/process_audio/speaker_verified) — only how it's dispatched.
    """

    def __init__(self, on_transcript, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_transcript = on_transcript
        self._audio_queue: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def callback(self, recognizer, audio):
        # Runs on speech_recognition's capture thread: hand off and return immediately.
        self._audio_queue.put(audio)

    def _worker(self):
        while True:
            audio = self._audio_queue.get()
            self._transcribe(audio)

    def _transcribe(self, audio):
        transcribed_file = f"transcribed_{uuid.uuid4().hex}.wav"
        try:
            with open(transcribed_file, "wb") as f:
                f.write(audio.get_wav_data())

            output = self.turn_detection(transcribed_file)

            clean_output = str(output).replace(" ", "").replace(".", "").replace(",", "").lower()
            if clean_output and clean_output != "you":
                print("(Whisper) You Said: " + output)
                self._on_transcript(output)

        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        finally:
            if os.path.exists(transcribed_file):
                os.remove(transcribed_file)

    def listen(self, stop_event=None):
        print('Listening (mode: WhisperCPP)...')
        r = sr.Recognizer()
        with sr.Microphone(sample_rate=16000) as source:
            r.adjust_for_ambient_noise(source)

        stop_listening = r.listen_in_background(source, self.callback)
        try:
            while not (stop_event and stop_event.is_set()):
                time.sleep(0.1)
        finally:
            stop_listening(wait_for_stop=False)


# Desired turn-detection config, applied to the STT instance whenever it exists
# (and used as the initial config when it's first created).
turn_config = {"primary": "smartturn", "only_primary": True}


def _push_turn_status(predictions: dict):
    push({"type": "turn_status", "predictions": predictions})


def get_stt() -> WhisperSTT:
    global _stt_instance
    with _stt_lock:
        if _stt_instance is None:
            _stt_instance = LiveWhisperSTT(handle_input, turn_type=turn_config["primary"])
            _stt_instance.use_only_primary = turn_config["only_primary"]
            _stt_instance.on_turn_status = _push_turn_status
    return _stt_instance


class ListeningController:
    def __init__(self):
        self.stop_event = None
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=get_stt().listen, args=(self.stop_event,), daemon=True)
        self.thread.start()

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
        self.thread = None


listening_ctl = ListeningController()


# --- Routes ----------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/tts/<audio_id>")
def tts_audio(audio_id):
    with _tts_lock:
        data = _tts_cache.pop(audio_id, None)
    if data is None:
        return "", 404
    return Response(data, mimetype="audio/wav")


@app.route("/api/stream")
def stream():
    def gen():
        while True:
            yield f"data: {sse_queue.get()}\n\n"
    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/turn-config", methods=["POST"])
def turn_config_route():
    data = request.get_json(force=True) or {}
    if data.get("primary") in ("smartturn", "turnsense"):
        turn_config["primary"] = data["primary"]
        if _stt_instance is not None:
            _stt_instance.primary_model = data["primary"]
    if "only_primary" in data:
        turn_config["only_primary"] = bool(data["only_primary"])
        if _stt_instance is not None:
            _stt_instance.use_only_primary = turn_config["only_primary"]
    return {"ok": True, **turn_config}


@app.route("/api/listen/start", methods=["POST"])
def listen_start():
    listening_ctl.start()
    return {"ok": True}


@app.route("/api/listen/stop", methods=["POST"])
def listen_stop():
    listening_ctl.stop()
    return {"ok": True}


@app.route("/api/record", methods=["POST"])
def record():
    audio_file = request.files["audio"]
    path = f"push_to_talk_{uuid.uuid4().hex}.wav"
    audio_file.save(path)
    try:
        text = get_stt().process_audio(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    handle_input(text)
    return {"ok": True, "text": text}


if __name__ == "__main__":
    app.run(port=5001, debug=True, threaded=True)
