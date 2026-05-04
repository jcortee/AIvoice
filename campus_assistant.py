import pyaudio
import numpy as np
import wave
import whisper
import time
import ollama
import json
import subprocess
import threading
import traceback
import re
from openwakeword.model import Model
from collections import deque

# =========================================================================
# CONFIGURATION
# =========================================================================

CHUNK = 1280
RATE = 16000
WAKE_THRESHOLD = 0.45
WINDOW_SIZE = 5
REQUIRED_IN_WINDOW = 3
WAKE_IGNORE_TIME = 4  # seconds after triggering
SILENCE_THRESHOLD = 500
SILENCE_SECONDS = 1.2

prev_score = 0
LOCKOUT = 1.5  # seconds

# File paths
PIPER_MODEL = "/home/jcorte/aivoice/en_US-hfc_female-medium.onnx"
PIPER_EXE = "/home/jcorte/aivoice/wakeword-env/bin/piper"
CAMPUS_JSON = "/home/jcorte/aivoice/campus.json"

# LLM Settings (like Be More Agent)
OLLAMA_OPTIONS = {
    'temperature': 0.2,
    'num_predict': 200,
    'max_tokens': 200,
    'num_thread': 4,
    'top_k': 40,
    'top_p': 0.9
}

# System prompt for campus assistant
SYSTEM_PROMPT = """You are a helpful NJIT campus assistant.

INSTRUCTIONS:
- Answer questions about buildings, locations, and campus information.
- Use ONLY the information provided about buildings.
- Answer naturally and concisely.
- Do not explain your reasoning, list buildings, or discuss the task.
- Answer directly and helpfully.
"""

# =========================================================================
# STATE MANAGEMENT
# =========================================================================

class AssistantState:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
current_state = AssistantState.IDLE

# =========================================================================
# INITIALIZATION
# =========================================================================

print("[INIT] Loading Wake Word Model...")
try:
    wake_model = Model(
        wakeword_models=["/home/jcorte/aivoice/models/hey_kiosk.tflite"],
        inference_framework="tflite"
    )
    print("[INIT] Wake Word Model Loaded.")
except Exception as e:
    print(f"[CRITICAL] Failed to load wake word model: {e}")
    exit(1)

print("[INIT] Loading Whisper Model...")
try:
    whisper_model = whisper.load_model("tiny")
    print("[INIT] Whisper Model Loaded.")
except Exception as e:
    print(f"[CRITICAL] Failed to load Whisper model: {e}")
    exit(1)

print("[INIT] Initializing Audio...")
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)
print("[INIT] Audio Stream Ready.")

# Recording state
last_trigger_time = 0
score_window = deque(maxlen=WINDOW_SIZE)
exiting = False

# =========================================================================
# HELPER: Load Campus Data
# =========================================================================

def load_campus_data():
    """Load and flatten campus building data with rich fields for the LLM."""
    try:
        with open(CAMPUS_JSON, "r") as f:
            data = json.load(f)
        print("[CAMPUS] Loaded campus.json", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load campus data: {e}", flush=True)
        return []

    flat_buildings = []

    def make_building_obj(item, fallback_key):
        name = item.get("name", fallback_key)
        info = item.get("info", "")
        location = item.get("location", "")

        aliases = item.get("aliases", []) or []
        amenities = item.get("amenities", []) or []
        keywords = item.get("keywords", []) or []
        contact = item.get("contact", {}) or {}
        hours = item.get("hours", {}) or {}

        # Build a rich text blob the LLM can reason over
        parts = [
            f"Name: {name}",
            f"Info: {info}",
            f"Location: {location}",
        ]

        if aliases:
            parts.append("Aliases: " + ", ".join(aliases))
        if amenities:
            parts.append("Amenities: " + ", ".join(amenities))
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords))
        if contact:
            contact_lines = [f"{k}: {v}" for k, v in contact.items()]
            parts.append("Contact: " + " | ".join(contact_lines))
        if hours:
            hours_lines = [f"{day}: {hrs}" for day, hrs in hours.items()]
            parts.append("Hours: " + " | ".join(hours_lines))

        full_text = "\n".join(parts)

        return {
            "name": name,
            "info": info,
            "location": location,
            "aliases": aliases,
            "amenities": amenities,
            "keywords": keywords,
            "contact": contact,
            "hours": hours,
            "full_text": full_text,
        }

    for key, entry in data.items():
        if isinstance(entry, list):
            for item in entry:
                flat_buildings.append(make_building_obj(item, key))
        else:
            flat_buildings.append(make_building_obj(entry, key))

    return flat_buildings


# =========================================================================
# HELPER: Set State (like BotGUI.set_state)
# =========================================================================

def set_state(state, msg=""):
    """Update assistant state and print status."""
    global current_state
    if state != current_state:
        current_state = state
    if msg:
        print(f"[STATE] {state.upper()}: {msg}", flush=True)

# =========================================================================
# THE BRAIN: LLM-powered Campus Answer (upgraded from get_kiosk_answer)
# =========================================================================

def get_campus_answer(user_text):
    """
    Generate campus answer using Ollama with campus context.
    Like chat_and_respond but text-only.
    """
    set_state(AssistantState.THINKING, "Processing...")
    
    campus_data = load_campus_data()
    
    if not campus_data:
        return "Sorry, I couldn't load campus information."
    
    # Build building list text
    buildings_text = "\n".join(
        f"• {b['name']}: {b['info']} " 
        f"(Location: {b['location']}) "
        f"(Contact: {', '.join(f'{k}: {v}' for k, v in b['contact'].items()) if b['contact'] else 'None'})"
        f"(Hours: {', '.join(f'{day}: {hrs}' for day, hrs in b['hours'].items()) if b['hours'] else 'None'})"
        for b in campus_data
    )
    

    # Construct prompt (like SYSTEM_PROMPT + context)
    prompt = f"""{SYSTEM_PROMPT}

CAMPUS BUILDINGS:
{buildings_text}

User: {user_text}

Answer:"""
    
    try:
        print("[LLM] Sending to Ollama...", flush=True)
        response = ollama.generate(
            model='llama3.2',
            prompt=prompt,
            options=OLLAMA_OPTIONS,
            stream=False
        )
        
        raw = response.get("response", "").strip()
        
        # Clean up common prefixes (like _stream_to_text cleanup)
        for prefix in ["user:", "assistant:", "answer:", "q:", "a:"]:
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix):].strip()
        
        print(f"[LLM] Answer: {raw}", flush=True)
        set_state(AssistantState.SPEAKING, "Speaking...")
        return raw
        
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        traceback.print_exc()
        set_state(AssistantState.ERROR, f"Brain Error: {str(e)[:40]}")
        return "Sorry, I'm having trouble thinking right now."

# =========================================================================
# THE MOUTH: TTS with Piper (upgraded from speak function)
# =========================================================================

def speak(text):
    """
    Text-to-speech using Piper.
    Improved with error handling and process management (like Be More Agent).
    """
    if not text or not text.strip():
        return
    # -----------------------------
    # PHONE NUMBER NORMALIZATION
    # ----------------------------
    def normalize_phone_numbers(t):
        def expand(match):
            number = match.group()
            # Remove hyphens/spaces
            digits = number.replace("-", "").replace(" ", "")
            
            # Digit-to-word map (0 becomes "oh")
            speak = {
                "0": "oh",
                "1": "one",
                "2": "two",
                "3": "three",
                "4": "four",
                "5": "five",
                "6": "six",
                "7": "seven",
                "8": "eight",
                "9": "nine"
            }
            
            # Split into groups: 3-3-4
            if len(digits) == 10:
                g1 = " ... ".join(speak[d] for d in digits[0:3])
                g2 = " ... ".join(speak[d] for d in digits[3:6])
                g3 = " ... ".join(speak[d] for d in digits[6:10])
                return f"{g1}, {g2}, {g3}"

            # Fallback: speak all digits spaced
            return " ... ".join(speak[d] for d in digits)

        return re.sub(r"\b(?:\d[\s\-]?){10}\b", expand, t)

    cleaned = normalize_phone_numbers(text)
    
    # Remove Markdown asterisks so Piper doesn't say "asterisk"
    cleaned = cleaned.replace("*", "")

    # -----------------------------
    # ACRONYM EXPANSION
    # -----------------------------
    acronyms = {
        "NJIT": "N.J.I.T",
        "ECEC": "E.C.E.C",
        "ECE": "E.C.E",
        "CKB": "C.K.B",
        "WEC": "W.E.C",
        "HCAD": "H.C.A.D",
        "GITC": "G.I.T.C",
        "LSEC": "L.S.E.C",
        "YCEES": "Y.C.E.E.S",
        "OARS": "O.A.R.S",
        "MTSM": "M.T.S.M",
        "EOP": "E.O.P",
        "TLC": "T.L.C",
        "PC": "P.C",
        "FMH": "F.M.H",
        "CAB": "C.A.B",
    }
    
    for abbr, expanded in acronyms.items():
        cleaned = re.sub(rf"\b{abbr}\b", expanded, cleaned)
        cleaned = re.sub(rf"\b{abbr.lower()}\b", expanded.lower(), cleaned)

    print(f"[PIPER] Speaking: {cleaned}", flush=True)
    
    current_process = None
    try:
        # Create echo process
        echo_process = subprocess.Popen(
            ['echo', cleaned],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        # Create piper process
        piper_process = subprocess.Popen(
            [PIPER_EXE, "--model", PIPER_MODEL, "--output_raw"],
            stdin=echo_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        current_process = piper_process
        
        # Create aplay process
        aplay_process = subprocess.Popen(
            ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw'],
            stdin=piper_process.stdout,
            stderr=subprocess.DEVNULL
        )
        
        # Close parent process pipes
        echo_process.stdout.close()
        piper_process.stdout.close()
        
        # Wait for aplay to finish
        aplay_process.wait()
        
        print("[PIPER] Speech complete.", flush=True)
        
    except Exception as e:
        print(f"[PIPER ERROR] {e}")
        traceback.print_exc()
    finally:
        # Clean up
        if current_process and current_process.poll() is None:
            try:
                current_process.terminate()
                current_process.wait(timeout=1)
            except:
                pass

# =========================================================================
# THE EARS: Record until Silence (upgraded from record_command)
# =========================================================================

def record_command(dynamic_threshold):
    """
    Record audio until silence detected.
    Improved with better silence detection (like record_voice_adaptive).
    """
    set_state(AssistantState.LISTENING, "Recording...")
    print("[AUDIO] Listening...", flush=True)
    
    frames = []
    ring_buffer = deque(maxlen=int(SILENCE_SECONDS * RATE / CHUNK))
    triggered = False
    max_record_time = 10  # seconds
    start_time = time.time()
    
    try:
        while True:
            if exiting:
                return None
            
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            current_volume = np.abs(audio_data).mean()
            ring_buffer.append(current_volume)
            
            # State machine: waiting for speech → silence detection
            if triggered:
                # Check if entire buffer is quiet
                if all(v < dynamic_threshold for v in ring_buffer):
                    print("[AUDIO] Silence detected. Stopping.", flush=True)
                    break
            else:
                # Waiting for speech to start
                if current_volume > dynamic_threshold:
                    print("[AUDIO] Speech detected.", flush=True)
                    triggered = True
            
            # Safety timeout
            if time.time() - start_time > max_record_time:
                print("[AUDIO] Max recording time reached.", flush=True)
                break
        
        # Save to file
        filename = "command.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"[AUDIO] Saved to {filename}", flush=True)
        return filename
        
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")
        traceback.print_exc()
        return None

# =========================================================================
# TRANSCRIPTION: Whisper STT (like transcribe_audio)
# =========================================================================

def transcribe_audio(filename):
    """Transcribe audio file using Whisper."""
    set_state(AssistantState.THINKING, "Transcribing...")
    
    try:
        print("[WHISPER] Transcribing...", flush=True)
        result = whisper_model.transcribe(filename, fp16=False, language="en")
        user_text = result["text"].strip()
        
        print(f"[WHISPER] Heard: '{user_text}'", flush=True)
        return user_text
        
    except Exception as e:
        print(f"[WHISPER ERROR] {e}")
        traceback.print_exc()
        return ""

# =========================================================================
# NOISE CALIBRATION (like warm_up_logic)
# =========================================================================

def calibrate_noise_floor():
    """Calibrate microphone noise floor."""
    print("[AUDIO] Calibrating noise floor... stay quiet.", flush=True)
    
    calibration_frames = []
    for _ in range(int(2 * RATE / CHUNK)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        calibration_frames.append(np.frombuffer(data, dtype=np.int16))
    
    noise_floor = np.abs(np.concatenate(calibration_frames)).mean()
    dynamic_threshold = noise_floor * 2.5
    
    print(f"[AUDIO] Noise floor: {noise_floor:.2f}, Threshold: {dynamic_threshold:.2f}", flush=True)
    return dynamic_threshold

# =========================================================================
# MAIN CONVERSATION LOOP (like safe_main_execution)
# =========================================================================

def main_loop():
    """Main event loop: listen → record → transcribe → respond → speak."""
    global last_trigger_time, exiting, prev_score

    
    set_state(AssistantState.IDLE, "Ready")
    print("\n[SYSTEM] Campus Assistant Ready. Say 'Hey Kiosk'...\n", flush=True)
    
    # Calibrate once at startup
    dynamic_threshold = calibrate_noise_floor()
    
    try:
        while not exiting:
            # Listen for wake word
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            prediction = wake_model.predict(audio_data)
            current_time = time.time()
            
            for wakeword, score in prediction.items():
                
                # Sliding window detection (like detect_wake_word_or_ptt)
                # HARD LOCKOUT
                if current_time - last_trigger_time >= LOCKOUT:

                    # RISING EDGE DETECTION
                    if prev_score < WAKE_THRESHOLD and score >= WAKE_THRESHOLD:
                        print(f"\n[WAKE] {wakeword} detected ({score:.2f})", flush=True)
                        last_trigger_time = current_time

                        # Record audio
                        audio_file = record_command(dynamic_threshold)

                        if audio_file:
                            user_text = transcribe_audio(audio_file)
                            if user_text:
                                print(f"[USER] {user_text}", flush=True)
                                answer = get_campus_answer(user_text)
                                speak(answer)

                        set_state(AssistantState.IDLE, "Ready")
                        print("\n[SYSTEM] Waiting for next command...\n", flush=True)

                prev_score = score

    
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...", flush=True)
        safe_shutdown()
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        traceback.print_exc()
        safe_shutdown()

# =========================================================================
# GRACEFUL SHUTDOWN (like safe_exit)
# =========================================================================

def safe_shutdown():
    """Clean shutdown."""
    global exiting
    exiting = True
    
    print("[SHUTDOWN] Closing audio stream...", flush=True)
    try:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    except Exception as e:
        print(f"[SHUTDOWN] Error: {e}")
    
    print("[SHUTDOWN] Campus Assistant stopped.", flush=True)

# =========================================================================
# ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NJIT Campus Voice Assistant v2.0")
    print("  Upgraded with modern patterns from Be More Agent")
    print("=" * 60)
    
    main_loop()
