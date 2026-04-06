import pyaudio
import numpy as np
import wave
import whisper
import time
import ollama
import json
import subprocess
import shlex # We need to "sanitize" the AI's answer so that special characters (like apostrophes or quotes) don't break the Linux command.
import re
from openwakeword.model import Model
from collections import deque

# -----------------------------
# CONFIGURATION
# -----------------------------
CHUNK = 1280
RATE = 16000

# Path to your voice and piper files
PIPER_MODEL = "/home/jcorte/aivoice/en_US-hfc_female-medium.onnx"
PIPER_EXE = "/home/jcorte/aivoice/wakeword-env/bin/piper"
CAMPUS_JSON = "/home/jcorte/aivoice/campus.json"

# Wakeword detection
WAKE_THRESHOLD = 0.45
WINDOW_SIZE = 5
REQUIRED_IN_WINDOW = 3
WAKE_IGNORE_TIME = 4  # seconds after triggering

# Silence detection
SILENCE_THRESHOLD = 500
SILENCE_SECONDS = 1.2

# Recording state
last_trigger_time = 0
score_window = deque(maxlen=WINDOW_SIZE)

# -----------------------------
# LOAD MODELS
# -----------------------------
wake_model = Model(
    wakeword_models=["/home/jcorte/aivoice/models/hey_kiosk.tflite"],
    inference_framework="tflite"
)

whisper_model = whisper.load_model("tiny")

print("System Ready. Say 'Hey Kiosk'...")

# -----------------------------
# MICROPHONE SETUP
# -----------------------------
audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)


# --- FUNCTION: The Brain ---

import re

def get_kiosk_answer(user_text):
    # Load campus data
    try:
        with open(CAMPUS_JSON, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return f"Error loading campus data: {e}"

    # Normalize user text
    user_text_low = user_text.lower().strip()
    clean = re.sub(r"[^a-z0-9\s]", " ", user_text_low).strip()
    user_tokens = clean.split()

    # STRICT GREETING DETECTION (G1)
    greeting_set = {
        "hi", "hello", "hey",
        "good morning", "good afternoon", "good evening",
        "whats up", "what's up", "what s up", "what is up"
    }

    if clean in greeting_set:
        return "Hi! Welcome to NJIT. How can I help you today?"

    # -----------------------------------------
    # B3 HYBRID BUILDING DETECTION
    # -----------------------------------------

    # Extract candidate building name patterns
    building_patterns = [
        r"where is (the )?(.+?) building",
        r"where is (the )?(.+?) hall",
        r"where is (the )?(.+?) center",
        r"where is (the )?(.+?) lab",
        r"where is (the )?(.+?) room",
        r"where is (the )?(.+?)$"
    ]

    extracted_name = None

    for pattern in building_patterns:
        m = re.search(pattern, clean)
        if m:
            extracted_name = m.group(2).strip()
            break

    # Normalize extracted name
    if extracted_name:
        extracted_name = extracted_name.lower()

        # Map JSON keys to normalized names
        json_names = {key.lower(): key for key in data.keys()}

        # Direct match
        if extracted_name in json_names:
            key = json_names[extracted_name]
            entry = data[key]
        else:
            # Unknown building → F4 fallback
            return "I don’t have that location on file — check the map for official campus buildings."

        # Build context for TinyLlama
        name = entry.get("name", key)
        info = entry.get("info", "")
        location = entry.get("location", "")
        connection = entry.get("connection", "")

        context = f"{name}. {info} {location} {connection}".strip()

        prompt = f"""{context}

{user_text}

Answer in one short natural sentence, using only the information above:
"""

        response = ollama.generate(
            model='tinyllama',
            prompt=prompt,
            options={'temperature': 0.1, 'num_predict': 40, 'top_k': 20}
        )

        raw = response.get("response", "").strip()

        for p in ["user:", "assistant:", "answer:", "q:", "a:"]:
            if raw.lower().startswith(p):
                raw = raw[len(p):].strip()

        raw = raw.strip('"').strip("'")

        if not raw or "virtual" in raw.lower() or "not a physical" in raw.lower():
            return "I don’t have that location on file — check the map for official campus buildings."

        return raw

    # -----------------------------------------
    # FACILITY SCORING (gym, pool, parking, study, bathrooms)
    # -----------------------------------------

    def score_entry(name, details):
        matches = []

        def score_terms(terms):
            score = 0
            for term in terms:
                term_clean = re.sub(r"[^a-z0-9\s]", " ", term)
                term_tokens = term_clean.split()

                # Exact match = 2 points
                if any(t == u for t in term_tokens for u in user_tokens):
                    score += 2

                # Partial match = 1 point
                elif any(t in u or u in t for t in term_tokens for u in user_tokens):
                    score += 1

            return score

        if isinstance(details, list):
            for item in details:
                building_name = item.get("name", name)
                terms = [name.lower(), building_name.lower()]
                terms.extend([k.lower() for k in item.get("keywords", [])])
                terms.extend([a.lower() for a in item.get("amenities", [])])

                s = score_terms(terms)
                if s > 0:
                    matches.append((s, item))
        else:
            building_name = details.get("name", name)
            terms = [name.lower(), building_name.lower()]
            terms.extend([k.lower() for k in details.get("keywords", [])])
            terms.extend([a.lower() for a in details.get("amenities", [])])

            s = score_terms(terms)
            if s > 0:
                matches.append((s, details))

        if not matches:
            return None

        return max(matches, key=lambda x: x[0]), len(matches)

    # Find best facility match
    best_score = 0
    best_entry = None
    multi_match_count = 0

    for key, details in data.items():
        result = score_entry(key, details)
        if result:
            (s, entry), count = result
            if s > best_score:
                best_score = s
                best_entry = (key, entry)
                multi_match_count = count

    if not best_entry:
        return "I don’t have that location on file — check the map for official campus buildings."

    key, entry = best_entry

    # Build context
    name = entry.get("name", key)
    info = entry.get("info", "")
    location = entry.get("location", "")
    connection = entry.get("connection", "")

    context = f"{name}. {info} {location} {connection}".strip()

    prompt = f"""{context}

{user_text}

Answer in one short natural sentence, using only the information above:
"""

    response = ollama.generate(
        model='tinyllama',
        prompt=prompt,
        options={'temperature': 0.1, 'num_predict': 40, 'top_k': 20}
    )

    raw = response.get("response", "").strip()

    for p in ["user:", "assistant:", "answer:", "q:", "a:"]:
        if raw.lower().startswith(p):
            raw = raw[len(p):].strip()

    raw = raw.strip('"').strip("'")

    if not raw or "virtual" in raw.lower() or "not a physical" in raw.lower():
        return "I don’t have that location on file — check the map for official campus buildings."

    # Multi-match message (C2)
    if multi_match_count > 1:
        raw = f"{raw}. There are other locations on campus — check the map for pins."

    return raw

    
# --- FUNCTION: The Mouth: Speech --- 
def speak(text):
    # 1. Clean up acronyms for the TTS engine
    text = text.replace("NJIT", "N.J.I.T").replace("njit", "n.j.i.t")
    text = text.replace("ECEC", "E.C.E.C").replace("ecec", "e.c.e.c")
    text = text.replace("ECE", "E.C.E").replace("ece", "e.c.e")
    text = text.replace("CKB", "C.K.B").replace("ckb", "c.k.b")
    
    print(f"Speaking: {text}")

    try:
        # 2. Define the individual parts of the command pipe
        # We use 'pg_echo' to pipe the text into piper
        echo_process = subprocess.Popen(['echo', text], stdout=subprocess.PIPE)
        
        # 3. Define the Piper command
        piper_cmd = [
            PIPER_EXE, 
            "--model", PIPER_MODEL, 
            "--output_raw"
        ]
        piper_process = subprocess.Popen(piper_cmd, stdin=echo_process.stdout, stdout=subprocess.PIPE)
        
        # 4. Define the Aplay command
        aplay_cmd = ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw']
        subprocess.run(aplay_cmd, stdin=piper_process.stdout)
        
        # Clean up
        echo_process.stdout.close()
        piper_process.stdout.close()

    except Exception as e:
        print(f"Error in speech synthesis: {e}")

# -----------------------------
# RECORD COMMAND UNTIL SILENCE
# -----------------------------
def record_command():
    print("Listening...")
    frames = []
    # We use a deque to keep track of the last 1 second of audio volume
    ring_buffer = deque(maxlen=int(SILENCE_SECONDS * RATE / CHUNK))
    triggered = False

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        audio_data = np.frombuffer(data, dtype=np.int16)
        current_volume = np.abs(audio_data).mean()
        ring_buffer.append(current_volume)

        # If the average volume in our buffer drops below the threshold
        if triggered:
            # Check if the ENTIRE buffer is now quiet
            if all(v < DYNAMIC_THRESHOLD for v in ring_buffer):
                print("End of speech detected.")
                break
        else:
            # Check if we've started speaking
            if current_volume > DYNAMIC_THRESHOLD:
                print("User is speaking...")
                triggered = True

        # Safety: Stop if recording exceeds 10 seconds
        if len(frames) > (10 * RATE / CHUNK):
            break

   # Replace the 'save_wave_file' line with this:
    filename = "command.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

# -----------------------------
# MAIN LOOP
# -----------------------------
# noise floor calibration
print("Calibrating for background noise... stay quiet.")
calibration_frames = []
for _ in range(int(2 * RATE / CHUNK)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    calibration_frames.append(np.frombuffer(data, dtype=np.int16))

# Calculate the average background volume
noise_floor = np.abs(np.concatenate(calibration_frames)).mean()
# Set your threshold to be 2.5x the background noise
DYNAMIC_THRESHOLD = noise_floor * 2.5
print(f"Calibration complete. Noise floor: {noise_floor:.2f}. Threshold set to: {DYNAMIC_THRESHOLD:.2f}")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        prediction = wake_model.predict(audio_data)
        current_time = time.time()

        for wakeword, score in prediction.items():

            # Ignore triggers during cooldown
            if current_time - last_trigger_time < WAKE_IGNORE_TIME:
                continue

            # Add score to sliding window
            score_window.append(score)

            # Check if enough frames in window are above threshold
            if sum(s > WAKE_THRESHOLD for s in score_window) >= REQUIRED_IN_WINDOW:

                print(f"\nWake word detected: {wakeword} ({score:.2f})")
                last_trigger_time = current_time
                score_window.clear()

                # Record command until user stops talking
                audio_file = record_command()

                # Transcribe command
                print("Transcribing...")
                result = whisper_model.transcribe(audio_file, fp16=False, language="en")
                user_text = result["text"].strip()
                print("User said:", user_text)

                # New step for Ollama
                if user_text:
                    answer = get_kiosk_answer(user_text)
                    print("Kiosk:", answer)
                    speak(answer)

                print("\nReturning to wake-word listening...\n")
                time.sleep(0.5)

except KeyboardInterrupt:
    print("\nShutting down...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
