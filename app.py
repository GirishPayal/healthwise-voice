from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# ── LOAD CREDENTIALS FROM ENVIRONMENT ────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSISTANT_ID       = os.getenv("ASSISTANT_ID")
VOICE_ID           = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # default to Rachel

# ── HELPERS ─────────────────────────────────────────────────────────────────
def transcribe_audio(url):
    resp = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={
            "file": ('audio.mp3', requests.get(url).content),
            "model": (None, "whisper-1")
        }
    )
    return resp.json().get("text", "")

def run_assistant(message, thread_id=None):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # 1️⃣ Create thread if new
    if not thread_id:
        r = requests.post(
            "https://api.openai.com/v1/threads",
            headers=headers,
            json={"assistant": ASSISTANT_ID}
        )
        thread_id = r.json()["id"]

    # 2️⃣ Send user message
    requests.post(
        f"https://api.openai.com/v1/threads/{thread_id}/messages",
        headers=headers,
        json={"content": message}
    )

    # 3️⃣ Run the assistant
    run = requests.post(
        f"https://api.openai.com/v1/threads/{thread_id}/runs",
        headers=headers
    ).json()

    # 4️⃣ Grab the reply
    outputs = run.get("outputs", [])
    if outputs:
        reply = outputs[0]["message"]["content"]
    else:
        run_id = run["id"]
        while True:
            status = requests.get(
                f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                headers=headers
            ).json()
            if status["status"] == "succeeded":
                reply = status["outputs"][0]["message"]["content"]
                break

    return reply, thread_id

def synthesize_voice(text):
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "text": text,
            "voice_settings": {"stability":0.5, "similarity_boost":0.7}
        }
    )
    return resp.content

# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return "✅ HealthWise William (voice) is ONLINE!"

@app.route("/chat", methods=["POST"])
def chat():
    data      = request.json or {}
    audio_url = data.get("audio_url")
    thread_id = data.get("thread_id")

    if not audio_url:
        return jsonify({"error":"Please send 'audio_url' in JSON"}), 400

    transcript = transcribe_audio(audio_url)
    reply, tid  = run_assistant(transcript, thread_id)
    voice_bytes = synthesize_voice(reply)

    return (
        voice_bytes,
        200,
        {
            "Content-Type":"audio/mpeg",
            "X-Thread-ID": tid
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
