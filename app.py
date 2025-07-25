from flask import Flask, request, jsonify, render_template, make_response
import requests, os

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load credentials from environment ───────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSISTANT_ID       = os.getenv("ASSISTANT_ID")
VOICE_ID           = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

# ── Helpers ─────────────────────────────────────────────────────────────────
def transcribe_audio(file_bytes):
    resp = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={
            "file": ("audio.mp3", file_bytes),
            "model": (None, "whisper-1")
        }
    )
    return resp.json().get("text", "")

def run_assistant(message, thread_id=None):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # 1️⃣ Create a new thread if none exists (global endpoint)
    if not thread_id:
        r = requests.post(
            "https://api.openai.com/v1/threads",
            headers=headers,
            json={"assistant": ASSISTANT_ID}
        )
        r.raise_for_status()
        thread_id = r.json()["id"]

    # 2️⃣ Post the user's message
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

    # 4️⃣ Extract the assistant's reply
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
            if status.get("status") == "succeeded":
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
            "voice_settings": {"stability": 0.3, "similarity_boost": 0.5}
        }
    )
    return resp.content

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_bytes = request.files["audio"].read()
    transcript  = transcribe_audio(audio_bytes)
    reply, tid   = run_assistant(transcript, request.form.get("thread_id"))
    voice_bytes  = synthesize_voice(reply)

    resp = make_response(voice_bytes)
    resp.headers.set("Content-Type", "audio/mpeg")
    resp.headers.set("X-Thread-ID", tid)
    return resp

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
