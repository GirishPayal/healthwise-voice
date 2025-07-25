from flask import Flask, request, jsonify, render_template, make_response
import requests, os

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load secrets from env ───────────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSISTANT_ID       = os.getenv("ASSISTANT_ID")
VOICE_ID           = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# ── Helpers ─────────────────────────────────────────────────────────────────
def transcribe_audio_file(file_bytes):
    resp = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={"file": ("audio.mp3", file_bytes), "model": (None, "whisper-1")}
    )
    return resp.json().get("text","")

def run_assistant(message, thread_id=None):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    # 1️⃣ Create a new thread if none exists
    if not thread_id:
        r = requests.post(
            f"https://api.openai.com/v1/assistants/{ASSISTANT_ID}/threads",
            headers=headers
        )
        thread_id = r.json()["id"]

    # 2️⃣ Post the user message
    requests.post(
        f"https://api.openai.com/v1/assistants/{ASSISTANT_ID}/threads/{thread_id}/messages",
        headers=headers,
        json={"content": message}
    )

    # 3️⃣ Run the assistant on that thread
    run = requests.post(
        f"https://api.openai.com/v1/assistants/{ASSISTANT_ID}/threads/{thread_id}/runs",
        headers=headers
    ).json()

    # 4️⃣ Extract the reply
    outputs = run.get("outputs", [])
    if outputs:
        reply = outputs[0]["message"]["content"]
    else:
        # fallback: poll until completed
        run_id = run["id"]
        while True:
            status = requests.get(
                f"https://api.openai.com/v1/assistants/{ASSISTANT_ID}/threads/{thread_id}/runs/{run_id}",
                headers=headers
            ).json()
            if status["status"] == "succeeded":
                reply = status["outputs"][0]["message"]["content"]
                break

    return reply, thread_id

def synthesize_voice(text):
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={"text": text, "voice_settings":{"stability":0.3,"similarity_boost":0.5}}
    )
    return resp.content

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if "audio" not in request.files:
        return jsonify({"error":"No file part 'audio'"}), 400
    audio_bytes = request.files["audio"].read()

    # 1. Transcribe
    transcript = transcribe_audio_file(audio_bytes)

    # 2. Assistant reply + thread
    reply, tid = run_assistant(transcript, request.form.get("thread_id"))

    # 3. TTS
    voice_data = synthesize_voice(reply)

    # 4. Return audio + thread header
    resp = make_response(voice_data)
    resp.headers.set("Content-Type", "audio/mpeg")
    resp.headers.set("X-Thread-ID", tid)
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",10000)))
