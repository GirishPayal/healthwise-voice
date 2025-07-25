from flask import Flask, request, render_template, make_response, jsonify
import requests, os

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load keys from environment ───────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID           = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

# ── Helpers ─────────────────────────────────────────────────────────────────
def transcribe_audio(file_bytes):
    r = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={
            "file": ("audio.mp3", file_bytes),
            "model": (None, "whisper-1")
        }
    )
    return r.json().get("text", "")

def generate_reply(prompt):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are William, a friendly diabetes health assistant."},
                {"role": "user", "content": prompt}
            ]
        }
    )
    return r.json()["choices"][0]["message"]["content"]

def synthesize_voice(text):
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        },
        json={"text": text, "voice_settings": {"stability":0.3, "similarity_boost":0.5}}
    )
    return r.content

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if "audio" not in request.files:
        return jsonify({"error":"No audio file provided"}), 400

    audio_bytes = request.files["audio"].read()
    transcript  = transcribe_audio(audio_bytes)
    reply_text  = generate_reply(transcript)
    audio_out   = synthesize_voice(reply_text)

    resp = make_response(audio_out)
    resp.headers.set("Content-Type","audio/mpeg")
    return resp

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
