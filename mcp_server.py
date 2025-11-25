from flask import Flask, request, jsonify
import base64
import os
from main import process_video_optimized

app = Flask(__name__)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return {"status": "MCP API Running", "message": "Use POST /run-tracking"}, 200


@app.route("/run-tracking", methods=["POST"])
def run_tracking():
    try:
        data = request.get_json()

        if not data or "video_base64" not in data:
            return jsonify({"error": "video_base64 missing"}), 400

        # 1️⃣ Decode incoming video
        input_path = os.path.join(UPLOAD_DIR, "input_api.mp4")
        output_path = os.path.join(OUTPUT_DIR, "output_api.mp4")

        with open(input_path, "wb") as f:
            f.write(base64.b64decode(data["video_base64"]))

        # 2️⃣ Run your pipeline
        try:
            process_video_optimized(input_path, output_path)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Processing failed: {str(e)}"}), 500

        # 3️⃣ Encode output video to base64
        with open(output_path, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode()

        return jsonify({
            "status": "success",
            "output_video_base64": out_b64,
            "message": "Processing completed"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
