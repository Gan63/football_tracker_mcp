import os
import base64
import traceback
from flask import Flask, request, jsonify
from main import process_video_optimized

# ============================================================
# 🔥 Disable ALL Ultralytics internet, GitHub, and version checks
# ============================================================

os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_NO_VERIFY"] = "1"
os.environ["ULTRALYTICS_HUB"] = "0"
os.environ["YOLO_DISABLE_UPDATE"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import ultralytics
ultralytics.checks.check_yolo = lambda *a, **k: None
ultralytics.checks.check_version = lambda *a, **k: None
ultralytics.checks.check_latest_pip_version = lambda *a, **k: None

# ============================================================
# Flask App Configuration
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Health Check Route
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return {"status": "MCP API Running", "message": "Use POST /run-tracking"}, 200

# ============================================================
# MAIN TRACKING ENDPOINT
# ============================================================

@app.route("/run-tracking", methods=["POST"])
def run_tracking():
    try:
        data = request.get_json()

        if not data or "video_base64" not in data:
            return jsonify({"error": "video_base64 missing"}), 400

        input_path = os.path.join(UPLOAD_DIR, "input_api.mp4")
        output_path = os.path.join(OUTPUT_DIR, "output_api.avi")  # AVI SAFE

        # -------------------------------------------------------
        # 1️⃣ Decode Base64 → Save Video File
        # -------------------------------------------------------
        try:
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(data["video_base64"]))
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": "Video decode failed",
                "details": str(e)
            }), 500

        # -------------------------------------------------------
        # 2️⃣ Run Football Tracking Pipeline
        # -------------------------------------------------------
        try:
            process_video_optimized(input_path, output_path)
        except Exception:
            return jsonify({
                "status": "error",
                "message": "Processing failed",
                "details": traceback.format_exc()
            }), 500

        # -------------------------------------------------------
        # 3️⃣ Read Output File and Return as Base64
        # -------------------------------------------------------
        try:
            with open(output_path, "rb") as f:
                out_b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": "Output read failed",
                "details": str(e)
            }), 500

        return jsonify({
            "status": "success",
            "output_video_base64": out_b64
        })

    except Exception:
        return jsonify({
            "status": "error",
            "message": "Fatal server error",
            "details": traceback.format_exc()
        }), 500

# ============================================================
# Local Debug Run (Render uses Gunicorn)
# ============================================================

if __name__ == "__main__":
    print("⚡ MCP Server Running at http://0.0.0.0:10000")
    app.run(host="0.0.0.0", port=10000)
