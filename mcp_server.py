import sys
import os
import base64
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from football_core import tracker_api

server = FastMCP("football-mcp")

@server.tool()
def run_tracking(video_base64: str):
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    input_path = "uploads/mcp_input.mp4"

    try:
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_base64))
    except Exception as e:
        return {"status": "error", "message": str(e)}

    result = tracker_api.process_video_safe(input_path)

    if "processed_video_url" not in result:
        return {"status": "error", "message": result.get("error", "unknown error")}

    output_path = f"output/{result['processed_video_url']}"

    with open(output_path, "rb") as f:
        output_b64 = base64.b64encode(f.read()).decode()

    return {
        "status": "success",
        "tracking_data": result,
        "output_video_base64": output_b64
    }

server.run()
