import os
import base64
from mcp.server.fastmcp import FastMCP
from main import process_video_optimized

server = FastMCP("football-mcp")

@server.tool()
def run_tracking(video_base64: str):
    """
    Runs the optimized football tracking pipeline via MCP.
    """
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    input_path = "uploads/mcp_input.mp4"
    output_path = "output/mcp_output.mp4"

    # 1️⃣ Save incoming video
    try:
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_base64))
    except Exception as e:
        return {"status": "error", "message": f"Video decode failed: {str(e)}"}

    # 2️⃣ Run your pipeline
    try:
        process_video_optimized(input_path, output_path)
    except Exception as e:
        return {"status": "error", "message": f"Processing failed: {str(e)}"}

    # 3️⃣ Encode the output video
    try:
        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        return {"status": "error", "message": f"Encoding failed: {str(e)}"}

    return {
        "status": "success",
        "output_video_base64": output_b64,
        "message": "Tracking completed successfully"
    }


server.run()
