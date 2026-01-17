import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

# Configuration
LOG_FILE = Path("brain_guard.log")
HOST = "0.0.0.0"
PORT = 8001

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>BrainGuard Log Dashboard</title>
        <style>
            body { font-family: 'Consolas', 'Monaco', monospace; background: #1e1e1e; color: #d4d4d4; margin: 0; padding: 20px; }
            h1 { color: #569cd6; border-bottom: 1px solid #333; padding-bottom: 10px; }
            #logs { background: #000; padding: 15px; border-radius: 5px; height: 80vh; overflow-y: scroll; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; box-shadow: 0 0 10px rgba(0,0,0,0.5); }
            .log-line { border-bottom: 1px solid #333; padding: 2px 0; }
            .log-info { color: #4ec9b0; }
            .log-warn { color: #ce9178; }
            .log-error { color: #f44747; font-weight: bold; }
            #status { margin-bottom: 10px; font-weight: bold; }
            .connected { color: #6a9955; }
            .disconnected { color: #f44747; }
        </style>
    </head>
    <body>
        <h1>🧠 BrainGuard Live Logs</h1>
        <div id="status" class="disconnected">● Disconnected</div>
        <div id="logs"></div>
        <script>
            var ws = new WebSocket("ws://" + window.location.host + "/ws/logs");
            var logsDiv = document.getElementById("logs");
            var statusDiv = document.getElementById("status");

            ws.onopen = function() {
                statusDiv.innerText = "● Connected";
                statusDiv.className = "connected";
                logsDiv.innerHTML += "<div class='log-line'><i>--- Connection Established ---</i></div>";
            };

            ws.onmessage = function(event) {
                var line = event.data;
                var className = "log-line";
                if (line.includes("ERROR") || line.includes("CRITICAL")) className += " log-error";
                else if (line.includes("WARNING")) className += " log-warn";
                else if (line.includes("INFO")) className += " log-info";
                
                var div = document.createElement("div");
                div.className = className;
                div.innerText = line;
                logsDiv.appendChild(div);
                
                // Auto-scroll if near bottom
                if(logsDiv.scrollHeight - logsDiv.scrollTop < logsDiv.clientHeight + 100) {
                     logsDiv.scrollTop = logsDiv.scrollHeight;
                }
            };

            ws.onclose = function() {
                statusDiv.innerText = "● Disconnected";
                statusDiv.className = "disconnected";
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    fd = None
    try:
        # Initial tail (last 50 lines)
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-50:]
                for line in lines:
                    await websocket.send_text(line.strip())
        
        # Stream new lines
        if LOG_FILE.exists():
            fd = open(LOG_FILE, "r", encoding="utf-8", errors="ignore")
            fd.seek(0, os.SEEK_END)
            
            while True:
                line = fd.readline()
                if line:
                    await websocket.send_text(line.strip())
                else:
                    await asyncio.sleep(0.5)
        else:
            await websocket.send_text("Waiting for log file to be created...")
            while not LOG_FILE.exists():
                await asyncio.sleep(1)
            fd = open(LOG_FILE, "r", encoding="utf-8", errors="ignore")
            # Recursive call or just continue loop would be better, but simple restart logic here
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.close()
        except: pass
    finally:
        if fd: fd.close()

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Log Server on {HOST}:{PORT}...", flush=True)
    try:
        uvicorn.run(app, host=HOST, port=PORT)
    except Exception as e:
        print(f"Server crashed: {e}")
    print("Server stopped.")
