// Offscreen document — maintains persistent WebSocket to the server.
// Relays CDP commands to background.js (service worker) via chrome.runtime messages.

const WS_URL = "ws://127.0.0.1:8000/ws/extension";
const AUTH_TOKEN = "proactiveclaw_browser_token";
const RECONNECT_DELAY_MS = 2000;

let ws = null;

function connectWS() {
  if (ws && ws.readyState <= 1) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log("[offscreen] WebSocket connected");
    ws.send(JSON.stringify({ token: AUTH_TOKEN }));
  };

  ws.onmessage = async (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      console.error("[offscreen] Bad message:", event.data);
      return;
    }

    if (msg.type === "cdp_command") {
      // Forward CDP command to background.js for chrome.debugger execution
      try {
        const response = await chrome.runtime.sendMessage({
          type: "cdp_command",
          id: msg.id,
          method: msg.method,
          params: msg.params || {},
        });
        // Send result back to server
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(response));
        }
      } catch (err) {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: "cdp_result",
            id: msg.id,
            error: err.message || String(err),
          }));
        }
      }
    }
  };

  ws.onclose = () => {
    console.log("[offscreen] WebSocket closed, reconnecting...");
    ws = null;
    setTimeout(connectWS, RECONNECT_DELAY_MS);
  };

  ws.onerror = (err) => {
    console.error("[offscreen] WebSocket error");
    ws.close();
  };
}

// Keep-alive: ping every 20s to prevent any idle timeouts
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "ping" }));
  } else {
    connectWS();
  }
}, 20000);

connectWS();
