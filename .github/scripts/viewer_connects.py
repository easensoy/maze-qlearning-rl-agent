"""Check that the served viewer opens its WebSocket to the backend, in a real browser.

Expects a headless Chrome with --remote-debugging-port (default 9222) already running. Loads
the viewer, then polls the page text until the connection indicator, which the viewer sets on
the WebSocket's open event, reads "Connected". Only the standard library and `websockets` are
needed, so CI runs it in the backend image:

    docker run --rm -i --network host maze-backend python - < .github/scripts/viewer_connects.py

Environment: VIEWER_URL (default http://localhost:3001/), CDP_PORT (default 9222).
"""
import json
import os
import time
import urllib.request

from websockets.sync.client import connect

VIEWER = os.environ.get("VIEWER_URL", "http://localhost:3001/")
CDP = f"http://127.0.0.1:{os.environ.get('CDP_PORT', '9222')}"


def page_target():
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{CDP}/json/list", timeout=2) as response:
                pages = [t for t in json.load(response) if t.get("type") == "page"]
            if pages:
                return pages[0]
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit(f"no headless Chrome answered on {CDP}")


def main():
    target = page_target()
    with connect(target["webSocketDebuggerUrl"], max_size=None, open_timeout=10) as cdp:
        ids = iter(range(1, 10_000))

        def call(method, **params):
            i = next(ids)
            cdp.send(json.dumps({"id": i, "method": method, "params": params}))
            while True:
                message = json.loads(cdp.recv(timeout=30))
                if message.get("id") == i:
                    return message.get("result", {})

        call("Page.navigate", url=VIEWER)
        text = ""
        for _ in range(60):
            time.sleep(1)
            result = call("Runtime.evaluate", expression="document.body ? document.body.innerText : ''",
                          returnByValue=True)
            text = result.get("result", {}).get("value") or ""
            if "Disconnected" not in text and "Connected" in text:
                print(f"{VIEWER}: the viewer is connected to the backend")
                return
    raise SystemExit(f"{VIEWER}: the viewer did not connect; page text: {text[:400]!r}")


if __name__ == "__main__":
    main()
