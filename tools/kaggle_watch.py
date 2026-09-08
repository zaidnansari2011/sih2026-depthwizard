"""Poll a Kaggle kernel and print one line whenever its state changes.

Kaggle gives no push notification for a kernel run, and a GPU run here is 4-5 hours, so
this polls. It emits on every state CHANGE and on every terminal state -- including the
failure ones. A watcher that printed only "COMPLETE" would stay silent through an error
and look identical to still running.

    python tools/kaggle_watch.py zaidansari8974/depthwizard-run06-v1-large
"""
from __future__ import annotations

import json
import os
import sys
import time

TERMINAL = {"COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}
INTERVAL = 300          # remote API; 5 minutes is plenty for a multi-hour run


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else \
        "zaidansari8974/depthwizard-run06-v1-large"

    # Bearer, not Basic. The stored credential is a KGAT token, and kagglesdk only
    # switches to Bearer when this variable is present; with kaggle.json alone it sends
    # HTTP Basic, which Kaggle accepts for some read endpoints and rejects for the rest.
    if not os.getenv("KAGGLE_API_TOKEN"):
        tok = os.path.expanduser("~/.kaggle/access_token")
        if os.path.exists(tok):
            with open(tok) as fh:
                os.environ["KAGGLE_API_TOKEN"] = fh.read().strip()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    last = None
    t0 = time.time()
    while True:
        try:
            s = api.kernels_status(ref)
            # kernels_status returns a JSON STRING, not a dict, so isinstance(s, dict)
            # is False and str(s) yields the whole blob. Left unparsed, `state` never
            # matches TERMINAL, the watch never exits, and an ERROR looks exactly like
            # still-running -- the silent-failure mode this file exists to avoid.
            if hasattr(s, "to_dict"):
                s = s.to_dict()                       # ApiGetKernelSessionStatusResponse
            elif isinstance(s, (str, bytes)):
                s = json.loads(s)
            raw = s.get("status")
            state = str(getattr(raw, "name", raw) or "UNKNOWN").upper()
            msg = s.get("failureMessage") or s.get("failure_message") or ""
        except Exception as exc:                      # a blip must not kill the watch
            state, msg = "POLL_FAILED", f"{type(exc).__name__}: {str(exc)[:120]}"

        if state != last:
            mins = (time.time() - t0) / 60
            line = f"kaggle {ref}: {state} at {mins:.0f} min"
            if msg:
                line += f" -- {msg}"
            print(line, flush=True)
            last = state

        if state in TERMINAL:
            return 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
