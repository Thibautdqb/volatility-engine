"""Where alerts go: terminal, desktop, phone (ntfy), Discord or Slack webhooks, a file.

Targets are given as strings, e.g. on the command line::

    terminal                     print the alert (default)
    bell                         ring the terminal bell
    desktop                      desktop notification (notify-send on Linux, osascript on macOS)
    ntfy:<topic>                 push to the ntfy app on a phone, via https://ntfy.sh/<topic>
    ntfy:<https://server/topic>  same, on a self-hosted ntfy server
    webhook:<url>                POST {"text": ..., "content": ...}: Slack and Discord webhooks
    file:<path>                  append the alert as one JSON object per line

Sending never blocks the monitor: network targets run in a background thread, with a
timeout and two retries, and a failure is logged rather than raised.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable

import httpx

log = logging.getLogger(__name__)
NTFY_SERVER = "https://ntfy.sh"


class Notifier:
    """Sends each alert to every configured target."""

    def __init__(self, targets: Iterable[str] = ("terminal",), timeout: float = 10.0):
        self.targets = list(targets)
        self.timeout = timeout
        self._senders: list[Callable[[dict, str, str], None]] = [
            self._parse(t) for t in self.targets
        ]

    def _parse(self, target: str):
        kind, _, arg = target.partition(":")
        if kind == "terminal":
            return lambda record, title, body: print(f"{title}\n  {body}", flush=True)
        if kind == "bell":
            return lambda record, title, body: (sys.stdout.write("\a"), sys.stdout.flush())
        if kind == "desktop":
            return self._desktop
        if kind == "ntfy" and arg:
            url = arg if arg.startswith("http") else f"{NTFY_SERVER}/{arg}"
            return lambda record, title, body: self._post_later(
                url, content=body.encode(), headers={"Title": title, "Priority": "high"}
            )
        if kind == "webhook" and arg:
            return lambda record, title, body: self._post_later(
                arg, json={"text": f"*{title}*\n{body}", "content": f"**{title}**\n{body}"}
            )
        if kind == "file" and arg:

            def write(record, title, body, path=arg):
                with open(path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            return write
        raise ValueError(f"unknown notification target {target!r}")

    def send(self, record: dict, title: str, body: str) -> None:
        for sender in self._senders:
            try:
                sender(record, title, body)
            except Exception:
                log.exception("notification failed")

    # -- transports -----------------------------------------------------------

    def _post_later(self, url: str, **kwargs) -> None:
        threading.Thread(target=self._post, args=(url,), kwargs=kwargs, daemon=True).start()

    def _post(self, url: str, **kwargs) -> None:
        for attempt in range(3):
            try:
                resp = httpx.post(url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return
            except httpx.HTTPError as exc:
                log.warning("notification to %s failed (%s)", url, exc)
                time.sleep(2**attempt)

    @staticmethod
    def _desktop(record: dict, title: str, body: str) -> None:
        if platform.system() == "Darwin":
            script = f"display notification {json.dumps(body)} with title {json.dumps(title)}"
            subprocess.run(["osascript", "-e", script], check=False, timeout=5)
        elif shutil.which("notify-send"):
            subprocess.run(["notify-send", "-u", "critical", title, body], check=False, timeout=5)
        else:
            log.warning("no desktop notifier found (notify-send or osascript)")
