import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from volengine.cli import main
from volengine.notify import Notifier


@pytest.fixture
def server():
    """A local HTTP server that records what it receives."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, dict(self.headers), body))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", received
    httpd.shutdown()


def wait_for(received, n=1, timeout=5.0):
    end = time.time() + timeout
    while len(received) < n and time.time() < end:
        time.sleep(0.02)
    return received


def test_webhook_and_ntfy_payloads(server):
    url, received = server
    n = Notifier([f"webhook:{url}/hook", f"ntfy:{url}/topic"])
    n.send({"x": 1}, "BTC: stress 99.5/100", "next 5 min: ...")
    got = {path: (headers, body) for path, headers, body in wait_for(received, 2)}
    hook = json.loads(got["/hook"][1])
    assert "BTC: stress 99.5/100" in hook["text"] and "BTC: stress 99.5/100" in hook["content"]
    headers, body = got["/topic"]
    assert headers["Title"] == "BTC: stress 99.5/100" and body == b"next 5 min: ..."


def test_file_target_and_bad_targets(tmp_path, capsys):
    path = tmp_path / "alerts.jsonl"
    Notifier(["terminal", f"file:{path}"]).send({"stress": 99.5}, "title", "body")
    assert json.loads(path.read_text()) == {"stress": 99.5}
    assert "title" in capsys.readouterr().out
    with pytest.raises(ValueError):
        Notifier(["pager"])
    with pytest.raises(ValueError):
        Notifier(["ntfy:"])


def test_a_failing_target_does_not_stop_the_others(tmp_path):
    path = tmp_path / "alerts.jsonl"
    n = Notifier([f"file:{tmp_path}/missing/dir/x.jsonl", f"file:{path}"])
    n.send({"ok": True}, "t", "b")
    assert path.exists()


def test_watch_test_sends_a_test_alert(tmp_path, capsys):
    path = tmp_path / "alerts.jsonl"
    main(["watch", "--test", "--notify", f"file:{path}"])
    assert json.loads(path.read_text()) == {"test": True}
    assert "test alert sent" in capsys.readouterr().out
