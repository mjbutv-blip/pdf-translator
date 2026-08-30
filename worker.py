from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import threading
import time
from uuid import uuid4

from config import OPENAI_API_KEY, PDF_WORKER_POLL_SECONDS, WORKER_HEARTBEAT_SECONDS
from translation_core import init_db
from translation_jobs import (
    claim_next_external_translation_job,
    heartbeat_translation_worker,
    register_translation_worker,
    run_claimed_translation_job,
    stop_translation_worker,
)


class ShutdownRequested:
    def __init__(self) -> None:
        self.value = False

    def request(self, signum, _frame) -> None:
        self.value = True
        _log_worker_event("worker_shutdown_requested", signal=signum)


def _make_worker_id() -> str:
    host = socket.gethostname() or "host"
    return f"pdf-worker-{host}-{os.getpid()}-{uuid4()}"


def _log_worker_event(event: str, **extra) -> None:
    payload = {
        "event": event,
        **{k: v for k, v in extra.items() if v is not None},
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _sleep_until_next_poll(seconds: int, shutdown: ShutdownRequested) -> None:
    end_at = time.monotonic() + seconds
    while not shutdown.value and time.monotonic() < end_at:
        time.sleep(min(0.5, end_at - time.monotonic()))


def _start_worker_heartbeat(worker_id: str, shutdown: ShutdownRequested) -> threading.Thread:
    def heartbeat_loop() -> None:
        while not shutdown.value:
            if not heartbeat_translation_worker(worker_id):
                register_translation_worker(worker_id)
            time.sleep(WORKER_HEARTBEAT_SECONDS)

    thread = threading.Thread(
        target=heartbeat_loop,
        name=f"translation-worker-heartbeat-{worker_id}",
        daemon=True,
    )
    thread.start()
    return thread


def run_worker(*, once: bool = False, worker_id: str | None = None, api_key: str | None = None) -> int:
    resolved_api_key = api_key or OPENAI_API_KEY
    resolved_worker_id = worker_id or _make_worker_id()
    shutdown = ShutdownRequested()
    signal.signal(signal.SIGINT, shutdown.request)
    signal.signal(signal.SIGTERM, shutdown.request)

    init_db()
    register_translation_worker(resolved_worker_id)
    _log_worker_event(
        "worker_registered",
        worker_id=resolved_worker_id,
        heartbeat_seconds=WORKER_HEARTBEAT_SECONDS,
    )
    heartbeat_thread = _start_worker_heartbeat(resolved_worker_id, shutdown)
    _log_worker_event(
        "worker_started",
        worker_id=resolved_worker_id,
        poll_seconds=PDF_WORKER_POLL_SECONDS,
        once=once,
    )

    if not resolved_api_key:
        _log_worker_event("worker_no_api_key", worker_id=resolved_worker_id)
        shutdown.value = True
        heartbeat_thread.join(timeout=1)
        stop_translation_worker(resolved_worker_id)
        _log_worker_event("worker_stopped", worker_id=resolved_worker_id)
        return 1
    try:
        while not shutdown.value:
            job = claim_next_external_translation_job(resolved_worker_id)
            if job:
                run_claimed_translation_job(job, resolved_api_key, resolved_worker_id)
                if once:
                    break
                continue

            _log_worker_event("worker_idle", worker_id=resolved_worker_id)
            if once:
                break
            _sleep_until_next_poll(PDF_WORKER_POLL_SECONDS, shutdown)
    finally:
        shutdown.value = True
        heartbeat_thread.join(timeout=1)
        stop_translation_worker(resolved_worker_id)
        _log_worker_event("worker_stopped", worker_id=resolved_worker_id)

    _log_worker_event("worker_shutdown", worker_id=resolved_worker_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PDF Project external PDF translation worker")
    parser.add_argument("--once", action="store_true", help="Poll once, run at most one job, then exit.")
    parser.add_argument("--worker-id", default="", help="Optional explicit worker id for diagnostics/tests.")
    args = parser.parse_args(argv)
    return run_worker(once=args.once, worker_id=args.worker_id or None)


if __name__ == "__main__":
    raise SystemExit(main())
