#!/usr/bin/env python3
"""Start the visual workspace or the legacy GT demo."""
import argparse
import threading
from http.server import ThreadingHTTPServer
from legacy_gt.server import LiveApp, handler_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["visual", "gt"], default="visual", help="Default visual workspace or legacy GT UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--fast", action="store_true", help="Disable realtime pacing for automated tests")
    args = parser.parse_args()
    if not 0.1 <= args.radius <= 1:
        parser.error("radius must be between 0.1 and 1 meter")
    thread = None
    if args.mode == 'visual':
        from app.visual_web import VisualApp, handler_for_visual
        app = VisualApp(args.scene)
        handler = handler_for_visual(app, lambda: LiveApp(args.scene, args.radius, realtime=not args.fast))
    else:
        app = LiveApp(args.scene, args.radius, realtime=not args.fast)
        handler = handler_for(app)
        thread = threading.Thread(target=app.worker, name="habitat-owner", daemon=True)
        thread.start()
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Browser: http://{args.host}:{args.port} (waiting for scene)", flush=True)
    import signal
    def terminate_service(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate_service)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown.set()
        app.stop()
        server.server_close()
        if thread:
            thread.join(timeout=5)
        legacy = getattr(handler, 'legacy_apps', {}).get('app')
        if legacy:
            legacy.shutdown.set()
            legacy.stop()


if __name__ == "__main__":
    main()
