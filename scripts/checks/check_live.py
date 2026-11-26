"""Exercise the running browser service using two real offline navigation turns."""
import argparse
import io
import json
import time
from urllib import request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--check-stop", action="store_true")
    args = parser.parse_args()
    opener = request.build_opener(request.ProxyHandler({}))

    def get(path):
        with opener.open(args.url + path, timeout=10) as response:
            return response.read()

    def post(data, path="/api/command"):
        req = request.Request(args.url + path, data=json.dumps(data).encode(),
                              headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=10) as response:
            return json.load(response)

    deadline = time.monotonic() + 180
    while True:
        status = json.loads(get("/api/status"))
        if status["error"]:
            raise RuntimeError(status["error"])
        if status["ready"]:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Scene startup timed out")
        time.sleep(0.5)
    assert b"/api/frame.jpg" in get("/")
    from PIL import Image
    Image.open(io.BytesIO(get("/api/frame.jpg"))).verify()
    with opener.open(args.url + "/stream.mjpg", timeout=10) as response:
        assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        assert response.read(128).startswith(b"--frame\r\n")
    first_frame = status["frame_id"]
    chairs = [o for o in status["targets"] if o["category"] == "chair"]
    first = next(o for o in chairs if "chair_04_:0000" in o["id"])
    second = next(o for o in chairs if "chair_05_:0000" in o["id"])
    ids = [post({"target": first["id"]})["id"], post({"target": second["id"]})["id"]]
    print("Queued consecutive tasks:", ids, flush=True)
    deadline = time.monotonic() + 180
    while True:
        status = json.loads(get("/api/status"))
        jobs = {j["id"]: j for j in status["jobs"]}
        if all(jobs[i]["status"] not in {"queued", "running"} for i in ids):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Navigation tasks timed out")
        time.sleep(0.5)
    results = [jobs[i]["result"] for i in ids]
    assert all(r["status"] == "succeeded" for r in results), results
    assert results[0]["final_position"] == results[1]["start_position"], results
    assert results[0]["start_position"] != results[1]["start_position"]
    assert status["frame_id"] > first_frame
    assert len(status["messages"]) >= 4
    if args.check_stop:
        stopped_ids = [post({"target": first["id"]})["id"], post({"target": second["id"]})["id"]]
        deadline = time.monotonic() + 30
        while json.loads(get("/api/status"))["phase"] != "walking":
            if time.monotonic() > deadline:
                raise TimeoutError("Task did not begin walking")
            time.sleep(0.1)
        post({}, "/api/stop")
        while True:
            state = json.loads(get("/api/status"))
            stopped = {j["id"]: j for j in state["jobs"]}
            if all(stopped[i]["status"] == "cancelled" for i in stopped_ids):
                assert state["queued"] == 0
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Stop did not cancel current and queued tasks")
            time.sleep(0.1)
        print("STOP + QUEUE CLEAR: OK", flush=True)
    print(json.dumps({"http": "OK", "jpeg": "OK", "mjpeg": "OK", "continuous_pose": "OK",
                      "tasks": [{"id": i, "status": jobs[i]["status"],
                                 "error_m": jobs[i]["result"]["final_error_m"]} for i in ids]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
