"""Real HTTP/GPU integration test for LangGraph tasks and mid-walk controls."""
import argparse
import json
import time
from urllib import request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument("--agent", action="store_true", help="Also test one natural-language destination")
    args = parser.parse_args()
    opener = request.build_opener(request.ProxyHandler({}))

    def call(path="/api/status", data=None):
        req = request.Request(args.url+path, data=None if data is None else json.dumps(data).encode(),
                              headers={} if data is None else {"Content-Type": "application/json"})
        with opener.open(req, timeout=10) as response:
            return json.load(response)

    def wait(predicate, timeout=240):
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            state = call()
            if state.get("error"):
                raise RuntimeError(state["error"])
            if predicate(state):
                return state
            time.sleep(0.2)
        raise TimeoutError("Live workflow condition timed out")

    def job(state, job_id):
        return next(j for j in state["jobs"] if j["id"] == job_id)

    def finished(job_id):
        state = wait(lambda s: job(s, job_id)["status"] not in {"queued", "running"}, timeout=600)
        return job(state, job_id)

    state = wait(lambda s: s["ready"])
    assert state["engine"] == "langgraph"
    first = next(o["id"] for o in state["targets"] if "chair_05_:0000" in o["id"])
    second = next(o["id"] for o in state["targets"] if "chair_04_:0000" in o["id"])
    task = call("/api/command", {"targets": [first, second], "return_to_start": True})
    wait(lambda s: s["phase"] == "walking")
    call("/api/pause", {})
    paused = wait(lambda s: s["phase"] == "paused")
    time.sleep(0.5)
    assert call()["position"] == paused["position"], "Robot moved while paused"
    call("/api/resume", {})
    result = finished(task["id"])["result"]
    assert result["status"] == "succeeded", result
    assert result["completed_steps"] == 3, result
    distance = sum((a-b)**2 for a,b in zip(result["start_position"], result["final_position"]))**0.5
    assert distance < 0.1, distance
    print(f"MULTI + RETURN + PAUSE/RESUME OK: return error {distance:.4f} m", flush=True)
    old = call("/api/command", {"target": first})
    wait(lambda s: s["phase"] == "walking")
    replacement = call("/api/command", {"target": second, "replace": True})
    old_job = finished(old["id"])
    new_job = finished(replacement["id"])
    assert old_job["status"] == "cancelled", old_job
    assert new_job["status"] == "succeeded", new_job
    assert old_job["result"]["final_position"] == new_job["result"]["start_position"]
    print("REPLACE FROM ACTUAL POSITION OK", flush=True)
    if args.agent:
        task = call("/api/command", {"instruction": "去沙发旁边"})
        print(f"REAL MODEL TASK {task['id']} submitted", flush=True)
        result = finished(task["id"])["result"]
        assert result["status"] == "succeeded", result
        assert result["completed_steps"] == 1, result
        assert result["final_error_m"] < 0.1, result
        print("REAL MODEL SINGLE-TARGET OK", flush=True)


if __name__ == "__main__":
    main()
