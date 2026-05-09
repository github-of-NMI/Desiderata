from huggingface_hub import scan_cache_dir
from tqdm import tqdm
import ast
import re
import os
import hashlib
import json
import sys
import subprocess
import requests
import time
from datetime import date

def extract_last_boxed(text):
    matches = re.findall(r'\\boxed\{(.*)\}', text)
    if matches:
        return matches[-1]
    return None

def read_state():
    repeats = 20

    models_path = "models.txt"
    models = set()
    with open(models_path, "r") as f:
        for m in f.read().splitlines():
            models.add(m)

    noise_path = "closedAI_bench/noise.txt"
    noises = set()
    with open(noise_path, "r") as f:
        for m in f.read().splitlines():
            noises.add(m)

    qna_path = "closedAI_bench/qnas.txt"
    qnas = set()
    with open(qna_path, "r") as f:
        for l in f.read().splitlines():
            tup = ast.literal_eval(l)
            assert type(tup) is tuple
            qnas.add(tup)

    expected_len = len(qnas) * len(noises)
    result_path = "results/"
    completed_models = set()
    for file in [file for file in os.listdir(result_path) if os.path.isfile(os.path.join(result_path, file))]:
        with open(os.path.join(result_path, file), "r") as f:
            data = json.load(f)

            completed_repeats = data.get("repeats", 0)
            if completed_repeats < repeats:
                continue

            if len([key for key in list(data.keys()) if re.fullmatch(r"^[a-fA-F0-9]{64}$", key)]) != expected_len:
                continue
            
            completed_models.add(data["model"])#TODO: if there is a discrepancy between the saved thing and the expected thing, only compute the diff.


    models_to_evaluate = models - completed_models
    return models_to_evaluate, qnas, noises, repeats

def start_proc():
    server_cmd = [
        "llama-server",
        "-hf", model,
        "-c", "0",
        "--port", "5000"
    ]
    process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    for _ in range(60):  # Try for 60 seconds
        try:
            # Check the health endpoint
            response = requests.get("http://localhost:5000/health")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        process.terminate()
        raise TimeoutError("llama-server took too long to start.")
    return process

models, qnas, noises, repeats= read_state()
print(models)
print(f"models: {len(models)}")
print(f"qnas: {len(qnas)}")
print(f"noises: {len(noises)}")
print(f"repeats: {repeats}")
print(f"total: {len(models) * len(qnas) * len(noises) * repeats}")
ans_in_boxed = "\nPut the final value in a markdown \\boxed{}.\n"
for model in models:
    print(model)
    stats = dict()
    stats["date"]    = f"{date.today()}"
    stats["model"]   = model
    stats["repeats"] = repeats
    

    process = start_proc()
    

    for fullq, correct_ans in tqdm([(noise + q + ans_in_boxed, correct_ans,) for noise in noises for q, correct_ans in qnas], smoothing=0):
            qhash = hashlib.sha256(fullq.encode()).hexdigest()
            stats[qhash]                = dict()
            stats[qhash]["correct"]     = 0
            stats[qhash]["incorrect"]   = 0
            stats[qhash]["None_answer"] = 0

            unique_ans = set()
            for _ in tqdm(range(repeats)):
                payload = {#TODO: thinking is disabled, limited compute
                    "prompt": fullq,
                    "n_predict": 1024,
                    "stream": False  # Set to True if you want to process tokens one by one
                }
                response = requests.post("http://localhost:5000/completion", json=payload)
                response.raise_for_status()
                data = response.json()
                response = data["content"]

                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                ans = extract_last_boxed(response)
                unique_ans.add(ans)
                if ans is None:
                    stats[qhash]["None_answer"] += 1

                if correct_ans == "":#for questions that do not have a correct answer
                    break

                if ans == correct_ans:
                    stats[qhash]["correct"] += 1
                else:
                    stats[qhash]["incorrect"] += 1
            
            stats[qhash]["unique_answers"] = len(unique_ans)
    
    process.terminate()
    process.wait(timeout=5)
    process.kill()

    safe_model_name = model.replace("/", "_")
    with open(f"results/{safe_model_name}.json", "w+") as f:
        json.dump(stats, f, sort_keys=True, indent=4)

    cache_info = scan_cache_dir()
    repo_to_delete = next((r for r in cache_info.repos if r.repo_id == model), None)
    if repo_to_delete:
        for revhash in [revision.commit_hash for revision in repo_to_delete.revisions]:
            to_delete = cache_info.delete_revisions(revhash)
            to_delete.execute()
        print(f"deleted {model}")
