"""
In Reflexion, there are up to 3 attempts to replay. To evaluate the approach,
we take only the last episode (it can be either won or aborted/lost, if all
attempts were used). This is a short helper snippet to distill the last
episode of each instance.
"""
import json, shutil
from pathlib import Path

RAW_PATH = Path(r"C:\Users\white\Desktop\MSc_thesis\reflexion\playpen-records\reflexion-Qwen3.5-9B")
FINAL_PATH = Path(r"C:\Users\white\Desktop\MSc_thesis\reflexion\selected_episodes\reflexion-Qwen3.5-9B")

last = {}

for trace_path in RAW_PATH.glob("**/reflexion_trace.json"):
    trace = json.load(open(trace_path, encoding="utf-8"))
    folder = trace_path.parent
    k = (folder.parent.parent.name, folder.parent.name, trace["key"])
    if k not in last or trace["trial"] > last[k][0]:
          last[k] = (trace["trial"], folder)

for (game, exp, key), (trial, folder) in last.items():
      gid = int(key.split("/")[-1])
      dest = FINAL_PATH / game / exp / ("instance_%05d" % gid)
      dest.mkdir(parents=True, exist_ok=True)
      for name in ("interactions.json", "instance.json", "interactions_reflexion.json", "reflexion_trace.json",
                   "transcript_reflexion.html", "transcript_reflexion.tex"):
          shutil.copy(folder / name, dest / name)

for src in RAW_PATH.glob("**/experiment.json"):
      rel = src.relative_to(RAW_PATH)
      (FINAL_PATH / rel).parent.mkdir(parents=True, exist_ok=True)
      shutil.copy(src, FINAL_PATH / rel)