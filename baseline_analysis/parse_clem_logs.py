import re
from pathlib import Path

logs_path = Path(r"C:\Users\white\Desktop\MSc_thesis\baseline_runs")
MODELS = ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "Llama-3.1-8B-Instruct"]
RUN_RE = re.compile(r"Running (\S+) took: (\d+):(\d\d):(\d\d(?:\.\d+)?)")
all_total = 0

for model in MODELS:
    total = 0
    log_file = logs_path / model / "clem_baseline.log"
    if log_file.exists():
        text = log_file.read_text(encoding="utf-8")
        for line in text.splitlines():
              match = RUN_RE.search(line)
              if match:
                if match.group(1) != "privateshared":    # was exluded from the analysis, late encountered clembench peculiarity
                    total += (int(match.group(2)) * 3600  + int(match.group(3)) * 60 + float(match.group(4)))
    all_total += total

print("Total for" + str(logs_path) + "in hrs:")
print(all_total / 3600)