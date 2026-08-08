import pandas as pd
import json
from pathlib import Path

import csv


"""
Error analysis for the clembench transcripts.

Input: a root folder path: <root><model><game><experiment><instance.json>
Uses "interactions.json" and "scores.json" for each game instance (= episode)
Output: aborts.csv
        instance_counts.csv

aborts.csv contains names of all episodes which were aborted, and the reasons why,
accompanied by the last 5 events of the episode (events are flattened across
rounds first, because games log the abort reason at different depths: usually it's
the final event, but for adventuregame it's the penultimate, for privateshared 4 
events back).
The reasons for abort were manually collected from the games source code (master.py, 
log_to_self function). 

"""

root = Path(r"C:\Users\white\Desktop\MSc_thesis\results_baseline")
instance_counts = {}

#turns = interactions['turns']
#last = turns[-1]                #last round (starting from the GM message to P1)
#print(last[-1])                 #the final event that happened
#print(last[-2])                 #the penultimate event

CAUSES = ["turn_limit_reached", "invalid_format",
      "validation error" ,
      "invalid format", "error",
      "invalid content", "abort game", "aborted",
      "metadata", "turns_limit"]

def count_instances(root_dir):
  print("Instance counts --- for reference:")
  for model_dir in root_dir.iterdir():       
    if model_dir.is_dir():       
        #replace with "interactions.json" if needed
        n = len(list(model_dir.rglob("scores.json")))  
        instance_counts[model_dir.name] = n  
        
        print()
        print(model_dir.name + ": " + str(n))


# Uncomment this block to rewrite instance_counts.csv
  # with open("instance_counts.csv", "w", newline="", encoding="utf-8") as f:
  #    w = csv.writer(f)
  #    w.writerow(["model", "n_instances"])
  #    for model, n in instance_counts.items():
  #        w.writerow([model, n])


def collect_aborts(root_dir):
  rows = []

  for scores_file in root_dir.rglob("scores.json"):    
        instance_dir = scores_file.parent              
        interactions_file = instance_dir / "interactions.json"
        scores = json.loads(scores_file.read_text(encoding="utf-8")) 
        model, game, experiment, instance = scores_file.relative_to(root_dir).parts[:4]
        ep = scores['episode scores']
        if ep["Aborted"]:
          outcome = "Aborted"
          interactions= json.loads(interactions_file.read_text(encoding="utf-8"))
          
          turns = interactions["turns"]   #all turns
          flat = []
          for t in turns:
              for e in t:
                  flat.append(e)
          reason = None

          tail = []
          for e in reversed(flat[-5:]):
            a = e["action"]
            tail.append({"from": e["from"], "to": e["to"],
                    "type": a["type"], "content": str(a["content"])[:300]})
        
          for e in reversed(flat[-5:]):        # newest event gets written down first
              if e["action"]["type"] in CAUSES:
                  reason = e["action"]
                  detail = reason["content"]

                  rows.append({
                  "model": model,
                  "game": game,
                  "experiment": experiment,
                  "instance": instance,
                  "cause_type": reason["type"],
                  "detail": str(detail)[:200],
                  "n_rounds": len(turns),
                  "tail": json.dumps(tail, ensure_ascii=False),
                  })
                  break
          else:
            rows.append({"model": model, "game": game,
                  "experiment": experiment,
                  "instance": instance, "cause_type": "UNMATCHED",
                    "detail": "", "n_rounds": len(turns), "tail": json.dumps(tail, ensure_ascii=False)})
        elif ep["Success"]:
          outcome = "Success"
        else:
          outcome = "Lose"
  return rows
    
def main():
    count_instances(root)
    rows = collect_aborts(root)
    pd.DataFrame(rows).to_csv("aborts.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    main()          