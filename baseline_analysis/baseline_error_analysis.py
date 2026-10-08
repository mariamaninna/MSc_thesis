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

interactions file structure for every clemgame episode:
{ "meta"  : {...},
  "player_models" : {...},
  "players"       : { ... },
  "turns" : a list of lists
            [ each list is one round, each dict
              inside a list is one event; an event
              always has the following fields: 
                            "from": "..."
                            "to": "...",
                            "timestamp": "...",
                            "action": { "type": "...", "content": "..." }
            ]
}
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from clemcore.clemeval import save_clem_table


root = Path(r"C:\Users\white\Desktop\MSc_thesis\baseline_runs")
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
STATIC = ["bbh", "cladder", "ifeval", "mmlu_pro", "eqbench"]
GENERIC = {"abort game", "game_result = abort", "continue", "end game"}


# === GATHERING GAME OUTCOMES ===

def subsample_static(root_dir):
    """
    The static benchmark scores need to be analyzed separately from the game scores.
    """
    raw_table = pd.read_csv(Path(root_dir) / "raw.csv", index_col=0)
    is_static = raw_table["game"].isin(STATIC)
    static_rows = raw_table[is_static == True]
    game_rows = raw_table[is_static == False]
    save_clem_table(static_rows, "baseline_analysis/static")
    save_clem_table(game_rows, "baseline_analysis/games")

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
#   with open(r"C:\Users\white\Desktop\MSc_thesis\baseline_analysis\instance_counts.csv", "w", newline="", encoding="utf-8") as f:
#      w = csv.writer(f)
#      w.writerow(["model", "n_instances"])
#      for model, n in instance_counts.items():
#          w.writerow([model, n])


def collect_aborts(root_dir):
  rows = []       #for the aborts.csv df
  outcomes = []   #for general outcome count

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

          gm_reason = ""
          for e in flat[-5:]:                                  # oldest -> newest
            a = e["action"]
            content = str(a["content"]).strip()
            if not content or content.lower() in GENERIC:
                continue                                      # skip the content-free markers
            if a["type"] == "metadata" and not content.startswith("Error:"):
                continue                                      # skip wordle's turn-stats lines
            if a["type"] in CAUSES:
                gm_reason = content[:300]        

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
                  "gm_reason": gm_reason,
                  "detail": str(detail)[:200],
                  "n_rounds": len(turns),
                  "tail": json.dumps(tail, ensure_ascii=False),
                  "group": "static" if game in STATIC else "games"
                  })
                  break
          else:
            rows.append({"model": model, "game": game,
                  "experiment": experiment,
                  "instance": instance, "cause_type": "UNMATCHED",
                    "detail": "", "n_rounds": len(turns), 
                    "group": "static" if game in STATIC else "games",
                    "tail": json.dumps(tail, ensure_ascii=False)})
        elif ep["Success"]:
          outcome = "Success"
        else:
          outcome = "Lose"
        outcomes.append({"model": model, "game": game,      
                           "experiment": experiment,
                           "instance": instance,
                           "group": "static" if game in STATIC else "games",
                           "outcome": outcome})
  return rows, outcomes

# === PLOTTING ===

def plot_outcomes(df, title, out_stem):
        OUTCOME_COLORS = {"Success": "#60d169", "Lose": "#74787f", "Aborted": "#c14f4d"}
        ORDER = ["Success", "Lose", "Aborted"]

        ct = pd.crosstab(df.model, df.outcome).reindex(columns=ORDER, fill_value=0)
        pct = ct.div(ct.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(8, 2.6))
        left = pd.Series(0.0, index=pct.index)
        for oc in ORDER:
            ax.barh(pct.index, pct[oc], left=left, color=OUTCOME_COLORS[oc],
                    label=oc, height=0.62, edgecolor="#fcfcfb", linewidth=2)
            for m in pct.index:                                  
                if pct.loc[m, oc] > 6:
                    ax.text(left[m] + pct.loc[m, oc] / 2, m, f"{pct.loc[m, oc]:.0f}%",
                            ha="center", va="center", color="black", fontsize=9)
            left += pct[oc]

        ax.set_xlim(0, 100)
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.45))
        ax.set_title(title, loc="left", pad=12)
        fig.tight_layout()
        fig.savefig(out_stem + ".png", bbox_inches="tight", dpi = 300)
        fig.savefig(out_stem + ".pdf", bbox_inches="tight")

def canonical_reason(s):
    """
    For codenames, which writes an abort reason as dictionary (in all other games, it's a string),
    and wordle, which always puts a prefix "Error:", we need some additional parsing.
    """
    s = str(s).strip()
    if not s or s == "nan":
        return "(not logged)"
    m = re.search(r"'type':\s*'([^']*)'", s)        
    if m:
        return m.group(1)
    if s.startswith("Error:"):                       
        return s[len("Error:"):].strip()
    return s


def plot_causes_by_game_model(df_aborts, df_outcomes, out_path):

        FORMAT = {"The response should contain the 'guess:' keyword exactly once.",
            "The guess should be a single word and should only contain letters.",
            "Invalid grid format","Abort: invalid format in probing.",
            'proposal does not start with "proposal:"',"rambling error",
            "proposal does not include number/name pairs", "clue contains spaces",
            "target appears more than once in utterance","guess appears more than once in utterance"     }
        CAT_ORDER  = ["format", "rule"]
        CAT_COLORS = {"format": "#2a78d6", "rule": "#eb6834"}
        GAME_GROUPS = {
      "word games":        ["taboo", "wordle", "wordle_withclue", "wordle_withcritic", "wordle_noexp", "wordle_withclue_noexp", "wordle_withcritic_noexp"],
      "reference":         ["referencegame", "imagegame", "matchit_ascii"],
      "navigation":        ["adventuregame", "textmapworld", "textmapworld_graphreasoning",
                            "textmapworld_specificroom"],
      "dialogue / social": ["codenames", "dond", "guesswhat", "privateshared"],
  }
        d = df_aborts.copy()

        d["reason"] = [canonical_reason(x) for x in d.gm_reason]
        # this is very hardcoded, but I see no other way to do this because loggers write
        # the reasons why the game aborted a bit differently
        d["category"] = ["format" if r in FORMAT else None if r == "(not logged)" else "rule" for r in d["reason"]]

        all_games = sorted(df_outcomes.game.unique())
        pooled = pd.crosstab(d.game, d.category).reindex(all_games, fill_value=0)
        order = [g for grp in GAME_GROUPS.values() for g in grp][::-1]
        models = sorted(d.model.unique())
        xmax   = pd.crosstab([d.model, d.game], d.category).sum(axis=1).max() * 1.15

        fig, axes = plt.subplots(1, len(models), figsize=(10, 5), sharey=True)
        for ax, m in zip(axes, models):
            ct = (pd.crosstab(d[d.model == m].game, d[d.model == m].category)
                    .reindex(index=order, columns=CAT_ORDER, fill_value=0))
            left = np.zeros(len(ct))
            for c in CAT_ORDER:
                ax.barh(ct.index, ct[c], left=left, color=CAT_COLORS[c], label=c,
                        height=0.7, edgecolor="#fcfcfb", linewidth=1.2)
                left += ct[c].values
            ax.set_xlim(0, xmax)
            ax.set_title(m, loc="left", fontsize=9)
            ax.set_xlabel("aborted episodes")
            ax.tick_params(labelsize=8)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=3,
                loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("", x=0.02, ha="left")                              #abort causes by game and model
        fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")




def main():

    OUT = Path(r"C:\Users\white\Desktop\MSc_thesis\baseline_analysis")
    IMG = OUT / "images"
    IMG.mkdir(parents=True, exist_ok=True)

    subsample_static(root)
    count_instances(root)
    rows, outcomes = collect_aborts(root)

    df = pd.DataFrame(outcomes)
    df_aborts = pd.DataFrame(rows)
    df.to_csv(OUT / "outcomes.csv", index=False, encoding="utf-8")
    df_aborts.to_csv(OUT / "aborts.csv", index=False, encoding="utf-8")

    games_df = df[df.group == "games"]
    static_df = df[df.group == "static"]
    n_games = games_df.groupby("model").size().iloc[0]      # Caution: need to verify if all instance counts per model are equal!
    n_static  = static_df.groupby("model").size().iloc[0]   # (e.g. by rewriting instance_counts.csv)

    print("Total game episodes: ", n_games)
    print("Total static episodes: ", n_static)

    plt.rcParams.update({
      "font.family": "serif",
      "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
      "font.size": 10,
      "mathtext.fontset": "custom",          
      "mathtext.rm": "Palatino Linotype",
      "mathtext.it": "Palatino Linotype:italic",
                        })
    plot_outcomes(df[df.group == "games"],
                    "",
                    str(IMG / "figure_general_outcomes"))

    plot_outcomes(df[df.group == "static"],
                    "",
                    str(IMG / "figure_general_outcomes_static"))

    plot_causes_by_game_model(df_aborts[df_aborts.group == "games"],
                                df[df.group == "games"],
                                str(IMG / "fig_causes_by_game_model.pdf"))


        

if __name__ == "__main__":
    main()          