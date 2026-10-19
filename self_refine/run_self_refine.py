import os
import sys
import time

CLEMBENCH_HOME = r"/home/users/mmanina/project/mmanina/MSc_thesis/libs/playpen/clembench"
os.environ["CLEMBENCH_HOME"] = CLEMBENCH_HOME
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

os.chdir("/home/users/mmanina/project/mmanina/MSc_thesis/libs/playpen")

import json
from pathlib import Path

from clemcore.backends import load_model
from clemcore.clemgame import env, episode_results_folder_callbacks
from clemcore.clemgame.transcripts.builder import build_tex, build_transcript

sys.path.insert(0, "/home/users/mmanina/project/mmanina/MSc_thesis/self_refine")
from self_refine_agent import (SelfRefineAgent, RefineOnlyAgent, inject_traces, FEEDBACK_PROMPT, 
                               REFINEMENT_PROMPT, REFINE_ONLY_PROMPT)

RESULTS_ROOT = "playpen-records"
CONDITION = os.environ.get("SR_CONDITION", "selfrefine")
AGENT_CLASSES = {"selfrefine": SelfRefineAgent, "refineonly": RefineOnlyAgent}
AgentClass = AGENT_CLASSES[CONDITION]
TEMPERATURE = 0.0
MAX_TOKENS = 600

# ifeval is last on purpose: is overwrites the max_tokens.
# if anything was after that, it would've been run with 
# max_tokens = 1280.
GAMES = [
    "adventuregame", "codenames", "dond", "guesswhat", "imagegame",
    "matchit_ascii", "privateshared", "referencegame", "taboo",
    "textmapworld", "textmapworld_graphreasoning", "textmapworld_specificroom",
    "wordle", "wordle_withclue", "wordle_withcritic",
    "wordle_noexp", "wordle_withclue_noexp", "wordle_withcritic_noexp",
    "bbh", "cladder", "eqbench", "mmlu_pro",
    "ifeval",
]

EXPECTED = {
    "adventuregame": 40, "codenames": 130, "dond": 40, "guesswhat": 60,
    "imagegame": 40, "matchit_ascii": 40, "privateshared": 50,
    "referencegame": 90, "taboo": 60, "textmapworld": 50,
    "textmapworld_graphreasoning": 30, "textmapworld_specificroom": 30,
    "wordle": 30, "wordle_withclue": 30, "wordle_withcritic": 30,
    "wordle_noexp": 30, "wordle_withclue_noexp": 30,
    "wordle_withcritic_noexp": 30,
    "bbh": 104, "cladder": 100, "eqbench": 100, "mmlu_pro": 100, "ifeval": 100,
}


def build_agent_mapping(game_name, roles, player_0, player_1):
    if len(roles) == 1 or game_name == "privateshared":
        return {"player_0": player_0}
    return {"player_0": player_0, "player_1": player_1}


def already_done(run_dir, game_name):
    folder = Path(RESULTS_ROOT) / run_dir / game_name
    if not folder.is_dir():
        return False
    found = len(list(folder.glob("**/interactions.json")))
    want = EXPECTED.get(game_name)
    if want is not None and found == want:
        print("skipping %s: already has %d/%d episodes" % (game_name, found, want))
        return True
    print("WARNING: %s has %d episodes, expected %s - delete the folder to redo it"
          % (game_name, found, want))
    return True


def run_game(game_name, run_dir, player_0, player_1, shared_model, model_infos):
    callbacks = episode_results_folder_callbacks(
        run_dir=run_dir,
        result_dir_path=RESULTS_ROOT,
        player_model_infos=model_infos,
    )

    game_env = env(game_name, single_pass=True, callbacks=callbacks)
    results_folder = callbacks.callbacks[0].results_folder

    roles = game_env.unwrapped.game_benchmark.game_spec["roles"]
    agent_mapping = build_agent_mapping(game_name, roles, player_0, player_1)
    print("roles: %s   agent mapping: %s" % (roles, list(agent_mapping)))

    episodes = 0
    while True:
        reset_options = {}
        if game_name in ("ifeval", "privateshared"):
            reset_options["player_models"] = [shared_model]

        try:
            game_env.reset(options=reset_options)
        except StopIteration:
            break

        for agent in (player_0, player_1):
            agent.reset()

        episode_traces = []

        for step_idx, agent_id in enumerate(game_env.agent_iter()):
            context, reward, termination, truncation, info = game_env.last()

            if termination or truncation:
                response = None
            elif agent_id in agent_mapping:
                response = agent_mapping[agent_id](context)
            else:
                response = game_env.unwrapped.player_by_agent_id[agent_id](context)

            acting = agent_mapping.get(agent_id)
            if acting is not None and acting.trace:
                episode_traces.append({
                    "step":    step_idx,
                    "agent":   agent_id,
                    "context": context["content"],
                    "trace":   acting.trace,
                })
                acting.trace = []      
            if game_name == "privateshared":
                env_player = game_env.unwrapped.player_by_agent_id.get(agent_id)
                if env_player is not None and response is not None:
                    env_player._messages.append(
                        dict(role="user", content=context["content"]))
                    env_player._messages.append(
                        dict(role="assistant", content=response))

            game_env.step(response)

        episode_path = results_folder.to_instance_dir_path(
            game_env.unwrapped.game_master,
            game_env.unwrapped.game_instance)

        with open(episode_path / "agent_trace.json", "w", encoding="utf-8") as f:
            json.dump(episode_traces, f, indent=2, ensure_ascii=False)

        enriched = inject_traces(episode_path, episode_traces)
        (episode_path / "transcript_selfrefine.tex").write_text(
            build_tex(enriched), encoding="utf-8")
        (episode_path / "transcript_selfrefine.html").write_text(
            build_transcript(enriched), encoding="utf-8")

        episodes += 1
        print("  episode %d: %d traced steps" % (episodes, len(episode_traces)))

    return episodes


def write_run_config(run_dir, model_name, shared_model, dtype, games):
    """Record the settings and the exact prompts next to the results.

    experiment.json only gets the player_model_infos string when the ClemAgent
    setup is used instead of clem run; that's why we have to write into a separate
    file
    """
    folder = Path(RESULTS_ROOT) / run_dir
    folder.mkdir(parents=True, exist_ok=True)

    spec = {}
    model_spec = getattr(shared_model, "model_spec", None)
    for key in ("model_name", "backend", "huggingface_id", "parameters",
                "release_date", "context_size"):
        value = getattr(model_spec, key, None)
        if value is not None:
            spec[key] = value
    model_config = getattr(model_spec, "model_config", None)
    if isinstance(model_config, dict):
        spec["model_config"] = model_config

    config = {
        "condition": CONDITION,
        "agent": AgentClass.__name__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": model_name,
        "model_spec": spec,
        "gen_args": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        "torch_dtype": dtype,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "clembench_home": os.environ.get("CLEMBENCH_HOME"),
        "games": list(games),
        "feedback_prompt":   FEEDBACK_PROMPT if CONDITION == "selfrefine" else None,
        "refinement_prompt": REFINEMENT_PROMPT if CONDITION == "selfrefine" else REFINE_ONLY_PROMPT,
    }

    with open(folder / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("wrote     : %s" % (folder / "run_config.json"))


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-4B"
    games = sys.argv[2:] if len(sys.argv) > 2 else GAMES
    run_dir = "%s-%s" % (CONDITION, model_name)

    print("model     : %s" % model_name)
    print("run_dir   : %s/%s" % (RESULTS_ROOT, run_dir))
    print("gpu       : nvidia-smi %s" % os.environ["CUDA_VISIBLE_DEVICES"])
    print("games     : %d" % len(games))

    shared_model = load_model(model_name,
                              gen_args=dict(temperature=TEMPERATURE,
                                            max_tokens=MAX_TOKENS))
    dtype = str(getattr(getattr(shared_model, "model", None), "dtype", "unknown"))
    print("dtype     : %s" % dtype)
    model_infos = "%s | %s | temperature=%s max_tokens=%s" % (AgentClass.__name__, model_name, TEMPERATURE, MAX_TOKENS)

    write_run_config(run_dir, model_name, shared_model, dtype, games)

    player_0 = AgentClass("player_0", shared_model)
    player_1 = AgentClass("player_1", shared_model)


    timings = []
    started = time.time()
    timings_path = Path(RESULTS_ROOT) / run_dir / "run_timings.json"

    def save_timings():
        """Rewritten after every game, so a crash or a kill keeps what finished.
        clem run puts this in clembench.log; the playpen-agent path does not."""
        payload = {
            "model": model_name,
            "condition": CONDITION,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            "elapsed_min_so_far": round((time.time() - started) / 60, 1),
            "games": timings,
        }
        timings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(timings_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    for game_name in games:
        if already_done(run_dir, game_name):
            timings.append({"game": game_name, "status": "skipped"})
            save_timings()
            continue

        print("\n" + "=" * 70)
        print("%s  (expected %s episodes)" % (game_name, EXPECTED.get(game_name, "?")))
        print("=" * 70)

        t0 = time.time()
        entry = {"game": game_name,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))}
        try:
            n = run_game(game_name, run_dir, player_0, player_1, shared_model,
                         model_infos)
            minutes = (time.time() - t0) / 60
            entry.update(status="ok", episodes=n,
                         minutes=round(minutes, 1),
                         sec_per_episode=round(minutes * 60 / n, 1) if n else None)
            print("%s done: %d episodes in %.1f min" % (game_name, n, minutes))
        except Exception:
            import traceback
            traceback.print_exc()
            minutes = (time.time() - t0) / 60
            entry.update(status="failed", minutes=round(minutes, 1))
            print("%s FAILED after %.1f min - continuing with the next game"
                  % (game_name, minutes))

        timings.append(entry)
        save_timings()

    total_min = (time.time() - started) / 60
    save_timings()

    print("\n" + "=" * 70)
    print("finished in %.1f min (%.1f h)" % (total_min, total_min / 60))
    print("timings written to %s" % timings_path)

    by_game = {}
    for entry in timings:
        by_game[entry["game"]] = entry

    for game_name in games:
        want = EXPECTED.get(game_name, "?")
        entry = by_game.get(game_name, {})
        status = entry.get("status", "not run")

        if status == "skipped":
            line = "skipped, already complete"
        elif status == "failed":
            line = "FAILED after %.1f min" % entry.get("minutes", 0)
        elif status == "ok":
            got = entry.get("episodes")
            line = "%s / %s" % (got, want)
            if got != want:
                line += "   <-- check"
        else:
            line = "not run"

        print("  %-30s %s" % (game_name, line))


if __name__ == "__main__":
    main()
