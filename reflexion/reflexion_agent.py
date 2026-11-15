from playpen.agents import ClemAgent, ClemObservation
from clemcore.clemgame.master import Outcome
from pathlib import Path
import json

PROMPT_DIR = Path("/home/users/mmanina/project/mmanina/MSc_thesis/reflexion")
REFLEXION_PROMPT = (PROMPT_DIR / "reflexion_prompt.txt").read_text(encoding="utf-8")
FEW_SHOT_EXAMPLES = (PROMPT_DIR / "few_shot_examples.txt").read_text(encoding="utf-8")
STATUS = {Outcome.FAILURE: "LOSE —— every message you sent was an acceptable move, but you failed to achieve the target state of the game. Your plan must change the strategy, not the formatting.", 
Outcome.ABORTED: "ABORT —— the Game Master rejected one of your messages because it violated the game rules, for example, your move did not follow the specified format or you did a thing you were specifically asked not to do. Your plan must focus on how to produce a well-formed and legal response."}

def as_transcript(history):
    """Render self.history as a readable transcript for the
    reflect() calls."""
    lines = []
    for message in history:
        if message["role"] == "user":
            who = "Game Master"
        else:
            who = "You"
        lines.append("%s: %s" % (who, message["content"]))
    return "\n\n".join(lines)


def _first_gm_event(interactions, player_name):
      """Find the first GM->player event in the episode (to inject reflections
      from last attempt)"""
      for turn in interactions["turns"]:
          for event in turn:
              if event["from"] == "GM" and event["to"] == player_name:
                  return event
      return None


def inject_reflexion(episode_path, to_json):

    interactions = json.load(open(episode_path / "interactions.json", encoding="utf-8"))
    if not interactions["turns"]:
        interactions["turns"] = [[]]

    players = []
    for name in interactions["players"]:
        if name != "GM":
            players.append(name)

    for i, name in enumerate(players):
        entries = to_json["agents"].get("player_%d" % i, [])

        for entry in entries:
            if entry["type"] == "memory shown":
                event = _first_gm_event(interactions, name)
                if event is None:
                    print("WARNING: no GM -> %s event to attach the memory to" % name)
                else:
                    event["action"]["content"] = (
                        entry["content"] + "\n\n" + event["action"]["content"])
            else:
                interactions["turns"][-1].append({
                    "from": name, "to": name,
                    "action": {"type": entry["type"],
                                "content": "Post-episode Reflexion (not sent to GM):\n%s" % entry["content"]}})

    with open(episode_path / "interactions_reflexion.json", "w", encoding="utf-8") as f:
        json.dump(interactions, f, indent=2, ensure_ascii=False)

    return interactions

class ReflexionAgent(ClemAgent):
    """
    An enhanced ClemAgent which iteratively replays any failed episode up to three times,
    reflecting on failures. The Reflector gets the outcome label for the failed episode
    (ABORT or LOSE) and tries to plan a better trajectory for the next attempt. The resulting
    trajectory gets appended to player's memory through being shown with the first
    Game Master prompt at the very beginning of the episode.
    """
    # act() method is the same as in ClemAgent

    def __init__(self, name, model):
        super().__init__()
        self.name = name
        self.model = model
        self.memory = []       # (this is not the Player's memory from the default clembench, this is
                               # the mem from the Reflexion algorithm)
        self.omega = 3     # num of stored memories (to adhere to max_context)
        self.trace = []    # helper: to log things


    def clear_memory(self):
        """
        Helper: when one instance is played successfully, the memory is wiped
        (in the default Reflexion setup, the memory is not passed on further across episodes)
        """
        self.memory = []

    def reset(self):
        super().reset()
        self.trace = []
   
    def render_memory(self, memory):
        """
        Helper: to present the last 3 memories to the player (mediated by the GM)
        """
        lines = ["You have attempted this same task before and did not succeed. These are the plans you wrote after those attempts."]
        for i, plan in enumerate(memory):
          lines.append("Trial #%d: %s" % (i+1, plan) + "\n")
        return "\n".join(lines)

    def trim_trajectory(self):
        """
        Helper: the opposite of render_memory (erase the memories from the transcript
        so that reflect() gets a structured mem at the end, not twice --- 
        mem is a part of GM's first message (see the call function), I wanted it to
        be erased)
        """
        msgs = []
        for message in self.history:
            msgs.append(dict(message))
        for entry in self.trace:
            if entry["type"] == "memory shown" and msgs:
                prefix = entry["content"] + "\n\n"
                if msgs[0]["content"].startswith(prefix):
                    msgs[0]["content"] =  msgs[0]["content"][len(prefix):]
        return as_transcript(msgs)


    def __call__(self, observation, *, memorize=True):
        # append reflexions from memory once in the GM prompt (first ever turn only)
        if not self.observations and self.memory:
              shown = self.render_memory(self.memory[-self.omega:])
              observation = dict(observation)
              observation["content"] = shown + "\n\n" + observation["content"]
              self.trace.append({"type": "memory shown", "content": shown})
        return self.act(self.observe(observation, memorize=memorize))


    def act(self, last: ClemObservation) -> str:
        # Generate initial trajectory using πθ

        _, _, response_text = self.model.generate_response(self.history) 
        # Observe our own response in the interaction history
        self.observe(dict(role="assistant", content=response_text))

        return response_text

    def reflect(self, reflexion_prompt, few_shot_examples, outcome):

        mem = self.memory[-self.omega:]
        replaced = reflexion_prompt.format(history=self.trim_trajectory(),
                                                    FEW_SHOT_EXAMPLES=few_shot_examples,
                                                    status=STATUS.get(outcome, str(outcome)))

        if mem:
            replaced += "\n\n Plans from past attempts:\n "             
            for id, m in enumerate(mem):
                replaced += "Trial #%d: %s\n" % (id+1, m)
        
        replaced += "\n\nNew plan:"
        reflexion_msgs = [{"role": "user", "content": replaced}]
        _, _, new_plan = self.model.generate_response(reflexion_msgs)

        # we don't observe reflexions as part of the Player's experience!
        # (but we append it to the "separate" memory)
        self.memory.append(new_plan)
        self.trace.append({"type": "reflection", "content": new_plan})
        return new_plan
