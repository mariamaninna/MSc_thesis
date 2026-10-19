
import json
import re
from pathlib import Path

from playpen.agents import ClemAgent, ClemObservation

PROMPT_DIR = Path("/home/users/mmanina/project/mmanina/MSc_thesis/self_refine")

FEEDBACK_PROMPT = (PROMPT_DIR / "feedback_prompt.txt").read_text(encoding="utf-8")
REFINEMENT_PROMPT = (PROMPT_DIR / "refinement_prompt.txt").read_text(encoding="utf-8")
REFINE_ONLY_PROMPT = (PROMPT_DIR / "refinement_prompt_no_feedback.txt").read_text(encoding="utf-8")


def as_transcript(history):
    """Render self.history as a readable transcript for the
    feedback() and refine() calls."""
    lines = []
    for message in history:
        if message["role"] == "user":
            who = "Game Master"
        else:
            who = "You"
        lines.append("%s: %s" % (who, message["content"]))
    return "\n\n".join(lines)


class SelfRefineAgent(ClemAgent):
    """
    An enhanced ClemAgent which emits the game responses according to the Self-Refine
    framework:
    First, a so-called draft is generated (in the same way that an ordinary Player
    response is generated). Then, the same model is queried to give feedback on this
    draft. Finally, based on the feedback, the answer is refined one final time and
    passed on to the Game Master.

    The draft and the feedback never enter self.history: the Game Master only ever
    sees the refined answer, so the history stays a faithful record of the dialogue.
    The intermediate steps go to self.trace, which the run loop collects per step.
    """

    def __init__(self, name, model):
        super().__init__()
        self.name = name
        self.model = model
        self.trace = []

    def reset(self):
        super().reset()
        self.trace = []

    def feedback(self, feedback_prompt, initial_draft) -> str:
        """
        Using the same player model, generate feedback on the initial answer,
        evaluating it, to be further used for self-refinement
        """
        replaced = feedback_prompt.format(transcript=as_transcript(self.history),
                                          draft=initial_draft)
        # Original Self-Feedback equation: fb_t = M(p_fb || x || y_t)
        feedback_msgs = [{"role": "user", "content": replaced}]

        first, second, feedback = self.model.generate_response(feedback_msgs)
        return feedback

    def refine(self, refinement_prompt, current_feedback, first_draft):
        """
        Using the same player model, refine the answer based on the feedback
        given during the same round
        """
        replaced = refinement_prompt.format(transcript=as_transcript(self.history),
                                            draft=first_draft,
                                            feedback=current_feedback)
        # Original Self-Refine equation: y_1 = M(p_refine || x || y_0 || fb_0)
        refine_msgs = [{"role": "user", "content": replaced}]

        prompt_obj, _, final_answer = self.model.generate_response(refine_msgs)
        return final_answer

    def act(self, last: ClemObservation) -> str:
        # Generate first answer draft
        # Original equation from the Self-Refine paper: y0 = M(pgen ∥ x)
        # Here, pgen and x are both provided in the initial GM prompt
        # e.g., pgen = "...Use the following output format:..."
        # x = "CLUE: virile"
        _, _, draft = self.model.generate_response(self.history)

        feedback_result = self.feedback(FEEDBACK_PROMPT, draft)
        refined = self.refine(REFINEMENT_PROMPT, feedback_result, draft)

        # Observe what the Game Master actually receives, never the draft, so that
        # self.history stays in sync with the game state.
        self.observe(dict(role="assistant", content=refined))

        self.trace.append({"type": "draft",         "content": draft})
        self.trace.append({"type": "self-feedback", "content": feedback_result})
        self.trace.append({"type": "refined",       "content": refined,
                           "changed": refined.strip() != draft.strip()})

        print("[%s] draft=%r refined=%r" % (self.name, draft[-40:], refined[-40:]))

        return refined

class RefineOnlyAgent(SelfRefineAgent):
    """
    An enhanced ClemAgent which emits the game responses according to the Self-Refine
    framework, with omission of the Feedback step (an ablation of the original 
    Self-Refine schema)
    First, a so-called draft is generated (in the same way that an ordinary Player
    response is generated). Then, the model is asked to refine the draft or output
    the first version if no refinement is necessary. Finally, the refined version
    is passed on to the Game Master.

    The draft never enters self.history: the Game Master only ever
    sees the refined answer, so the history stays a faithful record of the dialogue.
    The intermediate steps go to self.trace, which the run loop collects per step.
    """
    def refine_without_feedback(self, refinement_prompt, first_draft):
        """
        Using the same player model, refine the answer.
        """
        replaced = refinement_prompt.format(transcript=as_transcript(self.history),
                                            draft=first_draft)
        # Here, y_1 = M(p_refine || x || y_0)
        refine_msgs = [{"role": "user", "content": replaced}]

        prompt_obj, _, final_answer = self.model.generate_response(refine_msgs)
        return final_answer
    
    def act(self, last: ClemObservation) -> str:
            # Generate first answer draft
            # Original equation from the Self-Refine paper: y0 = M(pgen ∥ x)
            # Here, pgen and x are both provided in the initial GM prompt
            # e.g., pgen = "...Use the following output format:..."
            # x = "CLUE: virile"
            _, _, draft = self.model.generate_response(self.history)
    
            refined = self.refine_without_feedback(REFINE_ONLY_PROMPT, draft)
    
            # Observe what the Game Master actually receives, never the draft, so that
            # self.history stays in sync with the game state.
            self.observe(dict(role="assistant", content=refined))
    
            self.trace.append({"type": "draft",         "content": draft})
            self.trace.append({"type": "refined",       "content": refined,
                               "changed": refined.strip() != draft.strip()})
    
            print("[%s] draft=%r refined=%r" % (self.name, draft[-40:], refined[-40:]))
    
            return refined


def inject_traces(episode_path, episode_traces):
    """Return a copy of interactions.json with the self-refine steps interleaved
    as player->player events. interactions.json itself is left untouched, so the
    GameScorer and the official transcript are unaffected."""

    interactions = json.load(open(episode_path / "interactions.json", encoding="utf-8"))

    player_names = []
    for name in interactions["players"]:
        if name != "GM":
            player_names.append(name)

    agent_to_player = {}
    for i, name in enumerate(player_names):
        agent_to_player["player_%d" % i] = name

    # only traces from agents this run controls, in the order they happened
    pending = []
    for trace in episode_traces:
        if agent_to_player.get(trace["agent"]) is not None:
            pending.append(trace)

    for turn_idx, turn in enumerate(interactions["turns"]):
        new_turn = []

        for event in turn:
            new_turn.append(event)

            if not pending:
                continue

            trace = pending[0]
            player_name = agent_to_player[trace["agent"]]

            if (event["from"] == "GM"
                    and event["to"] == player_name
                    and event["action"].get("content") == trace["context"]):
                for entry in trace["trace"]:
                    if entry["type"] == "refined":
                        continue          # already in the record as player->GM
                    new_turn.append({
                        "from": player_name,
                        "to": player_name,
                        "action": {
                            "type": entry["type"],
                            "content": "[%s]\n%s" % (entry["type"].upper(),
                                                     entry["content"]),
                        },
                    })
                pending.pop(0)

        interactions["turns"][turn_idx] = new_turn

    if pending:
        print("WARNING: %d trace(s) never matched a GM event" % len(pending))

    with open(episode_path / "interactions_selfrefine.json", "w", encoding="utf-8") as f:
        json.dump(interactions, f, indent=2, ensure_ascii=False)

    return interactions
