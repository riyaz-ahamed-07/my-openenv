"""Quick validation script — ASCII only for Windows compatibility."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []
passes = []


def check(name, condition, msg=""):
    if condition:
        passes.append(name)
        print(f"PASS: {name}")
    else:
        errors.append(f"{name}: {msg}")
        print(f"FAIL: {name} - {msg}")


# 1. Corpus
from tasks.ticket_corpus import TICKETS, get_ticket_by_id, get_tickets_by_difficulty

check("corpus_size", len(TICKETS) >= 10, f"got {len(TICKETS)}")
check("easy_tickets", len(get_tickets_by_difficulty("easy")) >= 3)
check("medium_tickets", len(get_tickets_by_difficulty("medium")) >= 5)
check("hard_tickets", len(get_tickets_by_difficulty("hard")) >= 3)

t001 = get_ticket_by_id("T001")
check("ticket_fields", all(k in t001 for k in ["id", "text", "true_priority", "true_department", "sla_steps"]))

# 2. Models
from models import SupportTriageAction, SupportTriageObservation, SupportTriageState

a = SupportTriageAction(ticket_id="T001", priority="urgent", department="technical", response_draft="test")
check("action_model", a.priority == "urgent")

obs = SupportTriageObservation(
    ticket_id="T001", ticket_text="test", customer_tier="enterprise",
    previous_interactions=5, is_repeat_complaint=True, sla_deadline_steps=2, remaining_tickets=3
)
check("observation_model", obs.customer_tier == "enterprise")

try:
    SupportTriageAction(ticket_id="T001", priority="critical", department="technical")
    check("invalid_rejected", False, "Should have raised")
except Exception:
    check("invalid_rejected", True)

# 3. Grader
from tasks.grader import grade_priority, grade_routing, grade_response, grade_step

s, p, n = grade_priority("urgent", "urgent", "enterprise")
check("priority_exact", s == 1.0 and p == 0.0, f"got s={s} p={p}")

s2, p2, n2 = grade_priority("high", "urgent", "pro")
check("priority_adjacent", s2 == 0.5, f"got {s2}")

s3, p3, n3 = grade_priority("low", "urgent", "enterprise")
check("priority_miss_penalty", s3 == 0.0 and p3 >= 0.30, f"got s={s3} p={p3}")

sr, pr, nr = grade_routing("technical", "technical", "urgent")
check("routing_correct", sr == 1.0 and pr == 0.0)

sr2, pr2, nr2 = grade_routing("billing", "technical", "urgent")
check("routing_wrong_penalty", sr2 == 0.0 and pr2 >= 0.20)

ticket_h = get_ticket_by_id("T201")
good_resp = "I understand your CI/CD pipeline halted due to token expiration. I am escalating to infrastructure immediately as P1. Please rotate the service account token in Settings and re-trigger the pipeline. Our team will contact you in 15 minutes with an update."
rs, rn = grade_response(good_resp, ticket_h, "technical")
check("response_good_score", rs >= 0.4, f"got {rs} notes: {rn[:60]}")

_, rn_empty = grade_response("", ticket_h, "technical")
check("response_empty", True)  # just shouldn't crash

rs_vague, _ = grade_response("Thank you valued customer we apologize for any inconvenience.", ticket_h, "technical")
check("response_vague_low", rs_vague < 0.5, f"got {rs_vague}")

# 4. Task registry
from tasks.task_registry import Episode, TASKS, list_tasks

check("tasks_count", len(TASKS) == 3)
tasks_list = list_tasks()
diffs = {t["difficulty"] for t in tasks_list}
check("task_difficulties", diffs == {"easy", "medium", "hard"})

ep = Episode("task_1_easy")
check("easy_episode_tickets", ep.total_tickets == 1)
ep.advance()
check("easy_episode_done", ep.done)

ep2 = Episode("task_2_medium")
check("medium_episode_tickets", ep2.total_tickets == 5)
for _ in range(5):
    ep2.advance()
check("medium_episode_done", ep2.done)

# 5. grade_step
ticket3 = get_ticket_by_id("T003")
action_perfect = {"ticket_id": "T003", "priority": "urgent", "department": "technical", "response_draft": ""}
r = grade_step(action_perfect, ticket3, TASKS["task_1_easy"], 0, False)
check("step_perfect_reward", r["final_reward"] >= 0.8, f"got {r['final_reward']}")

action_wrong = {"ticket_id": "T003", "priority": "low", "department": "billing", "response_draft": ""}
r2 = grade_step(action_wrong, ticket3, TASKS["task_1_easy"], 0, False)
check("step_wrong_reward", r2["final_reward"] == 0.0, f"got {r2['final_reward']}")

# Compounding penalty
r3 = grade_step(action_wrong, ticket3, TASKS["task_1_easy"], 3, False)
check("compounding_penalty", r3["compounding_penalty"] > 0, f"got {r3['compounding_penalty']}")

# SLA breach
r4 = grade_step(action_perfect, ticket3, TASKS["task_1_easy"], 0, True)
check("sla_penalty", r4["sla_penalty"] == 0.15, f"got {r4['sla_penalty']}")

# Reward range validation
all_in_range = True
for task_id, task_cfg in TASKS.items():
    for tid in ["T001", "T003", "T101", "T201"]:
        tick = get_ticket_by_id(tid)
        for p in ["urgent", "high", "medium", "low"]:
            for d in ["billing", "technical", "account", "general"]:
                a = {"ticket_id": tid, "priority": p, "department": d, "response_draft": "I will help you resolve this issue immediately and escalate it to the right team."}
                res = grade_step(a, tick, task_cfg, 0, False)
                if not (0.0 <= res["final_reward"] <= 1.0):
                    all_in_range = False
                    print(f"  OUT OF RANGE: {task_id}/{tid}/{p}/{d} -> {res['final_reward']}")
check("all_rewards_in_range", all_in_range)

# Summary
print(f"\n=== Results: {len(passes)}/{len(passes)+len(errors)} passed ===")
if errors:
    print("FAILED TESTS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
