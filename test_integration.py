"""End-to-end integration test for all 3 tasks."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.support_triage_environment import SupportTriageEnvironment
from models import SupportTriageAction

print("=== SupportTriageEnv Integration Tests ===\n")
errors = []


def run_task1():
    env = SupportTriageEnvironment()
    obs = env.reset(task_id='task_1_easy')
    assert obs.ticket_id == 'T003', f"Expected T003, got {obs.ticket_id}"
    assert obs.customer_tier == 'enterprise'
    assert obs.sla_deadline_steps == 1  # urgent ticket
    assert not obs.done

    action = SupportTriageAction(
        ticket_id=obs.ticket_id,
        priority='urgent',
        department='technical',
        response_draft=''
    )
    obs2 = env.step(action)
    assert obs2.done, "Should be done after 1 step"
    assert obs2.step_reward >= 0.8, f"Perfect action should reward >= 0.8, got {obs2.step_reward}"
    assert env.state.tickets_processed == 1
    score = env.state.cumulative_reward / env.state.max_possible_reward
    print(f"Task1 PASS: reward={obs2.step_reward:.3f}, score={score:.3f}")
    return score


def run_task2():
    env = SupportTriageEnvironment()
    obs = env.reset(task_id='task_2_medium')
    assert env.state.total_tickets == 5
    assert not obs.done

    steps = 0
    rewards = []
    while not obs.done:
        # Mix correct and wrong answers
        action = SupportTriageAction(
            ticket_id=obs.ticket_id,
            priority='high',
            department='technical',
            response_draft=''
        )
        obs = env.step(action)
        rewards.append(obs.step_reward or 0.0)
        steps += 1
        assert steps <= 6, "Should not exceed 5 steps"

    assert steps == 5, f"Expected 5 steps, got {steps}"
    assert all(0.0 <= r <= 1.0 for r in rewards), f"Rewards out of range: {rewards}"
    score = env.state.cumulative_reward / env.state.max_possible_reward
    print(f"Task2 PASS: steps={steps}, rewards={[round(r,3) for r in rewards]}, score={score:.3f}")
    return score


def run_task3():
    env = SupportTriageEnvironment()
    obs = env.reset(task_id='task_3_hard')
    assert env.state.total_tickets == 3
    assert obs.customer_tier == 'enterprise'

    responses = [
        "I understand your CI/CD pipeline has halted due to token expiration. Escalating to infrastructure as P1. Rotate service account token in Settings now and re-trigger. Our team will contact you in 15 minutes.",
        "I understand you have three critical issues. I am escalating the billing discrepancy to our account team and connecting you with an account manager immediately. I will also create tickets for the pipeline and user access issues.",
        "Thank you for flagging these issues with your plan upgrade. I am investigating the analytics access and Slack integration failures now. Our team will review your billing plan and restore proper feature access within 2 hours.",
    ]

    steps = 0
    rewards = []
    resp_idx = 0
    while not obs.done:
        action = SupportTriageAction(
            ticket_id=obs.ticket_id,
            priority='urgent' if steps == 0 else 'high',
            department='technical' if steps != 1 else 'billing',
            response_draft=responses[resp_idx] if resp_idx < len(responses) else ""
        )
        obs = env.step(action)
        rewards.append(obs.step_reward or 0.0)
        steps += 1
        resp_idx += 1

    assert steps == 3, f"Expected 3 steps, got {steps}"
    assert all(0.0 <= r <= 1.0 for r in rewards), f"Rewards out of range: {rewards}"
    score = env.state.cumulative_reward / env.state.max_possible_reward
    print(f"Task3 PASS: steps={steps}, rewards={[round(r,3) for r in rewards]}, score={score:.3f}")
    return score


def run_reset_clean_state():
    """Verify reset properly cleans up previous episode state."""
    env = SupportTriageEnvironment()
    obs1 = env.reset(task_id='task_1_easy')
    action = SupportTriageAction(ticket_id=obs1.ticket_id, priority='low', department='general', response_draft='')
    env.step(action)
    assert env.state.consecutive_wrong > 0

    # Reset again
    obs2 = env.reset(task_id='task_1_easy')
    assert env.state.consecutive_wrong == 0, "State not cleaned on reset"
    assert env.state.cumulative_reward == 0.0
    assert env.state.tickets_processed == 0
    assert not obs2.done
    print("Reset clean state PASS")


# Run all tests
tests = [
    ("Task1 easy", run_task1),
    ("Task2 medium", run_task2),
    ("Task3 hard", run_task3),
    ("Reset cleans state", run_reset_clean_state),
]

all_scores = []
for name, fn in tests:
    try:
        result = fn()
        if isinstance(result, float):
            all_scores.append(result)
    except Exception as e:
        errors.append(f"{name}: {e}")
        print(f"FAIL {name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n=== Integration Results: {len(tests)-len(errors)}/{len(tests)} passed ===")
if all_scores:
    print(f"Scores: {[round(s,3) for s in all_scores]} | Avg: {sum(all_scores)/len(all_scores):.3f}")

if errors:
    print("ERRORS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL INTEGRATION TESTS PASSED")
    sys.exit(0)
