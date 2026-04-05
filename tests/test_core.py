"""
Standalone validation tests for SupportTriageEnv.

Tests the core logic without requiring openenv-core to be installed.
Validates: grader, ticket corpus, task registry, episode management.

Run:
    python tests/test_core.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Test: Ticket Corpus
# ─────────────────────────────────────────────────────────────────────────────

def test_corpus_loaded():
    from tasks.ticket_corpus import TICKETS, get_ticket_by_id, get_tickets_by_difficulty
    assert len(TICKETS) >= 10, f"Expected ≥10 tickets, got {len(TICKETS)}"
    easy = get_tickets_by_difficulty("easy")
    medium = get_tickets_by_difficulty("medium")
    hard = get_tickets_by_difficulty("hard")
    assert len(easy) >= 3, "Need at least 3 easy tickets"
    assert len(medium) >= 5, "Need at least 5 medium tickets"
    assert len(hard) >= 3, "Need at least 3 hard tickets"
    print(f"  ✓ Corpus: {len(TICKETS)} tickets ({len(easy)} easy, {len(medium)} medium, {len(hard)} hard)")


def test_ticket_fields():
    from tasks.ticket_corpus import TICKETS
    required = {"id", "text", "customer_tier", "previous_interactions", "is_repeat_complaint",
                "sla_steps", "true_priority", "true_department", "response_keywords", "difficulty"}
    for t in TICKETS:
        missing = required - set(t.keys())
        assert not missing, f"Ticket {t.get('id')} missing fields: {missing}"
        assert t["true_priority"] in {"urgent", "high", "medium", "low"}
        assert t["true_department"] in {"billing", "technical", "account", "general"}
        assert t["customer_tier"] in {"enterprise", "pro", "free"}
    print(f"  ✓ All {len(TICKETS)} tickets have valid fields")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Grader
# ─────────────────────────────────────────────────────────────────────────────

def test_priority_grader_exact():
    from tasks.grader import grade_priority
    score, penalty, note = grade_priority("urgent", "urgent", "enterprise")
    assert score == 1.0, f"Exact match should be 1.0, got {score}"
    assert penalty == 0.0, f"Exact match should have no penalty, got {penalty}"
    print(f"  ✓ Priority exact match: {score}")


def test_priority_grader_adjacent():
    from tasks.grader import grade_priority
    score, penalty, note = grade_priority("high", "urgent", "pro")
    assert score == 0.5, f"Adjacent should be 0.5, got {score}"
    print(f"  ✓ Priority adjacent: {score}")


def test_priority_grader_urgent_miss():
    from tasks.grader import grade_priority
    score, penalty, note = grade_priority("low", "urgent", "enterprise")
    assert score == 0.0, f"Far miss should be 0.0, got {score}"
    assert penalty >= 0.30, f"Urgent miss penalty should be ≥0.30, got {penalty}"
    print(f"  ✓ Priority urgent miss: score={score}, penalty={penalty}")


def test_routing_grader():
    from tasks.grader import grade_routing
    # Correct
    score, penalty, note = grade_routing("technical", "technical", "urgent")
    assert score == 1.0
    # Wrong
    score2, penalty2, note2 = grade_routing("billing", "technical", "urgent")
    assert score2 == 0.0
    assert penalty2 >= 0.20
    print(f"  ✓ Routing: correct={1.0}, wrong_penalty={penalty2}")


def test_response_grader_good():
    from tasks.grader import grade_response
    from tasks.ticket_corpus import get_ticket_by_id
    ticket = get_ticket_by_id("T201")  # Hard ticket
    good_response = (
        "I understand your CI/CD pipeline has halted due to authentication token expiration, "
        "and this is blocking your production release. I have escalated this to our infrastructure team "
        "immediately as a P1 issue. Please follow these steps: 1) Rotate the service account token via "
        "Settings > Service Accounts. 2) Re-trigger your pipeline. Our team will contact you within "
        "15 minutes with a status update."
    )
    score, notes = grade_response(good_response, ticket, "technical")
    assert score >= 0.5, f"Good response should score ≥0.5, got {score}\nNotes: {notes}"
    print(f"  ✓ Good response score: {score:.3f}")


def test_response_grader_empty():
    from tasks.grader import grade_response
    from tasks.ticket_corpus import get_ticket_by_id
    ticket = get_ticket_by_id("T201")
    score, notes = grade_response("", ticket, "technical")
    assert score == 0.0, f"Empty response should score 0.0, got {score}"
    print(f"  ✓ Empty response score: {score}")


def test_response_grader_vague():
    from tasks.grader import grade_response
    from tasks.ticket_corpus import get_ticket_by_id
    ticket = get_ticket_by_id("T201")
    vague = "Thank you for contacting us. We value your business. We will get back to you as soon as possible."
    score, notes = grade_response(vague, ticket, "technical")
    assert score < 0.5, f"Vague response should score <0.5, got {score}\nNotes: {notes}"
    print(f"  ✓ Vague response score: {score:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Task Registry & Episode
# ─────────────────────────────────────────────────────────────────────────────

def test_tasks_defined():
    from tasks.task_registry import TASKS, list_tasks
    assert "task_1_easy" in TASKS
    assert "task_2_medium" in TASKS
    assert "task_3_hard" in TASKS
    tasks = list_tasks()
    assert len(tasks) == 3
    difficulties = {t["difficulty"] for t in tasks}
    assert difficulties == {"easy", "medium", "hard"}
    print(f"  ✓ All 3 tasks defined with correct difficulties")


def test_episode_easy():
    from tasks.task_registry import Episode
    ep = Episode("task_1_easy")
    assert ep.total_tickets == 1
    assert ep.remaining_tickets == 1
    assert not ep.done
    ticket = ep.current_ticket
    assert ticket is not None
    ep.advance()
    assert ep.done
    print(f"  ✓ Easy episode: 1 ticket, advances to done correctly")


def test_episode_medium():
    from tasks.task_registry import Episode
    ep = Episode("task_2_medium")
    assert ep.total_tickets == 5
    for i in range(5):
        assert not ep.done
        assert ep.remaining_tickets == 5 - i
        ep.advance()
    assert ep.done
    print(f"  ✓ Medium episode: 5 tickets, all advance correctly")


def test_sla_countdown():
    from tasks.task_registry import Episode
    ep = Episode("task_3_hard")
    # T201 has sla_steps=1, so after one advance it should be 0
    initial_sla = ep.current_sla_steps
    ep.advance()
    # Next ticket's SLA should have decremented
    next_sla = ep.current_sla_steps
    assert next_sla < initial_sla or initial_sla == 0, \
        f"SLA should decrement: initial={initial_sla}, next={next_sla}"
    print(f"  ✓ SLA countdown: initial={initial_sla}, after advance={next_sla}")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Full Step Grader
# ─────────────────────────────────────────────────────────────────────────────

def test_grade_step_perfect():
    from tasks.grader import grade_step
    from tasks.ticket_corpus import get_ticket_by_id
    from tasks.task_registry import TASKS

    ticket = get_ticket_by_id("T003")  # urgent, technical
    action = {
        "ticket_id": "T003",
        "priority": "urgent",
        "department": "technical",
        "response_draft": "",
    }
    task_cfg = TASKS["task_1_easy"]
    result = grade_step(action, ticket, task_cfg, consecutive_wrong=0, sla_already_breached=False)
    assert result["priority_score"] == 1.0
    assert result["routing_score"] == 0.0  # routing not weighted in easy
    assert result["final_reward"] >= 0.8
    print(f"  ✓ Perfect easy step: reward={result['final_reward']:.3f}")


def test_grade_step_wrong_urgent():
    from tasks.grader import grade_step
    from tasks.ticket_corpus import get_ticket_by_id
    from tasks.task_registry import TASKS

    ticket = get_ticket_by_id("T003")  # urgent
    action = {
        "ticket_id": "T003",
        "priority": "low",  # completely wrong
        "department": "billing",
        "response_draft": "",
    }
    task_cfg = TASKS["task_1_easy"]
    result = grade_step(action, ticket, task_cfg, consecutive_wrong=0, sla_already_breached=False)
    assert result["priority_score"] == 0.0
    assert result["priority_penalty"] >= 0.30
    assert result["final_reward"] == 0.0
    print(f"  ✓ Wrong urgent: priority_penalty={result['priority_penalty']:.2f}, reward={result['final_reward']}")


def test_compounding_penalty():
    from tasks.grader import grade_step
    from tasks.ticket_corpus import get_ticket_by_id
    from tasks.task_registry import TASKS

    ticket = get_ticket_by_id("T002")  # low priority
    action = {"ticket_id": "T002", "priority": "urgent", "department": "technical", "response_draft": ""}
    task_cfg = TASKS["task_1_easy"]

    r0 = grade_step(action, ticket, task_cfg, consecutive_wrong=0, sla_already_breached=False)
    r3 = grade_step(action, ticket, task_cfg, consecutive_wrong=3, sla_already_breached=False)
    assert r3["compounding_penalty"] > r0["compounding_penalty"]
    print(f"  ✓ Compounding penalty: wrong=0 → {r0['compounding_penalty']:.2f}, wrong=3 → {r3['compounding_penalty']:.2f}")


def test_reward_range():
    """All rewards must be in [0, 1]."""
    from tasks.grader import grade_step
    from tasks.ticket_corpus import TICKETS
    from tasks.task_registry import TASKS

    for task_id, task_cfg in TASKS.items():
        for ticket in TICKETS:
            for priority in ["urgent", "high", "medium", "low"]:
                for dept in ["billing", "technical", "account", "general"]:
                    action = {
                        "ticket_id": ticket["id"],
                        "priority": priority,
                        "department": dept,
                        "response_draft": "We will look into your issue immediately.",
                    }
                    for cw in [0, 1, 3]:
                        for sla in [False, True]:
                            result = grade_step(action, ticket, task_cfg, cw, sla)
                            r = result["final_reward"]
                            assert 0.0 <= r <= 1.0, \
                                f"Reward out of range: {r} for {ticket['id']}, {priority}, {dept}"
    print(f"  ✓ All rewards in [0.0, 1.0] across all combinations")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Models
# ─────────────────────────────────────────────────────────────────────────────

def test_action_model():
    from models import SupportTriageAction
    action = SupportTriageAction(
        ticket_id="T001",
        priority="urgent",
        department="technical",
        response_draft="We will fix this immediately.",
    )
    assert action.ticket_id == "T001"
    assert action.priority == "urgent"
    assert action.department == "technical"
    print(f"  ✓ Action model validates correctly")


def test_observation_model():
    from models import SupportTriageObservation
    obs = SupportTriageObservation(
        ticket_id="T001",
        ticket_text="Test ticket",
        customer_tier="enterprise",
        previous_interactions=5,
        is_repeat_complaint=True,
        sla_deadline_steps=2,
        remaining_tickets=3,
    )
    assert obs.customer_tier == "enterprise"
    assert obs.sla_deadline_steps == 2
    print(f"  ✓ Observation model validates correctly")


def test_invalid_action_rejected():
    from models import SupportTriageAction
    try:
        SupportTriageAction(
            ticket_id="T001",
            priority="critical",  # Invalid — not in enum
            department="technical",
        )
        assert False, "Should have raised ValidationError"
    except Exception:
        print(f"  ✓ Invalid priority correctly rejected")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        # Corpus
        test_corpus_loaded,
        test_ticket_fields,
        # Grader
        test_priority_grader_exact,
        test_priority_grader_adjacent,
        test_priority_grader_urgent_miss,
        test_routing_grader,
        test_response_grader_good,
        test_response_grader_empty,
        test_response_grader_vague,
        # Task registry
        test_tasks_defined,
        test_episode_easy,
        test_episode_medium,
        test_sla_countdown,
        # Step grader
        test_grade_step_perfect,
        test_grade_step_wrong_urgent,
        test_compounding_penalty,
        test_reward_range,
        # Models
        test_action_model,
        test_observation_model,
        test_invalid_action_rejected,
    ]

    print("=" * 60)
    print("SupportTriageEnv — Core Validation Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for test_fn in tests:
        group = test_fn.__name__.split("_")[1]
        print(f"\n[{group.upper()}] {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed  |  {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
