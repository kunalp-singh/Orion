from pathlib import Path

from orion.context import InMemoryContextStore
from orion.executor import OrionActionExecutor
from orion.logging import ExperimentLogger
from orion.models import Action, ContextItem, State, TaskPhase
from orion.observer import HeuristicStateObserver
from orion.policy import NoManagementPolicy, RuleBasedOrion, StaticPolicy


def state(**overrides):
    values = dict(
        context_utilization=0.2,
        semantic_relevance=0.8,
        information_freshness=0.9,
        retrieval_confidence=0.8,
        remaining_token_budget=0.8,
        task_phase=TaskPhase.EDITING,
    )
    values.update(overrides)
    return State(**values)


def test_policies_choose_explicit_actions():
    assert StaticPolicy(0.5).decide(state(context_utilization=0.6)) is Action.COMPRESS
    assert RuleBasedOrion().decide(state(semantic_relevance=0.1)) is Action.RETRIEVE


def test_rule_based_policy_reaches_every_action_branch():
    policy = RuleBasedOrion()
    assert policy.decide(state(semantic_relevance=0.2)) is Action.RETRIEVE
    assert policy.decide(state(context_utilization=0.8)) is Action.COMPRESS
    assert policy.decide(
        state(context_utilization=0.6, task_phase=TaskPhase.TESTING)
    ) is Action.ISOLATE
    assert policy.decide(state(information_freshness=0.2)) is Action.STORE
    assert policy.decide(state()) is Action.IGNORE
    assert NoManagementPolicy().decide(state()) is Action.IGNORE


def test_actions_store_and_retrieve():
    store = InMemoryContextStore()
    stored = store.add(ContextItem(content="important API constraint", importance=1.0))
    executor = OrionActionExecutor(store)
    context = [{"role": "user", "content": "important API constraint"}]
    assert executor.execute(Action.RETRIEVE, context, state)
    trace = executor.last_retrieval
    assert trace is not None
    assert trace.query == "important API constraint"
    assert trace.item_ids == [stored.metadata["_id"]]
    assert trace.content_lengths == [24]
    assert trace.context_before["message_count"] == 1
    assert trace.context_after["message_count"] == 2
    assert trace.retrieval_tokens_added == 6
    assert trace.retrieval_latency_ms >= 0
    executor.execute(Action.STORE, context, state)
    assert executor.last_retrieval is None


def test_retrieval_trace_records_relevance_when_available():
    store = InMemoryContextStore()
    store.add(
        ContextItem(
            content="retrievable fact",
            metadata={"similarity": 0.75},
        )
    )
    executor = OrionActionExecutor(store)
    executor.execute(
        Action.RETRIEVE,
        [{"role": "user", "content": "retrievable fact"}],
        state,
    )
    assert executor.last_retrieval is not None
    assert executor.last_retrieval.relevance_scores == [0.75]


def test_context_store_returns_best_matching_item():
    store = InMemoryContextStore()
    best = store.add(ContextItem(content="calculator add integer behavior", importance=1.0))
    store.add(ContextItem(content="unrelated deployment notes", importance=1.0))
    results = store.retrieve("calculator add behavior")
    assert results[0].metadata["_id"] == best.metadata["_id"]
    assert results[0].metadata["similarity"] > 0


def test_ingestion_adds_provenanced_context():
    store = InMemoryContextStore()
    executor = OrionActionExecutor(store)
    item = executor.ingest(
        "read calculator.py: def add(a, b): return a + b",
        source_type="file_content",
        task_id="add_numbers",
        run_id="run-1",
    )
    assert item.source == "file_content"
    assert item.task_id == "add_numbers"
    assert item.metadata["source_type"] == "file_content"
    assert store.retrieve("calculator add")[0].metadata["_id"] == item.metadata["_id"]


def test_retrieval_returns_zero_and_adds_zero_tokens():
    executor = OrionActionExecutor(InMemoryContextStore())
    executor.execute(Action.RETRIEVE, [{"role": "user", "content": "missing fact"}], state)
    trace = executor.last_retrieval
    assert trace is not None
    assert trace.item_ids == []
    assert trace.retrieval_tokens_added == 0
    assert trace.context_before == trace.context_after


def test_observer_and_jsonl_logger(tmp_path: Path):
    observed = HeuristicStateObserver(100).observe([{"role": "user", "content": "test this change"}])
    assert 0 <= observed.context_utilization <= 1
    logger = ExperimentLogger(tmp_path / "run.jsonl")
    logger.log({"step": 1, "action": "ignore"})
    output = logger.export_csv(tmp_path / "run.csv")
    assert output.exists()
