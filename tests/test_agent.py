from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import pytest

from app.agents.base import AgentRequest
from app.agents.context import format_context
from app.agents.graph import (
    CRITIC,
    GENERATOR,
    MEMORY_LOADER,
    MEMORY_WRITER,
    PLANNER,
    RETRIEVER,
    TOOLS,
)
from app.agents.models import (
    Critique,
    PlannedFilter,
    PlannedToolCall,
    QueryAnalysis,
    RetrievalPlan,
)
from app.agents.nodes.analyzer import QueryAnalyzerNode
from app.agents.nodes.critic import CriticNode
from app.agents.nodes.generator import AnswerGeneratorNode
from app.agents.nodes.memory_loader import MemoryLoaderNode
from app.agents.nodes.memory_writer import MemoryWriterNode
from app.agents.nodes.planner import PlannerNode
from app.agents.nodes.retriever import RetrieverNode
from app.agents.nodes.tools import ToolNode
from app.agents.rag import AgenticRagAgent
from app.agents.state import RAGState, initial_state
from app.config.settings import AgentSettings, MemorySettings, ToolSettings
from app.core.exceptions import DependencyError
from app.memory.store import InMemoryStore
from app.models.base import Schema
from app.retrieval.base import Document, ScoredDocument
from app.retrieval.filters import FilterOperator
from app.retrieval.query import RetrievalQuery, RetrievalResult, SearchType
from app.tools.base import ToolRegistry
from app.tools.calculator import CalculatorTool

# --------------------------------------------------------------------------- doubles


class FakeChatService:
    """Scripted chat service: queued replies per schema, plus recorded prompts."""

    def __init__(self, **scripts: Sequence[Any]) -> None:
        self._scripts = {name: deque(values) for name, values in scripts.items()}
        self.prompts: list[tuple[str, str]] = []
        self.fail_on: set[str] = set()

    def _next(self, key: str) -> Any:
        queue = self._scripts.get(key)
        if not queue:
            raise AssertionError(f"FakeChatService has no scripted reply for {key!r}")
        return queue[0] if len(queue) == 1 else queue.popleft()

    async def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        if "text" in self.fail_on:
            raise DependencyError("chat down")
        return str(self._next("text"))

    async def structured(self, schema: type[Schema], system: str, user: str) -> Any:
        self.prompts.append((system, user))
        if schema.__name__ in self.fail_on:
            raise DependencyError("chat down")
        return self._next(schema.__name__)

    def prompt_containing(self, needle: str) -> str | None:
        return next((system for system, _ in self.prompts if needle in system), None)


class FakeRetriever:
    def __init__(self, documents: Sequence[ScoredDocument], fail: bool = False) -> None:
        self._documents = tuple(documents)
        self._fail = fail
        self.queries: list[RetrievalQuery] = []

    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        self.queries.append(query)
        if self._fail:
            raise DependencyError("store down")
        return RetrievalResult(
            documents=self._documents,
            search_type=query.search_type or SearchType.SIMILARITY,
            retriever="fake",
            candidates=len(self._documents),
        )


class FakeRetrieverFactory:
    def __init__(self, retriever: FakeRetriever) -> None:
        self._retriever = retriever

    def for_query(self, query: RetrievalQuery) -> FakeRetriever:
        return self._retriever


def _document(identifier: str, content: str, page: str = "1") -> ScoredDocument:
    return ScoredDocument(
        document=Document(
            id=identifier, content=content, metadata={"filename": "handbook.pdf", "page": page}
        ),
        score=0.9,
    )


def _settings(**overrides: Any) -> AgentSettings:
    return AgentSettings(**({"max_revisions": 2, "min_confidence": 0.5} | overrides))


def _agent(
    chat: FakeChatService,
    retriever: FakeRetriever,
    settings: AgentSettings | None = None,
    *,
    memory: MemorySettings | None = None,
    tools: ToolRegistry | None = None,
    store: InMemoryStore | None = None,
) -> AgenticRagAgent:
    resolved = settings or _settings()
    memory_settings = memory or MemorySettings(enabled=False)
    memory_store = store or InMemoryStore(memory_settings)
    registry = tools if tools is not None else ToolRegistry()
    return AgenticRagAgent(
        MemoryLoaderNode(memory_store, memory_settings),
        QueryAnalyzerNode(chat),  # type: ignore[arg-type]
        PlannerNode(chat, resolved, registry),  # type: ignore[arg-type]
        ToolNode(registry, ToolSettings()),
        RetrieverNode(FakeRetrieverFactory(retriever)),  # type: ignore[arg-type]
        AnswerGeneratorNode(chat, resolved),  # type: ignore[arg-type]
        CriticNode(chat, resolved),  # type: ignore[arg-type]
        MemoryWriterNode(memory_store, chat, memory_settings),  # type: ignore[arg-type]
        resolved,
    )


def _request() -> AgentRequest:
    return AgentRequest(session_id="s1", query="What is the leave policy?")


_ANALYSIS = QueryAnalysis(
    intent="factual", normalised_query="leave policy", keywords=("leave", "policy")
)
_PLAN = RetrievalPlan(search_type=SearchType.SIMILARITY, top_k=4, search_text="leave policy")
_ACCEPT = Critique(sufficient_context=True, grounded=True, confidence=0.9)
_REJECT = Critique(
    sufficient_context=False,
    grounded=True,
    confidence=0.9,
    feedback="search for holiday entitlement",
)


# ----------------------------------------------------------------------- graph shape


def test_graph_matches_the_specified_workflow() -> None:
    agent = _agent(
        FakeChatService(QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN], Critique=[_ACCEPT]),
        FakeRetriever([_document("d1", "text")]),
    )

    drawn = agent.graph.get_graph()
    nodes = set(drawn.nodes)
    edges = {(edge.source, edge.target) for edge in drawn.edges}

    assert {
        MEMORY_LOADER,
        "query_analyzer",
        PLANNER,
        TOOLS,
        RETRIEVER,
        GENERATOR,
        CRITIC,
        MEMORY_WRITER,
    } <= nodes
    assert ("__start__", MEMORY_LOADER) in edges
    assert (MEMORY_LOADER, "query_analyzer") in edges
    assert ("query_analyzer", PLANNER) in edges
    assert {(PLANNER, TOOLS), (PLANNER, RETRIEVER), (PLANNER, GENERATOR)} <= edges
    assert {(TOOLS, RETRIEVER), (TOOLS, GENERATOR)} <= edges
    assert (RETRIEVER, GENERATOR) in edges
    assert (GENERATOR, CRITIC) in edges
    assert {(CRITIC, PLANNER), (CRITIC, MEMORY_WRITER)} <= edges
    assert (MEMORY_WRITER, "__end__") in edges


# ------------------------------------------------------------------------ happy path


async def test_accepted_answer_runs_each_node_once() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS],
        RetrievalPlan=[_PLAN],
        text=["Staff receive 25 days [1]."],
        Critique=[_ACCEPT],
    )
    retriever = FakeRetriever([_document("d1", "Staff receive 25 days of leave.")])

    response = await _agent(chat, retriever).invoke(_request())

    assert response.answer == "Staff receive 25 days [1]."
    assert response.session_id == "s1"
    assert response.metadata["revisions"] == 0
    assert [step.split(":")[0] for step in response.metadata["trace"]] == [
        MEMORY_LOADER,
        "query_analyzer",
        PLANNER,
        RETRIEVER,
        GENERATOR,
        CRITIC,
        MEMORY_WRITER,
    ]


async def test_plan_drives_the_retrieval_query() -> None:
    plan = RetrievalPlan(
        search_type=SearchType.MMR,
        top_k=9,
        search_text="holiday entitlement",
        filters=(PlannedFilter(field="filename", values=("handbook.pdf",)),),
    )
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[plan], text=["answer"], Critique=[_ACCEPT]
    )
    retriever = FakeRetriever([_document("d1", "text")])

    await _agent(chat, retriever).invoke(_request())

    issued = retriever.queries[0]
    assert issued.text == "holiday entitlement"
    assert issued.top_k == 9
    assert issued.search_type is SearchType.MMR
    assert issued.filters is not None
    assert issued.filters.to_chroma() == {"filename": {"$eq": "handbook.pdf"}}


async def test_response_metadata_reports_the_run() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN], text=["answer [1]"], Critique=[_ACCEPT]
    )
    retriever = FakeRetriever([_document("d1", "text", page="7")])

    metadata = (await _agent(chat, retriever).invoke(_request())).metadata

    assert metadata["plan"]["search_type"] == "similarity"
    assert metadata["plan"]["top_k"] == 4
    assert metadata["critique"]["grounded"] is True
    assert metadata["citations"] == [
        {"marker": 1, "id": "d1", "filename": "handbook.pdf", "page": "7", "score": 0.9}
    ]


# ------------------------------------------------------------------------ critic loop


async def test_insufficient_context_returns_to_the_planner() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS],
        RetrievalPlan=[_PLAN, _PLAN],
        text=["draft", "revised"],
        Critique=[_REJECT, _ACCEPT],
    )
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever).invoke(_request())

    assert response.answer == "revised"
    assert response.metadata["revisions"] == 1
    assert len(retriever.queries) == 2
    assert [step.split(":")[0] for step in response.metadata["trace"]].count(PLANNER) == 2


async def test_replanning_passes_critic_feedback_forward() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS],
        RetrievalPlan=[_PLAN, _PLAN],
        text=["draft", "revised"],
        Critique=[_REJECT, _ACCEPT],
    )

    await _agent(chat, FakeRetriever([_document("d1", "text")])).invoke(_request())

    replan_prompt = chat.prompt_containing("search for holiday entitlement")
    assert replan_prompt is not None
    assert "previous attempt was judged inadequate" in replan_prompt


async def test_the_loop_is_bounded_by_max_revisions() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS],
        RetrievalPlan=[_PLAN],
        text=["draft"],
        Critique=[_REJECT],
    )
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever, _settings(max_revisions=2)).invoke(_request())

    assert response.metadata["revisions"] == 2
    assert len(retriever.queries) == 3  # the initial attempt plus two replans


async def test_no_replanning_is_allowed_when_the_budget_is_zero() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN], text=["draft"], Critique=[_REJECT]
    )
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever, _settings(max_revisions=0)).invoke(_request())

    assert response.metadata["revisions"] == 0
    assert len(retriever.queries) == 1


async def test_a_grounding_failure_does_not_trigger_retrieval_again() -> None:
    ungrounded = Critique(sufficient_context=True, grounded=False, confidence=0.9)
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN], text=["draft"], Critique=[ungrounded]
    )
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever).invoke(_request())

    # Re-retrieving cannot fix an answer that strayed from context it already had.
    assert len(retriever.queries) == 1
    assert response.metadata["critique"]["grounded"] is False


async def test_low_confidence_is_not_accepted() -> None:
    unsure = Critique(sufficient_context=False, grounded=True, confidence=0.2)
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS],
        RetrievalPlan=[_PLAN],
        text=["draft"],
        Critique=[unsure, _ACCEPT],
    )
    retriever = FakeRetriever([_document("d1", "text")])

    await _agent(chat, retriever, _settings(min_confidence=0.8)).invoke(_request())

    assert len(retriever.queries) == 2


# ------------------------------------------------------------------- no-retrieval path


async def test_planner_can_decide_retrieval_is_unnecessary() -> None:
    plan = RetrievalPlan(retrieval_needed=False, search_text="hello")
    chat = FakeChatService(QueryAnalysis=[_ANALYSIS], RetrievalPlan=[plan], text=["hi"])
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever).invoke(_request())

    assert retriever.queries == []
    assert "could not find anything" in response.answer
    assert response.metadata["critique"]["sufficient_context"] is False


async def test_empty_retrieval_is_critiqued_without_a_model_call() -> None:
    chat = FakeChatService(QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN])
    retriever = FakeRetriever([])

    response = await _agent(chat, retriever, _settings(max_revisions=0)).invoke(_request())

    assert response.metadata["critique"]["confidence"] == 1.0
    assert "Broaden the search" in response.metadata["critique"]["feedback"]


# ------------------------------------------------------------------- skipping stages


def _tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


_TOOL_PLAN = RetrievalPlan(
    retrieval_needed=False,
    tools_needed=True,
    tool_calls=(PlannedToolCall(tool="calculator", input="12 * 12"),),
    search_text="unused",
)


async def test_tools_are_skipped_when_the_plan_asks_for_none() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN], text=["answer"], Critique=[_ACCEPT]
    )
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever, tools=_tools()).invoke(_request())

    steps = [step.split(":")[0] for step in response.metadata["trace"]]
    assert TOOLS not in steps
    assert RETRIEVER in steps


async def test_retrieval_is_skipped_when_the_plan_asks_for_none() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_TOOL_PLAN], text=["144"], Critique=[_ACCEPT]
    )
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever, tools=_tools()).invoke(_request())

    steps = [step.split(":")[0] for step in response.metadata["trace"]]
    assert TOOLS in steps
    assert RETRIEVER not in steps
    assert retriever.queries == []
    assert response.answer == "144"


async def test_both_stages_run_when_the_plan_asks_for_both() -> None:
    plan = _TOOL_PLAN.model_copy(update={"retrieval_needed": True})
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[plan], text=["answer"], Critique=[_ACCEPT]
    )
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever, tools=_tools()).invoke(_request())

    steps = [step.split(":")[0] for step in response.metadata["trace"]]
    assert steps.index(TOOLS) < steps.index(RETRIEVER)
    assert len(retriever.queries) == 1


async def test_both_stages_are_skipped_when_the_plan_asks_for_neither() -> None:
    plan = RetrievalPlan(retrieval_needed=False, tools_needed=False, search_text="hi")
    chat = FakeChatService(QueryAnalysis=[_ANALYSIS], RetrievalPlan=[plan], text=["hello"])
    retriever = FakeRetriever([_document("d1", "text")])

    response = await _agent(chat, retriever, tools=_tools()).invoke(_request())

    steps = [step.split(":")[0] for step in response.metadata["trace"]]
    assert TOOLS not in steps
    assert RETRIEVER not in steps
    assert "could not find anything" in response.answer


async def test_a_tool_result_alone_is_enough_to_answer() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS],
        RetrievalPlan=[_TOOL_PLAN],
        text=["It is 144."],
        Critique=[_ACCEPT],
    )

    response = await _agent(chat, FakeRetriever([]), tools=_tools()).invoke(_request())

    assert response.answer == "It is 144."
    assert response.metadata["tools"] == [
        {"tool": "calculator", "input": "12 * 12", "ok": True, "error": ""}
    ]


async def test_a_flag_without_calls_does_not_enter_the_tool_node() -> None:
    plan = RetrievalPlan(tools_needed=True, tool_calls=(), search_text="q")
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[plan], text=["answer"], Critique=[_ACCEPT]
    )

    response = await _agent(chat, FakeRetriever([_document("d1", "text")]), tools=_tools()).invoke(
        _request()
    )

    assert TOOLS not in [step.split(":")[0] for step in response.metadata["trace"]]


async def test_the_planner_is_told_which_tools_exist() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN], text=["answer"], Critique=[_ACCEPT]
    )

    await _agent(chat, FakeRetriever([_document("d1", "t")]), tools=_tools()).invoke(_request())

    assert chat.prompt_containing("- calculator:") is not None


# ---------------------------------------------------------------------- memory in flow


async def test_memory_from_one_turn_reaches_the_next() -> None:
    settings = MemorySettings(short_term_turns=4, summary_after_turns=4, extract_user_context=False)
    store = InMemoryStore(settings)
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS],
        RetrievalPlan=[_PLAN],
        text=["first answer", "second answer"],
        Critique=[_ACCEPT],
    )
    agent = _agent(chat, FakeRetriever([_document("d1", "t")]), memory=settings, store=store)

    await agent.invoke(_request())
    await agent.invoke(AgentRequest(session_id="s1", query="and what about sick leave?"))

    assert chat.prompt_containing("first answer") is not None
    assert len(await store.history("s1")) == 4


async def test_a_run_is_remembered_even_when_the_critic_never_accepts() -> None:
    settings = MemorySettings(short_term_turns=4, summary_after_turns=4, extract_user_context=False)
    store = InMemoryStore(settings)
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN], text=["draft"], Critique=[_REJECT]
    )
    agent = _agent(
        chat,
        FakeRetriever([_document("d1", "t")]),
        _settings(max_revisions=1),
        memory=settings,
        store=store,
    )

    response = await agent.invoke(_request())

    assert response.metadata["revisions"] == 1
    assert len(await store.history("s1")) == 2


# ------------------------------------------------------------------ node independence


async def test_each_node_runs_standalone_from_a_hand_built_state() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS], RetrievalPlan=[_PLAN], text=["answer"], Critique=[_ACCEPT]
    )
    retriever = FakeRetriever([_document("d1", "text")])
    state: RAGState = initial_state("s1", "question")

    analysed = await QueryAnalyzerNode(chat)(state)  # type: ignore[arg-type]
    planned = await PlannerNode(chat, _settings())({**state, **analysed})  # type: ignore[arg-type]
    retrieved = await RetrieverNode(FakeRetrieverFactory(retriever))({**state, **planned})  # type: ignore[arg-type]
    generated = await AnswerGeneratorNode(chat, _settings())({**state, **retrieved})  # type: ignore[arg-type]
    critiqued = await CriticNode(chat, _settings())({**state, **retrieved, **generated})  # type: ignore[arg-type]

    assert analysed["analysis"] == _ANALYSIS
    assert planned["plan"] == _PLAN
    assert retrieved["documents"]
    assert generated["answer"] == "answer"
    assert critiqued["critique"] == _ACCEPT


async def test_nodes_ignore_state_they_do_not_own() -> None:
    retriever = FakeRetriever([_document("d1", "text")])
    node = RetrieverNode(FakeRetrieverFactory(retriever))  # type: ignore[arg-type]

    update = await node(RAGState(query="q", plan=_PLAN))

    assert set(update) == {"documents", "trace"}


# ------------------------------------------------------------------------- degradation


async def test_analyzer_failure_falls_back_to_the_raw_question() -> None:
    chat = FakeChatService(QueryAnalysis=[_ANALYSIS])
    chat.fail_on = {"QueryAnalysis"}

    update = await QueryAnalyzerNode(chat)(initial_state("s1", "raw question"))  # type: ignore[arg-type]

    assert update["analysis"] is not None
    assert update["analysis"].normalised_query == "raw question"
    assert update["trace"] == ["query_analyzer:fallback"]


async def test_planner_failure_still_produces_a_usable_plan() -> None:
    chat = FakeChatService(RetrievalPlan=[_PLAN])
    chat.fail_on = {"RetrievalPlan"}

    update = await PlannerNode(chat, _settings())(  # type: ignore[arg-type]
        RAGState(query="raw question", analysis=_ANALYSIS)
    )

    plan = update["plan"]
    assert plan is not None
    assert plan.retrieval_needed is True
    assert plan.search_text == "leave policy"


async def test_retrieval_failure_yields_no_documents_rather_than_an_error() -> None:
    node = RetrieverNode(FakeRetrieverFactory(FakeRetriever([], fail=True)))  # type: ignore[arg-type]

    update = await node(RAGState(query="q", plan=_PLAN))

    assert update["documents"] == []
    assert update["trace"] == ["retriever:failed"]


async def test_a_failed_critic_records_zero_confidence_instead_of_approving() -> None:
    chat = FakeChatService(Critique=[_ACCEPT])
    chat.fail_on = {"Critique"}

    update = await CriticNode(chat, _settings())(  # type: ignore[arg-type]
        RAGState(query="q", answer="a", documents=[_document("d1", "text")])
    )

    critique = update["critique"]
    assert critique is not None
    assert critique.confidence == 0.0
    assert critique.accepts(min_confidence=0.5) is False


# ---------------------------------------------------------------------------- context


def test_context_blocks_are_numbered_and_attributed() -> None:
    rendered = format_context(
        [_document("d1", "first", page="3"), _document("d2", "second", page="4")], limit=5
    )

    assert "[1] (handbook.pdf, page 3)\nfirst" in rendered
    assert "[2] (handbook.pdf, page 4)\nsecond" in rendered


def test_context_respects_the_document_limit() -> None:
    documents = [_document(f"d{index}", f"text {index}") for index in range(5)]

    rendered = format_context(documents, limit=2)

    assert "[2]" in rendered
    assert "[3]" not in rendered


def test_empty_context_says_so() -> None:
    assert format_context([], limit=5) == "No documents were retrieved."


# ----------------------------------------------------------------------------- models


def test_planned_filters_convert_to_retrieval_filters() -> None:
    plan = RetrievalPlan(
        filters=(
            PlannedFilter(field="filename", values=("a.pdf",)),
            PlannedFilter(field="page", operator=FilterOperator.IN, values=("1", "2")),
            PlannedFilter(field="", values=("ignored",)),
            PlannedFilter(field="page", values=()),
        )
    )

    filters = plan.to_metadata_filters()

    assert filters is not None
    assert filters.to_chroma() == {
        "$and": [{"filename": {"$eq": "a.pdf"}}, {"page": {"$in": ["1", "2"]}}]
    }


def test_a_plan_without_filters_produces_no_clause() -> None:
    assert RetrievalPlan().to_metadata_filters() is None


def test_critique_acceptance_requires_all_three_conditions() -> None:
    assert Critique(confidence=0.9).accepts(min_confidence=0.5)
    assert not Critique(sufficient_context=False, confidence=0.9).accepts(min_confidence=0.5)
    assert not Critique(grounded=False, confidence=0.9).accepts(min_confidence=0.5)
    assert not Critique(confidence=0.4).accepts(min_confidence=0.5)


# --------------------------------------------------------------------------- streaming


async def test_stream_yields_each_answer_the_run_produces() -> None:
    chat = FakeChatService(
        QueryAnalysis=[_ANALYSIS],
        RetrievalPlan=[_PLAN, _PLAN],
        text=["draft", "revised"],
        Critique=[_REJECT, _ACCEPT],
    )
    agent = _agent(chat, FakeRetriever([_document("d1", "text")]))

    chunks = [chunk async for chunk in agent.stream(_request())]

    assert chunks == ["draft", "revised"]


# ----------------------------------------------------------------------------- prompts


@pytest.mark.parametrize(
    ("template", "variables"),
    [
        ("query_analyzer", {"query": "q", "memory": "m"}),
        (
            "planner",
            {"query": "q", "analysis": "a", "memory": "m", "tools": "t", "feedback": "f"},
        ),
        ("answer_generator", {"context": "c", "tools": "t", "memory": "m", "query": "q"}),
        ("critic", {"query": "q", "context": "c", "tools": "t", "answer": "a"}),
        ("conversation_summary", {"summary": "s", "transcript": "t"}),
        ("user_context", {"question": "q", "answer": "a"}),
    ],
)
def test_prompt_templates_render(template: str, variables: dict[str, str]) -> None:
    from app.prompts import registry

    rendered = registry.render(template, **variables)

    assert rendered.strip()
    assert "$" not in rendered
