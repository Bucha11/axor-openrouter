"""Shared fixtures for axor-openrouter tests."""
from __future__ import annotations

import pytest

from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextFragment, ContextView, LineageSummary
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import (
    ChildMode, CompressionMode, ContextMode, ExecutionPolicy,
    ExportMode, TaskComplexity, ToolPolicy,
)
from axor_openrouter.cascade.tiers import DEFAULT_TIERS, TierMapper
from axor_openrouter.routing.provider_prefs import ProviderPrefs
from axor_openrouter.transport import OpenRouterTransport
from axor_openrouter.executor import OpenRouterExecutor


@pytest.fixture()
def lineage():
    return LineageSummary(
        node_id="node_test",
        parent_id=None,
        depth=0,
        ancestry_ids=[],
        inherited_restrictions=[],
    )


@pytest.fixture()
def policy():
    return ExecutionPolicy(
        name="test",
        derived_from=TaskComplexity.FOCUSED,
        context_mode=ContextMode.MINIMAL,
        compression_mode=CompressionMode.BALANCED,
        child_mode=ChildMode.DENIED,
        max_child_depth=0,
        tool_policy=ToolPolicy(allow_read=True),
        export_mode=ExportMode.SUMMARY,
    )


def _build_envelope(lineage, policy, task="say hello", depth=0, allowed_tools=None, **kwargs):
    ctx = ContextView(
        node_id=lineage.node_id,
        working_summary="",
        visible_fragments=[],
        active_constraints=[],
        lineage=lineage,
        token_count=0,
        compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset(allowed_tools or ["read"]),
        allow_children=False,
        allow_nested_children=False,
        allow_context_expansion=False,
        allow_export=True,
        allow_mutation=False,
        max_child_depth=0,
    )
    export = ExportContract(
        mode=policy.export_mode,
        allowed_fields=frozenset(["output"]),
        max_export_tokens=1024,
    )
    return ExecutionEnvelope(
        node_id=lineage.node_id,
        task=task,
        context=ctx,
        policy=policy,
        capabilities=caps,
        export_contract=export,
        lineage=lineage,
        cancel_token=make_token(),
        depth=depth,
        **kwargs,
    )


@pytest.fixture()
def envelope(lineage, policy):
    return _build_envelope(lineage, policy)


@pytest.fixture()
def envelope_with_skill(lineage, policy):
    ctx = ContextView(
        node_id=lineage.node_id,
        working_summary="",
        visible_fragments=[
            ContextFragment(
                kind="skill",
                content="Be helpful and concise.",
                token_estimate=5,
                source="CLAUDE.md",
            )
        ],
        active_constraints=[],
        lineage=lineage,
        token_count=5,
        compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset(["read"]),
        allow_children=False,
        allow_nested_children=False,
        allow_context_expansion=False,
        allow_export=True,
        allow_mutation=False,
        max_child_depth=0,
    )
    export = ExportContract(
        mode=policy.export_mode,
        allowed_fields=frozenset(["output"]),
        max_export_tokens=1024,
    )
    return ExecutionEnvelope(
        node_id=lineage.node_id,
        task="say hello",
        context=ctx,
        policy=policy,
        capabilities=caps,
        export_contract=export,
        lineage=lineage,
        cancel_token=make_token(),
    )


@pytest.fixture()
def executor():
    return OpenRouterExecutor(
        api_key="sk-or-test",
        model="anthropic/claude-sonnet-4-6",
        transport=OpenRouterTransport(),
        tier_mapper=TierMapper(DEFAULT_TIERS),
    )
