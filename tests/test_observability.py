"""
Tests for the observability data classes.

Covers:
    - LLMUsage.add() correctly accumulates across multiple calls
    - LLMUsage.estimated_cost_usd computes from settings.yaml pricing
    - StageTimings.total_s sums all stage fields

These are unit tests for the data classes themselves. The integration
side - "does run_query() correctly, call .add() on every LLM path" - is
covered by the existing test_pipeline.py instances (which assert
result.llm_usage values after running stubbed pipelines).

Run with:
    pytest tests/test_observability.py -v
"""

from src.pipeline import LLMUsage, StageTimings
from src.llm.provider import LLMResponse


# ===========================================================================
# LLMUsage.add() - token accumulation
# ===========================================================================

class TestLLMUsageAdd:
    """LLMUsage.add() should accumulate input_tokens, output_tokens, and
    call_count from every LLMResponse it receives."""

    def test_starts_at_zero(self):
        # A fresh LLMUsage has no calls and no tokens
        usage = LLMUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.call_count == 0

    def test_single_call_accumulates(self):
        # 1 LLM call --> tokens and call_count reflect that 1 call
        usage = LLMUsage()
        response = LLMResponse(
            text="some output", input_tokens=100, output_tokens=50
        )
        usage.add(response)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.call_count == 1

    def test_multiple_calls_accumulate(self):
        # 3 calls --> totals are sums; call_count is 3.
        # This is the core scenario for the typical DATA path:
        # classify (Tier 2) + SQL gen + narration
        usage = LLMUsage()
        usage.add(LLMResponse(text="a", input_tokens=10, output_tokens=5))
        usage.add(LLMResponse(text="b", input_tokens=200, output_tokens=20))
        usage.add(LLMResponse(text="c", input_tokens=50, output_tokens=15))

        assert usage.input_tokens == 260
        assert usage.output_tokens == 40
        assert usage.call_count == 3


# ===========================================================================
# LLMUsage.estimated_cost_usd - cost calculation
# ===========================================================================

class TestLLMUsageEstimatedCost:
    """estimated_cost_usd reads pricing from settings.yaml and returns
    USD cost based on accumulated tokens. We pin pricing via monkeypatch
    so the test doesn't depend on whatever's in the live settings file."""

    def test_zero_tokens_zero_cost(self):
        # No calls made --> cost is exactly 0
        usage = LLMUsage()
        assert usage.estimated_cost_usd == 0.0

    def test_cost_matches_settings_pricing(self, monkeypatch):
        # 1M input tokens at $3/Mtok + 0.5M output at $15/Mtok = $3 + $7.50
        # = $10.50. Pinning pricing here so the test stays correct even
        # if settings.yaml is updated later on.
        from src.config import PricingConfig
        import src.pipeline as pipeline_module

        class FakeSettings:
            class llm:
                pricing = PricingConfig(
                    input_per_mtok_usd=3.00,
                    output_per_mtok_usd=15.00,
                )

        monkeypatch.setattr(
            pipeline_module, "get_settings", lambda: FakeSettings
        )

        usage = LLMUsage(
            input_tokens=1_000_000,
            output_tokens=500_000,
            call_count=1,
        )
        assert usage.estimated_cost_usd == 10.50


# ===========================================================================
# StageTimings.total_s - sum of stages
# ===========================================================================

class TestStageTimingsTotal:
    """total_s should be the sum of all 6 stage fields, regardless
    of which were populated (zero-valued stages simply contribute 0)."""

    def test_default_total_is_zero(self):
        # Fresh StageTimings --> all stages at 0 --> total 0
        timings = StageTimings()
        assert timings.total_s == 0.0

    def test_total_sums_all_stages(self):
        # Every field populated --> total is exact sum
        timings = StageTimings(
            classify_s=0.1,
            retrieval_s=0.2,
            sql_generation_s=1.5,
            validation_s=0.05,
            execution_s=0.3,
            narration_s=0.8,
        )
        # Direct sum: 0.1 + 0.2 + 1.5 + 0.05 + 0.3 + 0.8 = 2.95
        # Use approx because float addition isn't always bit-exact
        assert abs(timings.total_s - 2.95) < 1e-9

    def test_partial_stages_only_count_what_ran(self):
        # CANNOT_ANSWER path: only classify, retrieval, sql_gen ran.
        # total_s should reflect just those 3
        timings = StageTimings(
            classify_s=0.0,        # heuristic fast-exit
            retrieval_s=0.15,
            sql_generation_s=2.4,
            # validation, execution, narration left at default 0.0
        )
        assert abs(timings.total_s - 2.55) < 1e-9