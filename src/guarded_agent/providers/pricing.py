"""Reference list prices (USD per million tokens) used for cost accounting and budgets.

Vendor prices change; pass ``pricing=`` to a provider to override. Mock pricing is simulated so
that budgets and cost reports exercise real arithmetic in offline runs.
"""

from guarded_agent.types import ModelPricing

SIMULATED_PRICING = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)

KNOWN_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0),
    "gpt-4.1": ModelPricing(2.0, 8.0),
    "gpt-4.1-mini": ModelPricing(0.4, 1.6),
    "gpt-4o": ModelPricing(2.5, 10.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.6),
}

UNKNOWN_PRICING = ModelPricing(0.0, 0.0)


def pricing_for(model: str) -> ModelPricing:
    """Return list pricing for a model id, matching dated snapshots by prefix."""
    if model in KNOWN_PRICING:
        return KNOWN_PRICING[model]
    for known, pricing in sorted(KNOWN_PRICING.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(known):
            return pricing
    return UNKNOWN_PRICING
