import time
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from guarded_agent.tools import (
    SideEffect,
    ToolArgumentError,
    ToolRegistry,
    ToolTimeoutError,
    serialize_result,
    tool,
)


@tool
def get_weather(
    city: Annotated[str, Field(description="City name")],
    days: Annotated[int, Field(ge=1, le=7)] = 1,
    units: str = "metric",
) -> dict[str, object]:
    """Get the weather forecast for a city.

    This second paragraph is not part of the description.
    """
    return {"city": city, "days": days, "units": units}


class TransferArgs(BaseModel):
    account_id: str
    amount: float = Field(gt=0)


@tool(side_effect=SideEffect.WRITE, timeout_s=2)
def transfer(args: TransferArgs) -> str:
    """Move money between accounts."""
    return f"moved {args.amount} to {args.account_id}"


def test_schema_derived_from_type_hints() -> None:
    spec = get_weather.spec
    assert spec.name == "get_weather"
    assert spec.description == "Get the weather forecast for a city."
    params = spec.parameters
    assert params["type"] == "object"
    assert params["required"] == ["city"]
    assert params["properties"]["city"] == {"type": "string", "description": "City name"}
    assert params["properties"]["days"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 7,
        "default": 1,
    }
    assert params["additionalProperties"] is False
    assert "title" not in params


def test_field_named_title_survives_schema_cleanup() -> None:
    @tool
    def create_ticket(title: str, body: str = "") -> str:
        """Open a ticket."""
        return title

    assert set(create_ticket.spec.parameters["properties"]) == {"title", "body"}


def test_pydantic_model_parameter_is_used_as_schema() -> None:
    assert transfer.model_param == "args"
    assert transfer.spec.parameters["properties"]["amount"]["exclusiveMinimum"] == 0
    assert (
        transfer.invoke(transfer.validate({"account_id": "A1", "amount": 5})) == "moved 5.0 to A1"
    )


def test_side_effect_defaults() -> None:
    assert get_weather.side_effect is SideEffect.READ
    assert get_weather.idempotent is True
    assert transfer.requires_approval is True
    assert transfer.idempotent is False


def test_validation_errors_are_readable() -> None:
    with pytest.raises(ToolArgumentError) as info:
        get_weather.validate({"city": "Oslo", "days": 9, "extra": 1})
    message = str(info.value)
    assert "days" in message
    assert "extra" in message


def test_invoke_passes_validated_values() -> None:
    args = get_weather.validate({"city": "Oslo", "days": "3"})
    assert get_weather.invoke(args) == {"city": "Oslo", "days": 3, "units": "metric"}


def test_invoke_enforces_timeout() -> None:
    @tool(timeout_s=0.05)
    def slow() -> str:
        """Sleep for a while."""
        time.sleep(0.5)
        return "late"

    with pytest.raises(ToolTimeoutError, match="timed out"):
        slow.invoke(slow.validate({}))


def test_tool_requires_docstring_and_hints() -> None:
    def undocumented(x: int) -> int:
        return x

    def untyped(x):  # type: ignore[no-untyped-def]
        """Has no hints."""
        return x

    with pytest.raises(ValueError, match="docstring"):
        tool(undocumented)
    with pytest.raises(TypeError, match="type hint"):
        tool(untyped)


def test_registry_rejects_duplicates_and_filters_specs() -> None:
    registry = ToolRegistry([get_weather, transfer])
    assert registry.names == ("get_weather", "transfer")
    assert [s.name for s in registry.specs(allowed={"transfer"})] == ["transfer"]
    assert "transfer" in registry
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(get_weather)


def test_serialize_result() -> None:
    assert serialize_result("plain") == "plain"
    assert serialize_result({"a": 1}) == '{"a": 1}'
    assert serialize_result(TransferArgs(account_id="A", amount=1)) == (
        '{"account_id":"A","amount":1.0}'
    )
