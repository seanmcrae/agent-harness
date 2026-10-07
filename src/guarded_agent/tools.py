"""Tool definitions: schema derivation from type hints, validation, timeouts, side effects."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from guarded_agent.types import ToolSpec


class SideEffect(StrEnum):
    READ = "read"
    WRITE = "write"


class ToolError(Exception):
    """Raised by a tool to report a failure the model should see and may recover from."""


class ToolArgumentError(ToolError):
    pass


class ToolTimeoutError(ToolError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    side_effect: SideEffect = SideEffect.READ
    idempotent: bool = True
    timeout_s: float = 10.0
    model_param: str | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, _clean_schema(self.args_model))

    @property
    def requires_approval(self) -> bool:
        return self.side_effect is SideEffect.WRITE

    def validate(self, arguments: Mapping[str, Any]) -> BaseModel:
        try:
            return self.args_model.model_validate(dict(arguments))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolArgumentError(f"Invalid arguments for {self.name}: {problems}") from exc

    def invoke(self, args: BaseModel, timeout_s: float | None = None) -> Any:
        """Run the tool body in a worker thread, enforcing a timeout.

        A timed-out thread cannot be killed in CPython; it is abandoned and its result ignored.
        Tools with external side effects should enforce their own I/O timeouts as well.
        """
        limit = self.timeout_s if timeout_s is None else min(timeout_s, self.timeout_s)
        if self.model_param is not None:
            kwargs: dict[str, Any] = {self.model_param: args}
        else:
            kwargs = {name: getattr(args, name) for name in type(args).model_fields}
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{self.name}")
        try:
            return executor.submit(self.fn, **kwargs).result(timeout=max(limit, 0.0))
        except FutureTimeout as exc:
            raise ToolTimeoutError(f"{self.name} timed out after {limit:.2f}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    side_effect: SideEffect = SideEffect.READ,
    idempotent: bool | None = None,
    timeout_s: float = 10.0,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effect: SideEffect = SideEffect.READ,
    idempotent: bool | None = None,
    timeout_s: float = 10.0,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a typed function into a Tool.

    The JSON schema comes from the signature: each parameter becomes a field (use
    ``Annotated[T, Field(...)]`` for constraints and descriptions), or a single parameter typed
    as a pydantic model is used as the whole argument schema. The description defaults to the
    docstring's first paragraph. Write tools default to non-idempotent.
    """

    def build(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        doc = description or (inspect.getdoc(func) or "").split("\n\n")[0].strip()
        if not doc:
            raise ValueError(f"tool {tool_name!r} needs a docstring or description")
        args_model, model_param = derive_args_model(func, tool_name)
        return Tool(
            name=tool_name,
            description=" ".join(doc.split()),
            fn=func,
            args_model=args_model,
            side_effect=side_effect,
            idempotent=(side_effect is SideEffect.READ) if idempotent is None else idempotent,
            timeout_s=timeout_s,
            model_param=model_param,
        )

    return build(fn) if fn is not None else build


def derive_args_model(
    func: Callable[..., Any], tool_name: str
) -> tuple[type[BaseModel], str | None]:
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    params = list(signature.parameters.values())
    for param in params:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"tool {tool_name!r}: *args/**kwargs are not supported")
        if param.name not in hints:
            raise TypeError(f"tool {tool_name!r}: parameter {param.name!r} needs a type hint")

    if len(params) == 1:
        hint = hints[params[0].name]
        if inspect.isclass(hint) and issubclass(hint, BaseModel):
            return hint, params[0].name

    fields: dict[str, Any] = {
        p.name: (hints[p.name], ... if p.default is p.empty else p.default) for p in params
    }
    model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Args"
    model: type[BaseModel] = create_model(
        model_name, __config__=ConfigDict(extra="forbid"), **fields
    )
    return model, None


def _clean_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema without pydantic's auto-generated titles, which only cost tokens."""

    def strip(node: Any, in_properties: bool = False) -> Any:
        if isinstance(node, dict):
            return {
                key: strip(value, in_properties=key == "properties" and not in_properties)
                for key, value in node.items()
                if in_properties or not (key == "title" and isinstance(value, str))
            }
        if isinstance(node, list):
            return [strip(item) for item in node]
        return node

    schema: dict[str, Any] = strip(model.model_json_schema())
    return schema


def serialize_result(value: Any) -> str:
    """Render a tool's return value as the text the model sees."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, default=str)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> Tool:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name {t.name!r}")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self, allowed: Collection[str] | None = None) -> tuple[ToolSpec, ...]:
        return tuple(t.spec for t in self._tools.values() if allowed is None or t.name in allowed)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
