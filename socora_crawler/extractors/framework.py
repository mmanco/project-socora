from __future__ import annotations

import importlib
import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, Union

from scrapy.http import Response

logger = logging.getLogger(__name__)


def _load_callable(path: str) -> Callable:
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Callable spec '{path}' must be in 'module:attribute' format")
    module = importlib.import_module(module_name)
    target = getattr(module, attr)
    if not callable(target):
        raise TypeError(f"Object '{path}' is not callable")
    return target


def _call_with_supported_args(fn: Callable, *args) -> Any:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return fn()
    supported = []
    for arg_name, arg_value in zip(["response", "item", "context"], args):
        if any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name == arg_name for p in params):
            supported.append(arg_value)
        elif any(p.kind == p.VAR_POSITIONAL for p in params):
            supported.append(arg_value)
        else:
            continue
    return fn(*supported)


def dedupe_list(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


@dataclass
class ExtractorContext:
    content: List[Any]
    links: List[str]
    item: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    requests: List[Any] = field(default_factory=list)
    item_updates: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor:
    name: Optional[str] = None

    def __init__(
        self,
        *,
        url_patterns: Optional[Sequence[str]] = None,
        predicate: Optional[Union[str, Callable]] = None,
        **config: Any,
    ) -> None:
        self.config = config
        self._compiled_patterns = [re.compile(p) for p in url_patterns or []]
        if isinstance(predicate, str):
            predicate = _load_callable(predicate)
        self.predicate: Optional[Callable] = predicate
        self.spider = None
        self.logger = logger

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "BaseExtractor":
        return cls(**(config or {}))

    def bind(self, spider) -> None:
        self.spider = spider
        self.logger = getattr(spider, "logger", logger)

    def matches(self, response: Response, item: Dict[str, Any], context: ExtractorContext) -> bool:
        url = response.url
        matched = False
        if self._compiled_patterns:
            matched = any(p.search(url) for p in self._compiled_patterns)
        if self.predicate:
            try:
                if _call_with_supported_args(self.predicate, response, item, context):
                    return True
            except Exception as exc:
                self.logger.debug("Extractor predicate failed for %s: %s", getattr(self, 'name', self.__class__.__name__), exc)
        if matched:
            return True
        if not self._compiled_patterns and not self.predicate:
            return True
        return False

    def apply(self, response: Response, item: Dict[str, Any], context: ExtractorContext) -> None:
        raise NotImplementedError


_extractor_registry: Dict[str, Type[BaseExtractor]] = {}


def register_extractor(name: str) -> Callable[[Type[BaseExtractor]], Type[BaseExtractor]]:
    def decorator(cls: Type[BaseExtractor]) -> Type[BaseExtractor]:
        cls.name = name
        _extractor_registry[name] = cls
        return cls

    return decorator


def _resolve_class(spec: str) -> Type[BaseExtractor]:
    if spec in _extractor_registry:
        return _extractor_registry[spec]
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Extractor spec '{spec}' must be a registered name or 'module:Class' path")
    module = importlib.import_module(module_name)
    target = getattr(module, attr)
    if inspect.isclass(target) and issubclass(target, BaseExtractor):
        return target
    if callable(target):
        instance = target()
        if isinstance(instance, BaseExtractor):
            return instance.__class__
    raise TypeError(f"Object '{spec}' is not a BaseExtractor class")


def _instantiate(cls: Type[BaseExtractor], config: Optional[Dict[str, Any]]) -> BaseExtractor:
    if hasattr(cls, "from_config"):
        return cls.from_config(config)
    if config:
        return cls(**config)
    return cls()


def load_extractor(spec: Union[str, Dict[str, Any], BaseExtractor]) -> BaseExtractor:
    if isinstance(spec, BaseExtractor):
        return spec
    if isinstance(spec, dict):
        spec = dict(spec)
        name = spec.pop("name", None) or spec.pop("path", None) or spec.pop("class", None)
        if not name:
            raise ValueError("Extractor dict must include 'name' or 'path'")
        config = spec.pop("config", {})
        if spec:
            config = {**spec, **config}
        cls = _resolve_class(name)
        return _instantiate(cls, config)
    if isinstance(spec, str):
        cls = _resolve_class(spec.strip())
        return _instantiate(cls, None)
    raise TypeError(f"Unsupported extractor spec type: {type(spec)!r}")


def build_extractors(specs: Iterable[Any]) -> List[BaseExtractor]:
    instances: List[BaseExtractor] = []
    for spec in specs:
        try:
            extractor = load_extractor(spec)
        except Exception as exc:
            logger.warning("Failed to load extractor %r: %s", spec, exc)
            continue
        instances.append(extractor)
    return instances


__all__ = [
    "ExtractorContext",
    "BaseExtractor",
    "register_extractor",
    "build_extractors",
    "load_extractor",
    "dedupe_list",
]

