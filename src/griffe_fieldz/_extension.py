"""Griffe Fieldz extension."""

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import fieldz
from fieldz._repr import display_as_type
from griffe import (
    Class,
    Docstring,
    DocstringAttribute,
    DocstringParameter,
    DocstringSection,
    DocstringSectionAttributes,
    DocstringSectionParameters,
    Extension,
    Object,
    ObjectNode,
    dynamic_import,
    get_logger,
    parse_docstring_annotation,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from griffe import Expr, Inspector, Visitor

logger = get_logger(__name__)


class FieldzExtension(Extension):
    """Griffe extension that injects field information for dataclass-likes."""

    def __init__(
        self,
        object_paths: list[str] | None = None,
        include_private: bool = False,
        include_inherited: bool = False,
        **kwargs: Any,
    ) -> None:
        self.object_paths = object_paths
        self._kwargs = kwargs
        self.include_private = include_private
        self.include_inherited = include_inherited

    def on_class_instance(
        self,
        *,
        node: ast.AST | ObjectNode,
        cls: Class,
        agent: Visitor | Inspector,
        **kwargs: Any,
    ) -> None:
        if isinstance(node, ObjectNode):
            return  # skip runtime objects

        if self.object_paths and cls.path not in self.object_paths:
            return  # skip objects that were not selected

        # import object to get its evaluated docstring
        try:
            runtime_obj = dynamic_import(cls.path)
        except ImportError:
            logger.debug(f"Could not get dynamic docstring for {cls.path}")
            return

        try:
            fieldz.get_adapter(runtime_obj)
        except TypeError:
            return
        self._inject_fields(node, cls, runtime_obj, agent)

    # ------------------------------

    def _inject_fields(
        self, node: ast.AST, obj: Object, runtime_obj: Any, agent: Visitor | Inspector
    ) -> None:
        # update the object instance with the evaluated docstring
        docstring = inspect.cleandoc(getattr(runtime_obj, "__doc__", "") or "")
        if not obj.docstring:
            obj.docstring = Docstring(docstring, parent=obj)
        sections = obj.docstring.parsed

        # collect field info
        fields = fieldz.fields(runtime_obj)
        if not self.include_inherited:
            annotations = getattr(runtime_obj, "__annotations__", {})
            fields = tuple(f for f in fields if f.name in annotations)

        with _agent_temp_members(agent):
            agent.visit(node)
            for subnode in node.body:
                if not isinstance(subnode, ast.Assign | ast.AnnAssign):
                    continue
                agent.visit(subnode)

            params, attrs = _fields_to_params(
                fields, agent, obj.docstring, self.include_private
            )

        # merge/add field info to docstring
        if params:
            for x in sections:
                if isinstance(x, DocstringSectionParameters):
                    _merge(x, params)
                    break
            else:
                sections.insert(1, DocstringSectionParameters(params))
        if attrs:
            for x in sections:
                if isinstance(x, DocstringSectionAttributes):
                    _merge(x, params)
                    break
            else:
                sections.append(DocstringSectionAttributes(attrs))


def _to_annotation(type_: Any, docstring: Docstring) -> str | Expr | None:
    """Create griffe annotation for a type."""
    if type_:
        return parse_docstring_annotation(
            display_as_type(type_, modern_union=True), docstring
        )
    return None


def _default_repr(field: fieldz.Field) -> str | None:
    """Return a repr for a field default."""
    if field.default is not field.MISSING:
        return repr(field.default)
    if field.default_factory is not field.MISSING:
        return repr(field.default_factory())
    return None


@contextmanager
def _agent_temp_members(agent: Visitor | Inspector) -> Iterator[None]:
    """Use a temporary members dictionary in the scope of the context."""
    prev_members = agent.current.members
    try:
        agent.current.members = {}
        yield
    finally:
        agent.current.members = prev_members


def _agent_docstring(
    agent: Visitor | Inspector, name: str
) -> tuple[str | None, Expr | str | None]:
    """Get the docstring from griffe's attribute handling."""
    member = agent.current.members.get(name)
    if not member:
        return None, None
    docstring = member.docstring.value if member.docstring else None
    annotation = member.annotation
    return docstring, annotation


def _fields_to_params(
    fields: Iterable[fieldz.Field],
    agent: Visitor | Inspector,
    docstring: Docstring,
    include_private: bool = False,
) -> tuple[list[DocstringParameter], list[DocstringAttribute]]:
    """Get all docstring attributes and parameters for fields."""
    params: list[DocstringParameter] = []
    attrs: list[DocstringAttribute] = []
    for field in fields:
        agent_doc, agent_ann = _agent_docstring(agent, field.name)
        description = (
            field.description
            or field.metadata.get("description", "")
            or agent_doc
            or ""
        )
        annotation = agent_ann or _to_annotation(field.type, docstring)
        kwargs: dict = {
            "name": field.name,
            "annotation": annotation,
            "description": textwrap.dedent(description).strip(),
            "value": _default_repr(field),
        }
        if field.init:
            params.append(DocstringParameter(**kwargs))
        elif include_private or not field.name.startswith("_"):
            attrs.append(DocstringAttribute(**kwargs))

    return params, attrs


def _merge(
    section: DocstringSection, field_params: Sequence[DocstringParameter]
) -> None:
    """Update DocstringSection with field params (if missing)."""
    existing_names = {x.name for x in section.value}
    for param in field_params:
        if param.name not in existing_names:
            section.value.append(param)
