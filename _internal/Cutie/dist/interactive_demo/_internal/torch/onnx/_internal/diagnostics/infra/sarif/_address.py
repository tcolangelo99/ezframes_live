# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Optional

from torch.onnx._internal.diagnostics.infra.sarif import _property_bag


@dataclasses.dataclass
class Address(object):
    """A physical or virtual address, or a range of addresses, in an 'addressable region' (memory or a binary file)."""

    absolute_address: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "absoluteAddress"}
    )
    fully_qualified_name: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "fullyQualifiedName"}
    )
    index: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "index"}
    )
    kind: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "kind"}
    )
    length: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "length"}
    )
    name: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "name"}
    )
    offset_from_parent: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "offsetFromParent"}
    )
    parent_index: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "parentIndex"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    relative_address: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "relativeAddress"}
    )


# flake8: noqa
