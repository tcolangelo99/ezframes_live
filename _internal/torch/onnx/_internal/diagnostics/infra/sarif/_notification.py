# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import List, Literal, Optional

from torch.onnx._internal.diagnostics.infra.sarif import (
    _exception,
    _location,
    _message,
    _property_bag,
    _reporting_descriptor_reference,
)


@dataclasses.dataclass
class Notification(object):
    """Describes a condition relevant to the tool itself, as opposed to being relevant to a target being analyzed by the tool."""

    message: _message.Message = dataclasses.field(
        metadata={"schema_property_name": "message"}
    )
    associated_rule: Optional[
        _reporting_descriptor_reference.ReportingDescriptorReference
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "associatedRule"}
    )
    descriptor: Optional[
        _reporting_descriptor_reference.ReportingDescriptorReference
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "descriptor"})
    exception: Optional[_exception.Exception] = dataclasses.field(
        default=None, metadata={"schema_property_name": "exception"}
    )
    level: Literal["none", "note", "warning", "error"] = dataclasses.field(
        default="warning", metadata={"schema_property_name": "level"}
    )
    locations: Optional[List[_location.Location]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "locations"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    thread_id: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "threadId"}
    )
    time_utc: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "timeUtc"}
    )


# flake8: noqa
