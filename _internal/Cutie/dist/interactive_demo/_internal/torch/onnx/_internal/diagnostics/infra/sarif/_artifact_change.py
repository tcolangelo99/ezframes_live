# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import List, Optional

from torch.onnx._internal.diagnostics.infra.sarif import (
    _artifact_location,
    _property_bag,
    _replacement,
)


@dataclasses.dataclass
class ArtifactChange(object):
    """A change to a single artifact."""

    artifact_location: _artifact_location.ArtifactLocation = dataclasses.field(
        metadata={"schema_property_name": "artifactLocation"}
    )
    replacements: List[_replacement.Replacement] = dataclasses.field(
        metadata={"schema_property_name": "replacements"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )


# flake8: noqa
