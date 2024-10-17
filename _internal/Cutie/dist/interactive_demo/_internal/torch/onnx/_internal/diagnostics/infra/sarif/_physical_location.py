# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Optional

from torch.onnx._internal.diagnostics.infra.sarif import (
    _address,
    _artifact_location,
    _property_bag,
    _region,
)


@dataclasses.dataclass
class PhysicalLocation(object):
    """A physical location relevant to a result. Specifies a reference to a programming artifact together with a range of bytes or characters within that artifact."""

    address: Optional[_address.Address] = dataclasses.field(
        default=None, metadata={"schema_property_name": "address"}
    )
    artifact_location: Optional[
        _artifact_location.ArtifactLocation
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "artifactLocation"}
    )
    context_region: Optional[_region.Region] = dataclasses.field(
        default=None, metadata={"schema_property_name": "contextRegion"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    region: Optional[_region.Region] = dataclasses.field(
        default=None, metadata={"schema_property_name": "region"}
    )


# flake8: noqa
