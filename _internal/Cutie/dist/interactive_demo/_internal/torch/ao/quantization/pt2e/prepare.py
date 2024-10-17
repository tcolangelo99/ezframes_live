# mypy: allow-untyped-defs
import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
    _insert_obs_or_fq,
    _save_state,
    _is_activation_post_process_node,
    _create_obs_or_fq_from_qspec,
)
from torch.fx import (
    GraphModule,
    Graph,
    Node,
)
from torch.fx.node import Argument

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
    EdgeOrNode,
    SharedQuantizationSpec,
    QuantizationSpecBase,
)
from torch.ao.quantization import ObserverOrFakeQuantize

# TODO: make pt2e folder private?
__all__ = [
    "prepare",
]


def _find_root_edge_or_node(edge_or_node: EdgeOrNode, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]) -> EdgeOrNode:
    """Find the root node for the sharing tree
    Args:
        edge_or_node: edge/node that we want to find the root
        shared_with_map: each edge/node points to the parent, the root node will points to itself

    Returns:
        root edge/node
    """
    parent = shared_with_map[edge_or_node]
    if parent == edge_or_node:
        return edge_or_node
    root = _find_root_edge_or_node(parent, shared_with_map)
    # path compression
    shared_with_map[edge_or_node] = root
    return root

def _union(parent: EdgeOrNode, child: EdgeOrNode, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]) -> None:
    """Merge the subtree for `child` with `parent`, the order is important here
    """
    root_parent = _find_root_edge_or_node(parent, shared_with_map)
    root_child = _find_root_edge_or_node(child, shared_with_map)
    # union the two trees by pointing the root of child to root of parent
    shared_with_map[root_child] = root_parent

def _update_shared_with(child: EdgeOrNode, qspec: QuantizationSpecBase, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]):
    """Update the `shared_with_map` based on the qspec, this applies the `SharedQuantizationSpec`
    configuration and established the relationship between `edge_or_node` with the edge/node that it
    is pointing to, we'll use this information in the end to get the group id
    """
    if isinstance(qspec, SharedQuantizationSpec):
        parent = qspec.edge_or_node
        # we point from edge_or_node to the node that it is sharing_with, e.g.
        # qspec for a = SharedQuantizationSpec(b) means `a` points to `b`
        _union(parent, child, shared_with_map)

def _unwrap_shared_qspec(
    qspec: QuantizationSpecBase,
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase],
    shared_with_map: Dict[EdgeOrNode, EdgeOrNode]
) -> QuantizationSpecBase:
    """Unwraps qspec to get the final root qspec (non SharedQuantizationSpec)
    if qspec is SharedQuantizationSpec
       (1). tries to find the root edge or node for the node that the qspec points to
       (2). recursively find the root qspec based on the qspec for the root node
    """
    if isinstance(qspec, SharedQuantizationSpec):
        sharing_with = qspec.edge_or_node
        root = _find_root_edge_or_node(sharing_with, shared_with_map)
        qspec = edge_or_node_to_qspec[root]
        return _unwrap_shared_qspec(qspec, edge_or_node_to_qspec, shared_with_map)
    return qspec

def _has_same_dtype(qspec_a: QuantizationSpecBase, qspec_b: QuantizationSpecBase):
    return (
        hasattr(qspec_a, "dtype") and
        hasattr(qspec_b, "dtype") and
        qspec_a.dtype == qspec_b.dtype
    )

def _has_same_is_dynamic(qspec_a: QuantizationSpecBase, qspec_b: QuantizationSpecBase):
    return (
        hasattr(qspec_a, "is_dynamic") and
        hasattr(qspec_b, "is_dynamic") and
        qspec_a.is_dynamic == qspec_b.is_dynamic
    )

def _get_edge_or_node_to_qspec(model: torch.fx.GraphModule) -> Dict[EdgeOrNode, QuantizationSpecBase]:
    """Get a map from EdgeOrNode to quantization spec based on annotations on the nodes
    """
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase] = {}
    for n in model.graph.nodes:
        if hasattr(n, "meta") and "quantization_annotation" in n.meta:
            qa = n.meta["quantization_annotation"]
            for input_to_n, qspec in qa.input_qspec_map.items():
                input_edge = (input_to_n, n)
                edge_or_node_to_qspec[input_edge] = qspec
            if qa.output_qspec is not None:
                output_node = n
                qspec = qa.output_qspec
                edge_or_node_to_qspec[output_node] = qspec
    return edge_or_node_to_qspec

def _union_input_edge_with(input_edge, input_edge_root_qspec, edge_or_node, edge_or_node_to_qspec, shared_with_map):
    """Union input edge with another edge or node, used in implicit sharing to point the current input
    edge to other user edges of the producer node, or the output of producer node since these are
    referring to the same Tensor
    """
    root_qspec = None
    if edge_or_node in edge_or_node_to_qspec:
        qspec = edge_or_node_to_qspec[edge_or_node]
        root_qspec = _unwrap_shared_qspec(qspec, edge_or_node_to_qspec, shared_with_map)
    # TODO: add assertions for types of root qspecs
    if (
        root_qspec is not None and
        _has_same_dtype(root_qspec, input_edge_root_qspec) and
        _has_same_is_dynamic(root_qspec, input_edge_root_qspec)
    ):
        # the input arg to the node should reuse the existing output observer for arg
        # since dtype is the same (we may want to extend this to be a more strict check
        # in the future)
        # so we point from `input_edge` to `arg` (output of the argument)
        _union(edge_or_node, input_edge, shared_with_map)


def _get_edge_or_node_to_group_id(edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase]) -> Dict[EdgeOrNode, int]:
    """Map from edge/node to the group ID, generated from quantization annotations,
    edge/node with the same group ID should use the same observer/fake_quant instance

    This is applying SharedQuantizationSpec configuration and map each edge/node to a group
    There is another implicit sharing that's built in the quantization, when we have the following:
       * op1 -> op2
       * output of op1: int8_qspec
       * (op1 -> op2) input edge: int8_qspec
    we'll assume sharing between the output of op1 and input of (op1 -> op2) since these are the same Tensor.

    Figuring out the correct group ID for all edge/node is a standard union find problem:
    https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/

    Args:
        edge_or_node_to_qspec: Dictionary from edge_or_node to the qspec, derived from annotations
    Returns:
        edge_or_node_to_group_id: Dictionary from edge_or_node to group_id (int), all edge or node that
        belongs to the same group should have the same id

    Example:
        op2 -> cat1 -> cat2
           op1 /        /
                     op3
        edge_or_node_to_qspec: {
            op1: int8_qspec,
            op2: int8_qspec,
            (op1, cat1): int8_qspc,
            (op2, cat1): SharedQuantizationSpec((op1, cat1)),
            cat1: SharedQuantizationSpec((op1, cat1)),
            (op3, cat2): int8_qspec,
            (cat1, cat2): SharedQuantizationSpec((op3, cat2)),
            cat2: SharedQuantizationSpec((op3, cat2)),
        }

        edge_or_node_to_group_id = _get_edge_or_node_to_group_id(edge_or_node_to_qspec)
        edge_or_node_to_group_id: {
            op1: 1,
            op2: 1,
            (op1, cat1): 1,
            (op2, cat1): 1,
            cat1: 1,
            (op3, cat2): 1,
            (cat1, cat2): 1,
            cat2: 1,
        }
        # everything are in the same group because (cat1) and (cat1, cat2) are implicitly shared, which
        # connects the two sharing group around cat1 and cat2 op due to transitive sharing
    """
    # means the observer of key should be shared with observer with value, by default it will
    # be shared with itself
    shared_with_map: Dict[EdgeOrNode, EdgeOrNode] = {k: k for k in edge_or_node_to_qspec.keys()}
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        if isinstance(edge_or_node, torch.fx.Node):
            output_node = edge_or_node
            _update_shared_with(output_node, qspec, shared_with_map)
        else:
            input_edge = edge_or_node
            input_edge_root_qspec = _unwrap_shared_qspec(qspec, edge_or_node_to_qspec, shared_with_map)

            assert isinstance(input_edge, tuple)
            arg, n = input_edge
            if n.meta["quantization_annotation"].allow_implicit_sharing:
                # NOTE: the order is important here, we first share with other users and then share with previous
                # output because the reverse order could cause circular dependency
                # e.g node1 -> node2
                #          \ -> node3
                # when processing (node1, node2), if we first point (node1, node2) to node1
                # Step 1. shared_map = {(node1, node2): node1}
                # Step 2. after that, we point the (node1, node2) to its other user (node1, node3) ,
                # which means shared_map = {(node1, node2): node1, node1: (node1, node3)}
                # because we will point the root of (node1, node2) (in this case node1) to the root of (node1, node3)
                # Step 3. and when we process (node1, node3), it can try to point to node1 as well, then we'll
                # have a circular dependency
                # the following order works around this issue, but this does not allow arbitrary configuration
                # of sharing so it might break in a different case in the future, when it breaks
                # quantizer writer can check the notes here to debug the issue

                # sharing with other users of the producer node
                # (arg, user)
                if not isinstance(arg, Node) or not isinstance(n, Node):
                    raise Exception(f"Expected input_edge to have type Tuple[Node, Node], but got: {arg, n}")  # noqa: TRY002
                for user in arg.users:
                    if user is n:
                        continue
                    arg_to_user_edge = (arg, user)
                    _union_input_edge_with(
                        input_edge,
                        input_edge_root_qspec,
                        arg_to_user_edge,
                        edge_or_node_to_qspec,
                        shared_with_map
                    )

                # sharing with output of producer node
                _union_input_edge_with(input_edge, input_edge_root_qspec, arg, edge_or_node_to_qspec, shared_with_map)

            _update_shared_with(input_edge, qspec, shared_with_map)

    # now that we get the sharing relations between all edges and nodes, we can assingn group ids
    cur_group_id = 0
    edge_or_node_to_group_id: Dict[EdgeOrNode, int] = {}
    for edge_or_node in shared_with_map.keys():
        root = _find_root_edge_or_node(edge_or_node, shared_with_map)
        if root not in edge_or_node_to_group_id:
            edge_or_node_to_group_id[root] = cur_group_id
            cur_group_id += 1
        edge_or_node_to_group_id[edge_or_node] = edge_or_node_to_group_id[root]

    return edge_or_node_to_group_id

def _get_obs_or_fq_map(
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase],
    is_qat: bool
) -> Dict[EdgeOrNode, ObserverOrFakeQuantize]:
    """Generates the EdgeOrNode to observer/fake_quant instances
    Makes sure that for EdgeOrNode that has the same group_id should have the same observer or fake quant
    instances
    """
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    group_id_to_obs_or_fq: Dict[int, ObserverOrFakeQuantize] = {}
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        group_id = edge_or_node_to_group_id[edge_or_node]
        if group_id not in group_id_to_obs_or_fq:
            # TODO: maybe edge_or_node_to_qspec should be edge_or_node_to_root_qspec, this will simplify
            # the implementation for _create_obs_or_fq_from_qspec
            group_id_to_obs_or_fq[group_id] = _create_obs_or_fq_from_qspec(qspec, obs_or_fq_map, is_qat)
        obs_or_fq_map[edge_or_node] = group_id_to_obs_or_fq[group_id]
    return obs_or_fq_map

def _maybe_insert_input_observer_for_arg_or_kwarg(
    node: Union[Node, Any],
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> Argument:
    """
    Given a `node` and an `arg`, inserts an input observer between
    `node` and `arg` if necessary.
    """
    # for ops such as torch.cat([x0, x1]),
    # traverse through the list
    if isinstance(arg, (list, tuple)):
        new_arg_to_return = []
        for inner_arg in arg:
            new_inner_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
                node, inner_arg, qconfig, model, named_modules, obs_or_fq_map, is_qat,
            )
            new_arg_to_return.append(new_inner_arg)
        return type(arg)(new_arg_to_return)

    if not isinstance(arg, Node):
        return arg
    assert isinstance(arg, Node)
    # default (no observer)
    new_arg = arg

    # find the original `arg` node to the current node, skipping inserted observer/fake_quant nodes
    original_arg = arg
    while _is_activation_post_process_node(original_arg, named_modules):
        original_arg = original_arg.args[0]  # type: ignore[assignment]
    assert isinstance(original_arg, Node), f"expect original argument to be a Node, but got: {type(original_arg)}"

    input_edge = (original_arg, node)
    if input_edge not in obs_or_fq_map:
        return new_arg
    # input_edge needs to be observed
    input_edge_obs_or_fq = obs_or_fq_map[input_edge]
    if input_edge_obs_or_fq is None:
        return new_arg

    arg_as_output_obs_or_fq = obs_or_fq_map.get(original_arg, None)
    # the arg is observed as the output and is using the same instance as the input_edge
    # we'll reuse the inserted observer/fake_quant
    if arg_as_output_obs_or_fq is not None and id(arg_as_output_obs_or_fq) == id(input_edge_obs_or_fq):
        return new_arg

    # otherwise, we'll insert a new observer/fake_quant node

    existing_obs_node = None
    # skip inserting new observers if the same observer instance is inserted before for another user
    # Example:
    # conv1 -> obs1 -> existing_obs -> conv2
    #             \ -> conv3
    #
    # instead of inserting new observers we will have:
    # conv1 -> obs1 -> existing_obs -> conv2
    #                            \ -> conv3
    for maybe_obs_node in arg.users.keys():
        if not _is_activation_post_process_node(maybe_obs_node, named_modules):
            continue
        maybe_obs_mod = named_modules[maybe_obs_node.target]  # type: ignore[index]
        if id(maybe_obs_mod) == id(input_edge_obs_or_fq):
            return maybe_obs_node

    new_arg = _insert_obs_or_fq(arg, input_edge_obs_or_fq, model, named_modules, model.graph)
    return new_arg

def _maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node

    """
    # Look through every input arg.  If that arg's target dtype does not
    # match the current node's target dtype, insert an observer.
    new_args = []
    # map from old arg to new arg, used for updating the numeric debug handle map
    remap = {}
    for arg in node.args:
        new_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node, arg, qconfig, model, named_modules, obs_or_fq_map, is_qat,
        )
        new_args.append(new_arg)
        remap[arg] = new_arg

    if "numeric_debug_handle" in node.meta:

        def remap_fn(x):
            return remap.get(x, x)

        numeric_debug_handle = node.meta["numeric_debug_handle"]
        node.meta["numeric_debug_handle"] = {remap_fn(k): v for k, v in numeric_debug_handle.items()}

    # Clone has a memory_format kwarg, zeros_like has a pin_memory kwarg, and
    # gelu has a has an approximate kwarg that persist in exported graph.
    # This is just a work around for these.
    assert (
        node.target == torch.ops.aten.clone.default or
        node.target == torch.ops.aten.zeros_like.default or
        node.target == torch.ops.aten.gelu.default or
        len(node.kwargs) == 0
    ), " expecting kwargs for aten op IR to be empty"

    # assign the new args to the node, inplace
    node.args = tuple(new_args)

def _maybe_insert_output_observer_for_node(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> Optional[Node]:
    if node in obs_or_fq_map:
        output_act_obs_or_fq = obs_or_fq_map[node]
        return _insert_obs_or_fq(node, output_act_obs_or_fq, model, named_modules, graph)
    return None

def _maybe_insert_input_and_output_observers_for_node(
    node: Node,
    model: torch.fx.GraphModule,
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
):
    this_node_quantization_annotation = node.meta["quantization_annotation"] if "quantization_annotation" in node.meta else None
    if this_node_quantization_annotation is None:
        return

    named_modules = dict(model.named_modules(remove_duplicate=False))
    _maybe_insert_input_observers_for_node(
        node,
        None,  # qconfig
        model,
        named_modules,
        obs_or_fq_map,
        is_qat,
    )

    output_is_a_tensor = "val" in node.meta and isinstance(node.meta["val"], FakeTensor)
    if not output_is_a_tensor:
        return

    # this returns the new observer node if it was needed
    maybe_output_obs_node = _maybe_insert_output_observer_for_node(
        node, model, named_modules, model.graph, obs_or_fq_map, is_qat)

    if maybe_output_obs_node is None:
        return
    # Update users of original node to use the output observer
    # instead. For example, change
    #
    #           next_node
    #          /
    #   cur_node -> obs
    #
    # to
    #
    #                 next_node
    #                 /
    #   cur_node -> obs
    #
    # We need to save orig users before updating uses because
    # the list of users will change as we update uses
    orig_users = list(node.users.keys())
    for user_node in orig_users:
        if user_node is maybe_output_obs_node:
            continue
        user_node.replace_input_with(node, maybe_output_obs_node)

def prepare(
    model: GraphModule,
    node_name_to_scope: Dict[str, Tuple[str, type]],
    is_qat: bool,
) -> GraphModule:
    # Since we are mutating the graph as we go, we iterate over the original
    # nodes before observer insertion, instead of model.graph.nodes.
    nodes_before_observation = list(model.graph.nodes)

    # At the high level we construct a map from EdgeOrNode to a observer_or_fake_quant instance
    # all edge/nodes that belongs to the same group will use the same instance
    # and when we insert observers we'll just query this map to get the correct observer_or_fake_quant
    # instance
    edge_or_node_to_qspec = _get_edge_or_node_to_qspec(model)
    edge_or_node_to_group_id = _get_edge_or_node_to_group_id(edge_or_node_to_qspec)
    obs_or_fq_map = _get_obs_or_fq_map(edge_or_node_to_group_id, edge_or_node_to_qspec, is_qat)

    for node in nodes_before_observation:
        # TODO: simplify logic for inserting observers
        _maybe_insert_input_and_output_observers_for_node(node, model, obs_or_fq_map, is_qat)

    model = GraphModule(model, model.graph)

    _save_state(
        model,
        {},  # node_name_to_qconfig
        node_name_to_scope,
        PrepareCustomConfig(),
        {},  # equalization_node_name_to_qconfig
        QConfigMapping(),
        is_qat,
        set()  # observed_node_names
    )
    return model
