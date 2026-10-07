"""Utility module for handling pytrees."""

import jax


def promote_structure(pytree, *others):
    """Promote pytrees to a common prefix structure.

    Args:
        pytree: The prefix pytree against which all `others` are promoted.
        *others: All other objects. Other pytrees that already have `pytree` as their prefix are unaltered. Any other object, like scalars or arrays, are replicated along the structure of the given prefix `pytree`.

    Returns:
        A tuple of the promoted `others`.
    """
    promoted_others = []
    for other in others:
        if is_prefix(pytree, other):
            promoted_others.append(other) # no promotion needed
        else:
            promoted_others.append( # repeat other across tree structure
                jax.tree.map(lambda _: other, pytree))  # noqa: B023
    if len(others) == 1:
        return promoted_others[0] # single input -> single output
    return tuple(promoted_others)

def is_prefix(pytree, other):
    """Check whether a given `pytree` is a prefix tree of another object.

    Only if it is, invoking `jax.tree.map(fn, pytree, other)` won't fail on their structure.

    Args:
        pytree: A potential prefix tree.
        other: Another object.

    Returns:
        True iff `pytree` is a prefix tree of `other`.
    """
    treedef = jax.tree.structure(pytree)
    try:
        treedef.flatten_up_to(other)
    except ValueError:
        return False
    return True
