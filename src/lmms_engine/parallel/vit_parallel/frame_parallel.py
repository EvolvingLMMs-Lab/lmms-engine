"""ViT frame parallelism: a single ``wrap_vit_forward`` that wires the
dispatch / gather around any vision-tower forward.

The actual dispatch and gather logic is passed in by the caller — different
ViT architectures may have slightly different input/output layouts, so we
keep this file as pure plumbing.
"""

from typing import Any, Callable


def wrap_vit_forward(
    input_dispatch: Callable[..., Any],
    orig_forward: Callable[..., Any],
    output_dispatch: Callable[..., Any],
) -> Callable[..., Any]:
    """Compose a frame-parallel forward from three callables.

    Args:
        input_dispatch: receives the original forward's ``(*args, **kwargs)``
            and returns ``(new_args, new_kwargs, ctx)``. ``ctx`` is an opaque
            object the caller may use to pass state (e.g. the LPT plan) into
            ``output_dispatch``.
        orig_forward: the upstream ViT forward, called as
            ``orig_forward(*new_args, **new_kwargs)``.
        output_dispatch: receives ``(forward_output, ctx)`` and returns the
            value the wrapped forward should return.

    Returns:
        A callable with the same signature as ``orig_forward`` that performs
        ``input_dispatch -> orig_forward -> output_dispatch``.
    """

    def wrapped(*args, **kwargs):
        new_args, new_kwargs, ctx = input_dispatch(*args, **kwargs)
        out = orig_forward(*new_args, **new_kwargs)
        return output_dispatch(out, ctx)

    return wrapped
