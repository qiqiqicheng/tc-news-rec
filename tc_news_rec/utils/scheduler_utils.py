import inspect
from functools import partial


def filter_scheduler_args(scheduler_partial, allowed_extra_args=None):
    """
    Filter arguments for a partial scheduler object, keeping only those
    accepted by the scheduler's constructor.
    """
    if not isinstance(scheduler_partial, partial):
        return {}

    scheduler_cls = scheduler_partial.func
    sig = inspect.signature(scheduler_cls)

    # Get all parameters that the scheduler accepts
    valid_params = set(sig.parameters.keys())

    # Add any extra allowed args specific to our training loop (like total_steps)
    if allowed_extra_args:
        valid_params.update(allowed_extra_args)

    return valid_params
