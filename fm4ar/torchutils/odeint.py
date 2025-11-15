import inspect
import torch
import time

from torchdiffeq import odeint 
from fm4ar.utils.nfe import NFEProfiler

class ODEFuncWrapper:
    """
    Wrap an ODE RHS function to automatically increment NFE counters
    and log per-batch events into the profiler.
    """

    def __init__(self, fn, profiler: NFEProfiler):
        self.fn = fn
        self.profiler = profiler
        self.key = self._infer_key_from_stack(fn)

    def _infer_key_from_stack(self, fn) -> str:
        """Infer the calling method name from the call stack."""
        skip = {"<lambda>", "odeint", "odeint_nfe", "__call__"}
        for frame_info in inspect.stack():
            name = frame_info.function
            if name not in skip and not name.startswith("_"):
                return name
        # fallback
        fn_name = getattr(fn, "__name__", None)
        if fn_name and fn_name not in skip:
            return fn_name
        return "unknown_ode_fn"

    def __call__(self, t, y):
        self.profiler.nfe.tick(self.key)
        t0 = time.time()
        result = self.fn(t, y)
        duration = time.time() - t0
        self.profiler.log_event(
            step=self.profiler.nfe[self.key],
            t=float(t) if not isinstance(t, torch.Tensor) else float(t.item()),
            key=self.key,
            duration=duration
        )
        return result

def odeint_nfe(func, y0, t, profiler: NFEProfiler, **odeint_kwargs):
    """
    ODE solver wrapper that tracks NFEs per batch using profiler.
    """
    # reset profiler for the current batch if batch size is provided
    profiler.reset()
    wrapped_func = ODEFuncWrapper(func, profiler)
    result = odeint(wrapped_func, y0, t, **odeint_kwargs)

    # do not finalize here; user may still want to run multiple batches before finalize
    return result