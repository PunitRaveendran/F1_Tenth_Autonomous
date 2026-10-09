def _patch_numba_coverage_compat():
    try:
        import coverage.types as coverage_types
    except Exception:
        return

    for name in ("Tracer", "TShouldTraceFn", "TShouldStartContextFn"):
        if not hasattr(coverage_types, name):
            setattr(coverage_types, name, object)


_patch_numba_coverage_compat()

from .f110_env import F110Env  # noqa: F401 This is needed to make easy imports work
