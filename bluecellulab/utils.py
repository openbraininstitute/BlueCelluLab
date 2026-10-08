"""Utility functions used within BlueCellulab."""

from __future__ import annotations
import contextlib
import io
import json
import multiprocessing
from multiprocessing import pool

import numpy as np


_MISSING_SETTING = object()


@contextlib.contextmanager
def efel_settings(settings: dict | None = None):
    """Temporarily apply eFEL settings and restore the previous values.

    eFEL settings are process-global, so leaving them modified makes later
    feature extractions depend on execution order. This context manager
    snapshots every setting it changes and restores it on exit, including when
    the body raises.

    Args:
        settings: Mapping of eFEL setting names to values. ``None`` or an empty
            mapping applies nothing and still restores cleanly.
    """
    import efel

    settings = settings or {}
    current = efel.get_settings()
    previous = {
        key: getattr(current, key, _MISSING_SETTING) for key in settings
    }
    try:
        for key, value in settings.items():
            efel.set_setting(key, value)
        yield
    finally:
        for key, previous_value in previous.items():
            if previous_value is _MISSING_SETTING:
                if hasattr(current, key):
                    delattr(current, key)
            else:
                efel.set_setting(key, previous_value)


def run_once(func):
    """A decorator to ensure a function is only called once."""

    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class CaptureOutput(list):
    def __enter__(self):
        self._stringio = io.StringIO()
        self._redirect_stdout = contextlib.redirect_stdout(self._stringio)
        self._redirect_stdout.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._redirect_stdout.__exit__(exc_type, exc_val, exc_tb)
        self.extend(self._stringio.getvalue().splitlines())


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process."""

    # pylint: disable=dangerous-default-value

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NestedPool(pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool."""

    Process = NoDaemonProcess
