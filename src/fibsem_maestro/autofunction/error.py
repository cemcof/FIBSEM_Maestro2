# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


class AutofunctionError(Exception):
    """
    Exception raised for errors related to autofunction usage.
    """


class AutofunctionNotConvergingError(AutofunctionError):
    """
    Raised when too many consecutive sweeps found their best value at the edge of the swept region.
    """
