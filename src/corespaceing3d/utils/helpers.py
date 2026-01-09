"""Public helper utilities for users of corespaceing3d."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["projection_correction"]


def _default_fit_params_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "data" / "products" / "fit_params_C_vs_N_SDR.yaml"
    return candidate if candidate.exists() else None


def _load_fit_params(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load fit parameters. Install with `pip install pyyaml`."
        ) from exc

    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}.")
    return data


def _extract_param(params: Mapping[str, Any], key: str) -> float | None:
    value = params.get(key)
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric, got {value!r}.") from exc


def _infer_inf_key(params: Mapping[str, Any]) -> str | None:
    for key in ("C_inf", "R_inf"):
        if key in params:
            return key
    for key in params:
        if key.endswith("_inf"):
            return key
    return None


def _maybe_scalar(value: np.ndarray) -> float | np.ndarray:
    if isinstance(value, np.ndarray) and value.shape == ():
        return float(value)
    return value


def projection_correction(
    N: float | list[float] | np.ndarray,
    SDR: float | list[float] | np.ndarray,
    *,
    params_path: str | Path | None = None,
    params: Mapping[str, Any] | None = None,
    C_inf: float | None = None,
    S0: float | None = None,
    beta: float | None = None,
    N0: float | None = None,
    apply_to: float | list[float] | np.ndarray | None = None,
    mode: str = "multiply",
    return_factor: bool = False,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """Compute the projection correction factor C(N, SDR) and optionally apply it.

    This uses the fitted relation:
        C(N, SDR) = C_inf * [1 - exp(-SDR / S0)] * (N / N0) ** beta

    By default, this function uses the fit parameters determined in Barnes et al. (2026).
    These are stored in the package data directory and loaded automatically. You can
    override these by providing a custom YAML file via ``params_path``, or by
    providing explicit parameter values.

    Parameters
    ----------
    N
        Sample size (scalar or array-like). Accepts float, list, or numpy array.
    SDR
        Spatial dynamic range (scalar or array-like).
    params_path
        Path to a YAML file with fit parameters (e.g.
        ``data/products/fit_params_C_vs_N_SDR.yaml``). The YAML can store
        either plain values (``C_inf: 1.9``) or dicts with a ``value`` field.
    params
        Mapping of fit parameters (same schema as the YAML). Useful when you
        already have parameters in memory.
    C_inf, S0, beta, N0
        Explicit fit parameters. These override values loaded from ``params``
        or ``params_path``.
    apply_to
        Optional values to correct. If provided, this function returns
        ``apply_to * C`` (for ``mode="multiply"``) or ``apply_to / C`` (for
        ``mode="divide"``). This is convenient for correcting projected
        measurements, for example mean nearest-neighbor lengths.
    mode
        Either ``"multiply"`` (default) or ``"divide"`` when ``apply_to`` is
        supplied.
    return_factor
        If True and ``apply_to`` is provided, return ``(corrected, C)``.

    Returns
    -------
    C or corrected
        ``C(N, SDR)`` if ``apply_to`` is None, otherwise the corrected values.
        If ``return_factor=True``, returns ``(corrected, C)``.
    """
    if params is not None and not isinstance(params, Mapping):
        raise TypeError("params must be a mapping of parameter names to values.")

    params_data: Mapping[str, Any] | None = params
    if params_data is None:
        if params_path is None:
            params_path = _default_fit_params_path()
        if params_path is not None:
            path = Path(params_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Fit-parameter file not found: {path}")
            params_data = _load_fit_params(path)

    if C_inf is None and params_data is not None:
        inf_key = _infer_inf_key(params_data)
        if inf_key is not None:
            C_inf = _extract_param(params_data, inf_key)
    if S0 is None and params_data is not None:
        S0 = _extract_param(params_data, "S0")
    if beta is None and params_data is not None:
        beta = _extract_param(params_data, "beta")
    if N0 is None:
        if params_data is not None:
            N0 = _extract_param(params_data, "N0")
        if N0 is None:
            N0 = 100.0

    missing = [name for name, val in (("C_inf", C_inf), ("S0", S0), ("beta", beta)) if val is None]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Missing fit parameters: {missing_str}. "
            "Provide params_path, params, or explicit parameter values."
        )

    n_arr = np.asarray(N, dtype=float)
    sdr_arr = np.asarray(SDR, dtype=float)
    C = float(C_inf) * (1.0 - np.exp(-sdr_arr / float(S0))) * (n_arr / float(N0)) ** float(beta)
    C = _maybe_scalar(C)

    if apply_to is None:
        return C

    values = np.asarray(apply_to, dtype=float)
    if mode == "multiply":
        corrected = values * C
    elif mode == "divide":
        corrected = values / C
    else:
        raise ValueError('mode must be "multiply" or "divide".')

    corrected = _maybe_scalar(corrected)
    if return_factor:
        return corrected, C
    return corrected
