from typing import Any, Dict, Set

import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
    
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch.nn as nn


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


_PREFERRED_CONV_ALGO: Any | None = None
_ALGO_RESOLUTION_DONE = False
_ALGO_KWARGS: Dict[str, Any] | None = None
_ALGO_CANDIDATES = ("MaskImplicitGemm", "ImplicitGemm")


def _resolve_preferred_conv_algo() -> Any | None:
    conv_algo_enum = getattr(getattr(spconv, "constants", None), "ConvAlgo", None)
    fallback_enum = getattr(spconv, "ConvAlgo", None)
    for enum_cls in filter(None, (conv_algo_enum, fallback_enum)):
        for candidate in _ALGO_CANDIDATES:
            if hasattr(enum_cls, candidate):
                return getattr(enum_cls, candidate)
    return None


def _get_algo_kwargs() -> Dict[str, Any]:
    global _PREFERRED_CONV_ALGO, _ALGO_RESOLUTION_DONE, _ALGO_KWARGS

    if _ALGO_KWARGS is not None:
        return _ALGO_KWARGS

    if not _ALGO_RESOLUTION_DONE:
        try:
            _PREFERRED_CONV_ALGO = _resolve_preferred_conv_algo()
        except Exception:
            _PREFERRED_CONV_ALGO = None
        _ALGO_RESOLUTION_DONE = True

    _ALGO_KWARGS = {"algo": _PREFERRED_CONV_ALGO} if _PREFERRED_CONV_ALGO is not None else {}
    return _ALGO_KWARGS


def make_sparse_conv(conv_cls: Any, *args, **kwargs):
    algo_kwargs = _get_algo_kwargs()
    merged_kwargs = dict(kwargs)
    if "algo" not in merged_kwargs and algo_kwargs:
        merged_kwargs.update(algo_kwargs)
    try:
        return conv_cls(*args, **merged_kwargs)
    except TypeError as err:
        if merged_kwargs.pop("algo", None) is None:
            raise
        try:
            return conv_cls(*args, **merged_kwargs)
        except TypeError:
            raise err
