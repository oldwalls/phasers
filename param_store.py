import json, pathlib, copy
from typing import Dict, Any

class ParamStore:
    """
    Lightweight disk-backed store for Michael simulation parameters.

    Example
    -------
    store = ParamStore("michael_params.json")
    store.add("balanced_v1", dict(
        temp               = 0.70,
        top_p              = 0.85,
        top_k              = 40,
        repetition_penalty = 1.15,
        max_forward_tokens = 256,
        max_reply_sentences= 4,
        top_n              = 15,
        weight             = 0.30,
        tau                = 0.35,
        lam                = 0.95,
        n_sieve            = 1
    ))
    params = store.get("balanced_v1")
    store.update("balanced_v1", temp=0.72)   # tweak one field
    store.delete("old_variant")
    store.save()     # explicit flush (auto-saves on __exit__)
    """

    REQUIRED_KEYS = {
        "temp","top_p","top_k","repetition_penalty",
        "max_forward_tokens","max_reply_sentences",
        "top_n","weight","tau","lam","n_sieve", "inference_mem", "sieve_rank_mem", "sigma"
    }

    def __init__(self, path:str="michael_params.json"):
        self.path = pathlib.Path(path)
        self._data: Dict[str, Dict[str,Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                self._data = json.load(f)
        # ensure every pack has all keys (fill with None)
        for pack in self._data.values():
            for k in self.REQUIRED_KEYS:
                pack.setdefault(k, None)

    # ------------------------------------------------------------------ I/O
    def save(self)->None:
        """Write current store to disk (pretty-printed)."""
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------ CRUD
    def add(self, name:str, params:Dict[str,Any])->None:
        if name in self._data:
            raise KeyError(f"'{name}' already exists.")
        self._data[name] = self._validate(params)

    def get(self, name:str)->Dict[str,Any]:
        return copy.deepcopy(self._data[name])

    def update(self, name:str, **patch)->None:
        if name not in self._data:
            raise KeyError(name)
        self._data[name].update(self._validate(patch, partial=True))

    def delete(self, name:str)->None:
        self._data.pop(name)

    def list(self):
        return list(self._data.keys())

    # ------------------------------------------------------------------ helpers
    def _validate(self, params:Dict[str,Any], *, partial=False)->Dict[str,Any]:
        # basic sanity—only known keys allowed
        unknown = set(params) - self.REQUIRED_KEYS
        if unknown:
            raise ValueError(f"Unknown param(s): {unknown}")
        if not partial:
            missing = self.REQUIRED_KEYS - set(params)
            if missing:
                raise ValueError(f"Missing required param(s): {missing}")
        return params

    # ------------------------------------------------------------------ context mgr
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.save()
