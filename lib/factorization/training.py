import torch
import torch.nn as nn
from typing import Dict, Optional
from lib.utils import gather_submodules, get_all_convs_and_linears, keys_passlist_should_do
from lib.factorization.factorize import to_low_rank_activation_aware_manual, merge_low_rank_layers


def l1_l2_ratio(vec: torch.Tensor) -> torch.Tensor:
    assert vec.ndim == 1, "Input vector must be 1-dimensional"
    l1 = torch.norm(vec, p=1)
    if l1 == 0:
        return torch.zeros_like(l1)
    return l1 / torch.norm(vec, p=2)

class SingularValsRegularizer(nn.Module):
    def __init__(self, model: nn.Module, layer_names: Optional[str] = None):
        super().__init__()
        self.model = model
        if layer_names is None:
            layer_names = get_all_convs_and_linears(model)
        self.layers = gather_submodules(model, keys_passlist_should_do(layer_names))

    def loss(self) -> torch.Tensor:
        out = 0.0
        for name, layer in self.layers:
            W = (
                layer.weight
                if not isinstance(layer, nn.Conv2d)
                else layer.weight.view(layer.out_channels, -1)
            )
            s = torch.linalg.svdvals(W)
            out = out + l1_l2_ratio(s) ** 2
        return out



class BatchWhiteningShrinker:
    """
    Periodically factorizes selected Linear/Conv2d layers into low-rank form using
    activation-aware whitening and then merges them back into standard layers.

    Usage:
        shrinker = BatchWhiteningShrinker(model, energy_frac=0.95, every=1000)
        shrinker.set_data(train_loader)  # or a dict[name -> activation batch]
        for step in range(num_steps):
            # ... training ...
            shrinker.step()  # will compress+merge every `every` calls
    """
    def __init__(
        self,
        model: nn.Module,
        *,
        energy_frac: float,
        every: int,
        layer_names: Optional[Dict[str, str]] = None
    ):
        self.model = model
        self.energy_frac = energy_frac
        self.every = every
        if layer_names is None:
            layer_names = get_all_convs_and_linears(model)

        # Store current target layers (list of (name, module)) and names
        self.layers = gather_submodules(model, keys_passlist_should_do(layer_names))
        self._layer_names = [name for name, _ in self.layers]

        # Data source used when compressing (either a dataloader or a dict[name -> tensor])
        self._data = None

        self._step = 0

    def set_data(self, data):
        self._data = data

    def _build_cfg(self) -> Dict[str, Dict[str, float]]:
        # Configure all targeted layers to keep enough singular-value energy.
        return {
            name: {"name": "svals_energy_ratio_to_keep", "value": 1 - self.energy_frac}
            for name in self._layer_names
        }

    def _refresh_targets(self):
        # Recompute targets after merges (names may remain but modules change)
        self.layers = gather_submodules(self.model, keys_passlist_should_do(self._layer_names))
        self._layer_names = [name for name, _ in self.layers]

    def step(self, clean_if_used: bool = True):
        if self._step % self.every == 0:
            if self._data is None:
                raise RuntimeError(
                    "No data set for BatchWhiteningShrinker. "
                    "Call `set_data(dataloader_or_acts_dict)` before `step()`."
                )
            cfg = self._build_cfg()
            # Run activation-aware low-rank factorization (whitening inside)
            to_low_rank_activation_aware_manual(
                self.model,
                self._data,
                cfg,
                inplace=True,
            )
            # Merge low-rank layers back to standard layers
            merge_low_rank_layers(self.model, inplace=True)
            # Refresh internal references post-merge
            self._refresh_targets()
            if clean_if_used:
                self._data = None
        self._step += 1