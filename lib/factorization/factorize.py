import copy
import functools
from typing import Callable, Dict, Union
import torch
from torch import nn
from lib.utils import (
    gather_submodules,
    keys_passlist_should_do,
    replace_with_factory,
    is_linear,
    is_conv2d,
)
from .layers import LowRankLinear, LowRankConv2d
import math
import numpy as np
from pathlib import Path
import time


def maximize_energy(
    cum_energy_vectors, cumulative_cost_vectors, total_cost, n_iters=300
):
    """
    Multiple-choice knapsack via Lagrangian relaxation.
    Assumptions (as in the paper):
      (1) The problem is feasible.
      (2) Within each layer, costs and energies are nonnegative.
      (3) Within each layer, costs and energies are nondecreasing with rank.
    Returns 1-based indices (one per layer, indicating the rank to keep).
    """

    def to_array(vec):
        if torch is not None and isinstance(vec, torch.Tensor):
            arr = vec.detach().cpu().numpy()
        else:
            arr = np.array(vec)
        return arr.astype(float)

    energies = [to_array(vec) for vec in cum_energy_vectors]
    costs = [to_array(vec) for vec in cumulative_cost_vectors]

    def compute_selection(lmbda):
        sel_idx = []
        total_c = 0.0
        for e_i, c_i in zip(energies, costs):
            vals = e_i - lmbda * c_i
            j = int(
                np.argmax(vals)
            )  # np.argmax returns the first index with the max value, i.e., min(argmax)
            # basically, as we have nondecreasing energies and costs, this is equivalent min(argmax(vals))
            sel_idx.append(j)
            total_c += float(c_i[j])
        return sel_idx, total_c

    lambda_min = 0.0
    lambda_max = 1.0

    # Initial feasibility check at lambda_min = 0 (in case we can keep all energy within budget)
    # Not usual, but it's better to be safe
    sel_min, cost_min = compute_selection(lambda_min)
    if cost_min <= float(total_cost):
        return [len(e_i) for e_i in energies]

    # Grow lambda_max until feasible
    while True:
        sel_max, cost_max = compute_selection(lambda_max)
        if cost_max <= float(total_cost):
            break
        lambda_max *= 2.0

    # Now, lambda_min selection is infeasible, lambda_max selection is feasible
    # By monotonicity, we know that the "best" feasible lambda in terms of the dual
    # is between lambda_min and lambda_max

    # Bisection for I iterations
    for _ in range(int(n_iters)):
        lmbda = 0.5 * (lambda_min + lambda_max)
        _, cost_mid = compute_selection(lmbda)
        if cost_mid > float(total_cost):
            lambda_min = lmbda
        else:
            lambda_max = lmbda

    # Final selection using lambda_max, which is feasible
    sel_final, _ = compute_selection(lambda_max)
    return [
        j + 1 for j in sel_final
    ]  # convert to 1-based (to denote number of singular values kept)


# ==============================================================================================
# Estimate cost of factorized layers
# P denotes the kept rank
# ==============================================================================================

# In general, the functions compute something along the lines of
# cost(P) = min(cost_low_rank(P), cost_original)
# Since if cost_low_rank(P) > cost_original, we would just keep the original layer

# NOTE: these are actually MACs, but as they are used relatively, it does not matter


def generate_cost_flops_linear(
    weight_shape: tuple, out_shape: tuple, module
) -> torch.Tensor:
    # A decomposed linear layer has shapes W_0 in [O, P] and W_1 in [P, I], input in [B, I] and output in [B, O]
    # flops(P) = min(B * P * (I + O), B * I * O)
    # The output shape might be [B, L, O] in ViTs, fuse into [B*L, O]
    if len(out_shape) == 3:
        out_shape = (out_shape[0] * out_shape[1], out_shape[2])
    O, I = weight_shape
    B = out_shape[0]
    P = torch.arange(1, min(O, I) + 1, 1)
    return B * torch.minimum(P * (I + O), torch.tensor(I * O))


def generate_cost_flops_conv2d(filter_shape: tuple, out_shape: tuple, module):
    # The original convolution has shape W in [C_out, C_in_grp, H_k, W_k], input in [B, C_in, H_in, W_in] and output in [B, C_out, H_out, W_out]
    # Where C_in_grp = C_in / grp (grp is the number of groups)
    # A factorized convolution has shape W_0 in [P * grp, C_in_grp, H_k, W_k] and W_1 in [C_out, P, 1, 1]
    # flops_1(P) = B * P * H_out * W_out * C_in * H_k * W_k + B * C_out * P * H_out * W_out = B * P * H_out * W_out * (C_in * H_k * W_k + C_out)
    # flops_2(P) = B * C_out * H_out * W_out * C_in_grp * H_k * W_k
    # flops(P) = min(flops_1(P), flops_2(P))
    grp = module.groups
    C_out, C_in_grp, H_k, W_k = filter_shape
    C_in = C_in_grp * grp
    B, H_out, W_out = out_shape[0], out_shape[2], out_shape[3]
    P = torch.arange(
        1,
        min(C_out // grp, C_in_grp * H_k * W_k) + 1,
        1,
    )
    return B * torch.minimum(
        P * H_out * W_out * (C_in * H_k * W_k + C_out),
        torch.tensor(C_out * H_out * W_out * H_k * W_k * C_in_grp),
    )


def generate_cost_params_linear(weight_shape: tuple, module) -> torch.Tensor:
    # A decomposed linear layer has shapes W_0 in [O, P] and W_1 in [P, I]
    # params(P) = min(P * (I + O), I * O)
    O, I = weight_shape
    P = torch.arange(1, min(O, I) + 1, 1)
    return torch.minimum(
        P * (I + O),
        torch.tensor(I * O),
    )


def generate_cost_params_conv2d(filter_shape: tuple, module) -> torch.Tensor:
    # A decomposed convolution has shapes W_0 in [P * grp, C_in_grp, H_k, W_k] and W_1 in [C_out, P, 1, 1]
    # params_1(P) = P * (C_in * H_k * W_k + C_out)
    # params_2(P) = C_out * C_in_grp * H_k * W_k
    # Note how the grp cancels out
    C_out, C_in_grp, H_k, W_k = filter_shape
    grp = module.groups
    C_in = C_in_grp * grp
    P = torch.arange(
        1,
        min(C_out // grp, C_in_grp * H_k * W_k) + 1,
        1,
    )
    return torch.minimum(
        P * (C_in * H_k * W_k + C_out), torch.tensor(C_out * C_in_grp * H_k * W_k)
    )


# ==================================
# Other helpers for factorization
# ==================================


def reshape_linear(w: torch.Tensor) -> torch.Tensor:
    assert w.dim() == 2, "Weight tensor must be 2D for linear layers"
    return w.T


def reshape_conv2d(w: torch.Tensor, n_groups) -> torch.Tensor:
    assert w.dim() == 4, "Weight tensor must be 4D for convolutional layers"
    C_o, C_i_by_grp, H_k, W_k = w.shape
    return w.reshape(n_groups, C_o // n_groups, C_i_by_grp * H_k * W_k).permute(
        0, 2, 1
    )  # shape (groups, C_i_by_grp * H_k * W_k, C_o // groups)


def get_reshape(module: nn.Module) -> callable:
    if is_linear(module):
        return reshape_linear
    elif is_conv2d(module):
        return functools.partial(reshape_conv2d, n_groups=module.groups)
    else:
        raise ValueError("Module should be either Linear or Conv2d")


def decompose_params(w: torch.Tensor):
    """
    Expects a matrix or a batch of matrices (each one from a group)
    """
    assert w.device.type == "cuda", "Weights must be on GPU for SVD"
    U, S, V_T = torch.linalg.svd(w, full_matrices=False)  # complete SVD
    return U, S, V_T


def crop_svd(U, S, V_T, rank):
    """
    Expects either a single U, S, V or a batch of them (each one from a group)
    """
    if U.dim() == 2:
        U = U.unsqueeze(0)
        S = S.unsqueeze(0)
        V_T = V_T.unsqueeze(0)
        batched = False
    else:
        batched = True
    ret = U[:, :, :rank], S[:, :rank], V_T[:, :rank, :]
    if not batched:
        ret = ret[0][0], ret[1][0], ret[2][0]
    return ret


def get_factors(U, S, V_T):
    """
    Expects unbatched U, S, V_T
    """
    W0 = U @ torch.diag(torch.sqrt(S))
    W1 = torch.diag(torch.sqrt(S)) @ V_T
    return W0, W1


def should_do_low_rank(W, rank):
    """
    Expects a 2D or 3D tensor (if 3D, the first dimension is the group dimension)
    """
    if W.dim() == 3:
        W = W[0]
    # rank is memory efficient <=> rank is compute efficient
    # by memory efficient I mean "factorizing leads to less parameters"
    # by compute efficient I mean "factorizing leads to less flops"
    m, n = W.shape
    cost_base = m * n
    cost_low_rank = (m + n) * rank
    return cost_low_rank < cost_base


def obtain_whitening_matrix(
    acts: torch.Tensor,
    module: nn.Module,
):
    # acts of shape (G, D, D), where G is the group dimension
    # Cusolver sometimes fails on well-conditioned matrices, set to magma instead
    torch.backends.cuda.preferred_linalg_library("magma")
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            acts.cuda().float()
        )  # acts might be in lower precision
    # on big matrices, eigh might fail (only on very big models, and very sporadic)
    except RuntimeError:
        eigenvalues, eigenvectors = torch.linalg.eig(acts.cuda())
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
    x_svals = torch.sqrt(eigenvalues)
    V = eigenvectors
    keep = x_svals > 1e-10  # of shape (G, D)
    x_svals = torch.where(keep, x_svals, torch.zeros_like(x_svals))
    x_svals_inv = torch.where(keep, 1 / x_svals, torch.zeros_like(x_svals))
    V = torch.where(
        keep.reshape(keep.shape[0], 1, keep.shape[1]), V, torch.zeros_like(V)
    )
    vmap_diag = torch.vmap(torch.diag, in_dims=0)
    return V @ vmap_diag(x_svals_inv), vmap_diag(x_svals) @ V.transpose(-1, -2)


def factorize_linear_whitened(
    module,
    get_rank: Callable,
    data_whitening_matrix,
    data_whitening_matrix_inverse,
    factors=None,
):
    W = module.weight.T
    if factors is None:
        U, S, V_T = decompose_params(data_whitening_matrix_inverse @ W)
    else:
        U, S, V_T = factors
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(W, rank):
        return module
    U, S, V_T = crop_svd(U, S, V_T, rank)
    # linear always has one group
    U = U[0]
    S = S[0]
    V_T = V_T[0]
    data_whitening_matrix = data_whitening_matrix[0]
    W0, W1 = get_factors(U, S, V_T)  # shape (in, rank), (out, rank)
    W0 = data_whitening_matrix @ W0
    low_rank_linear = (
        LowRankLinear(
            module.in_features,
            module.out_features,
            rank,
            bias=module.bias is not None,
        )
    ).to(module.weight.device)
    low_rank_linear.w0.data.copy_(W0)
    low_rank_linear.w1.data.copy_(W1.T)
    if module.bias is not None:
        low_rank_linear.bias.data.copy_(module.bias)
    return low_rank_linear


def factorize_conv2d_whitened(
    module,
    get_rank: Callable,
    data_whitening_matrix,
    data_whitening_matrix_inverse,
    factors=None,
):

    W = module.weight
    C_o, C_i_grp, H_k, W_k = W.shape
    groups = module.groups
    reshaped = W.reshape(groups, C_o // groups, C_i_grp * H_k * W_k).permute(
        0, 2, 1
    )  # shape (groups, C_i_by_grp * H_k * W_k, C_o // groups)
    # data_whitening_matrix of shape (G, D', D')
    # data_whitening_matrix_inverse of shape (G, D', D')
    # where D' = C_i_by_grp * H_k * W_k
    if factors is None:
        U, S, V_T = decompose_params(data_whitening_matrix_inverse @ reshaped)
    else:
        U, S, V_T = factors
    rank = get_rank(reshaped, U, S, V_T)
    if not should_do_low_rank(reshaped, rank):
        return module
    U, S, V_T = crop_svd(
        U, S, V_T, rank
    )  # [G, C_i * H_k * W_k, rank], [G, rank], [G, rank, C_o // groups]
    W0, W1 = torch.vmap(get_factors, in_dims=(0, 0, 0))(
        U, S, V_T
    )  # [G, C_i * H_k * W_k, rank], [G, rank, C_o]
    W0 = data_whitening_matrix @ W0
    W1 = W1.permute(0, 2, 1).reshape(C_o, rank, 1, 1)
    W0 = W0.transpose(-1, -2).reshape(groups * rank, C_i_grp, H_k, W_k)
    low_rank_conv2d = (
        LowRankConv2d(
            module.in_channels,
            module.out_channels,
            (H_k, W_k),
            rank,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
        )
    ).to(module.weight.device)
    low_rank_conv2d.w0.data.copy_(W0)
    low_rank_conv2d.w1.data.copy_(W1)
    if module.bias is not None:
        low_rank_conv2d.bias.data.copy_(module.bias)
    return low_rank_conv2d


def _process_act(act, mod):
    if isinstance(mod, nn.Conv2d):
        # Input should be of shape (B, C_in, H_in, W_in)
        groups = mod.groups
        assert act.dim() == 4
        im2coled = nn.functional.unfold(
            act,
            kernel_size=mod.kernel_size,
            padding=mod.padding,
            stride=mod.stride,
            dilation=mod.dilation,
        )
        im2coled = im2coled.permute(
            0, 2, 1
        )  # shape (B, H_out * W_out, C_in * H_k * W_k,)
        # groups
        im2coled = im2coled.reshape(
            im2coled.shape[0] * im2coled.shape[1], groups, im2coled.shape[2] // groups
        )  # shape (B * H_out * W_out, groups, C_in * H_k * W_k // groups)
        im2coled = im2coled.permute(
            1, 0, 2
        )  # shape (groups, B * H_out * W_out, C_in * H_k * W_k // groups)
    elif isinstance(mod, nn.Linear):
        # Input should be of shape (B, Cin)
        assert act.dim() == 2 or act.dim() == 3  # for language models, [B, L, D]
        im2coled = act.reshape(
            1, -1, act.shape[-1]
        )  # flatten the batch and sequence dimensions, shape (groups=1, B * L, D)
    return im2coled


def _move(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move(v, device) for v in obj)


@torch.no_grad()
def collect_activation_cache(model: nn.Module, data, keys):
    assert isinstance(
        data, torch.utils.data.DataLoader
    ), "data should be a DataLoader, got {}".format(type(data))
    length = len(data.dataset)
    mods = gather_submodules(model, should_do=keys_passlist_should_do(keys))
    device = next(model.parameters()).device
    acts, outs, hooks, inner_dim_count = {}, {}, [], {}

    def fn(n, m, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        a = _process_act(x.detach(), m)
        if acts.get(n) is None:
            acts[n] = torch.zeros(
                a.shape[0], a.shape[2], a.shape[2], device=device, dtype=a.dtype
            )
        acts[n] = acts[n].to(device, non_blocking=True)
        acts[n] += (
            a.transpose(-1, -2) @ a / length
        )  # we divide by length to avoid accumulation of very big numbers
        # but we actually need to divide by the total number of elements reduced in the inner dimension
        # this will be done later
        acts[n] = acts[n].to("cpu").detach()
        outs.setdefault(n, out.shape)
        inner_dim_count[n] = inner_dim_count.get(n, 0) + a.shape[1]

    for n, m in mods:
        hooks.append(m.register_forward_hook(functools.partial(fn, n)))
    state = model.training
    model.eval()
    with torch.no_grad():
        nbatches = len(data)
        it = 0
        for batch in data:
            print("batch {}/{}".format(it + 1, nbatches), end="\r")
            it += 1
            if isinstance(batch, (list, tuple)):
                model(_move(batch[0], device))
            else:
                raise ValueError(
                    "Data should be a tensor or a tuple/list (as in an ImageFolder dataset)"
                )
    model.train(state)
    for h in hooks:
        h.remove()

    # now correct the acts by dividing by the total number of elements reduced in the inner dimension
    for n in acts.keys():
        acts[n] = acts[n] * (length / inner_dim_count[n])
    return {"acts": acts, "outs": outs}


@torch.no_grad()
def to_low_rank_activation_aware_auto(
    model: nn.Module,
    data_or_cache,
    keys,
    ratio_to_keep: float,
    metric: str = "flops",
    inplace: bool = True,
    *,
    save_dir: Union[str, Path],
    n_iters: int = 300,
    benchmark: bool = False,
):
    if not 0 < ratio_to_keep <= 1:
        raise ValueError("ratio_to_keep must be in (0, 1].")
    if metric not in {"flops", "params"}:
        raise ValueError(f"Unknown metric '{metric}'.")
    if not inplace:
        model = copy.deepcopy(model)

    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    def _fname_whit(name: str) -> Path:
        if save_dir is None:
            return Path("")
        return save_dir / (name.replace(".", "__") + "__whit.pt")

    def _fname_fac(name: str) -> Path:
        if save_dir is None:
            return Path("")
        return save_dir / (name.replace(".", "__") + "__fac.pt")

    def _save_whit(name: str, whit):
        torch.save(whit, _fname_whit(name))

    def _save_fac(name: str, fac):
        torch.save(fac, _fname_fac(name))

    def _load_whit(name: str):
        return torch.load(_fname_whit(name), map_location="cuda", weights_only=True)

    def _load_fac(name: str):
        return torch.load(_fname_fac(name), map_location="cuda", weights_only=True)

    # reset max memory stats
    if benchmark:
        torch.cuda.reset_peak_memory_stats()

    benchmark_results = {}

    time_start_cache = time.perf_counter()
    if isinstance(data_or_cache, dict) and {"acts", "outs"} <= set(
        data_or_cache.keys()
    ):
        cache = data_or_cache
    else:
        cache = collect_activation_cache(model, data_or_cache, keys)

    torch.cuda.synchronize()
    time_end_cache = time.perf_counter()
    benchmark_results["time_activation_cache"] = time_end_cache - time_start_cache

    acts, outs = cache["acts"], cache["outs"]
    modules_to_replace = gather_submodules(
        model, should_do=keys_passlist_should_do(keys)
    )

    cum_energies, ws, out_shapes = [], [], []

    time_start_factorization_and_whitening = time.perf_counter()

    for name, module in modules_to_replace:
        if save_dir is not None and _fname_whit(name).exists():
            whit_tuple = _load_whit(name)
        else:
            whit_tuple = obtain_whitening_matrix(acts[name], module)
            _save_whit(name, whit_tuple)
        whit, whitinv = whit_tuple
        if save_dir is not None and _fname_fac(name).exists():
            U, S, V_T = _load_fac(name)
        else:
            reshaped = get_reshape(module)(module.weight.detach())
            aa = whitinv @ reshaped
            assert reshaped.device.type == "cuda"
            U, S, V_T = torch.linalg.svd(aa, full_matrices=False)
            _save_fac(name, (U.cpu(), S.cpu(), V_T.cpu()))

        energy = torch.cumsum((S**2), 1).sum(0)
        energy = energy / energy[-1]
        cum_energies.append(energy)

        ws.append(module.weight.detach())
        out_shapes.append(outs[name])

        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    time_end_factorization_and_whitening = time.perf_counter()
    benchmark_results["time_factorization_and_whitening"] = (
        time_end_factorization_and_whitening - time_start_factorization_and_whitening
    )

    if metric == "flops":
        make_cost = lambda w, o, mod: (
            generate_cost_flops_linear(w.shape, o, mod)
            if len(o) in {2, 3}
            else generate_cost_flops_conv2d(w.shape, o, mod)
        )
        costs = [
            make_cost(w, o, mod[1])
            for w, o, mod in zip(ws, out_shapes, modules_to_replace)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    else:  # params
        make_cost = lambda w, o, mod: (
            generate_cost_params_linear(w.shape, mod)
            if len(o) in {2, 3}
            else generate_cost_params_conv2d(w.shape, mod)
        )
        costs = [
            make_cost(w, o, mod[1])
            for w, o, mod in zip(ws, out_shapes, modules_to_replace)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep

    assert all(
        len(c) == len(e) for c, e in zip(costs, cum_energies)
    ), "Cost and energy vectors must have the same length, got lengths {}".format(
        list(zip([len(c) for c in costs], [len(e) for e in cum_energies]))
    )

    time_start_solver = time.perf_counter()

    selected_indices = maximize_energy(
        cum_energies, costs, total_budget, n_iters=n_iters
    )

    time_end_solver = time.perf_counter()

    benchmark_results["time_solver"] = time_end_solver - time_start_solver

    selected_per_mod = {n: s for (n, _), s in zip(modules_to_replace, selected_indices)}

    def factory_fn(name: str, module: nn.Module):
        whit, whitinv = _load_whit(name)

        U, S, V_T = _load_fac(name)
        fac = (U, S, V_T)
        selector = lambda *_: selected_per_mod[name]
        if is_linear(module):
            return factorize_linear_whitened(
                module, selector, whit, whitinv, factors=fac
            )
        if is_conv2d(module):
            return factorize_conv2d_whitened(
                module, selector, whit, whitinv, factors=fac
            )
        torch.cuda.empty_cache()
        return module

    di = {name: module for name, module in modules_to_replace}
    del modules_to_replace
    del ws
    time_start_replace = time.perf_counter()
    replace_with_factory(model, di, factory_fn)
    torch.cuda.synchronize()
    time_end_replace = time.perf_counter()
    benchmark_results["time_replace"] = time_end_replace - time_start_replace
    benchmark_results["time_total"] = time_end_replace - time_start_cache
    benchmark_results["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    if benchmark:
        print("Benchmark results:")
        for k, v in benchmark_results.items():
            if "memory" in k:
                print(f"  {k}: {v / (1024**2):.2f} MiB")
            else:
                print(f"  {k}: {v:.3f} seconds")
        return model, benchmark_results
    return model


def get_rank_to_keep_from_energy_ratio(X, S: torch.Tensor, energy_ratio: float) -> int:
    """
    X is either a matrix or a batch of matrices (each one from a group)
    S is either a vector or a batch of vectors (each one from a group)
    """
    assert 0.0 <= energy_ratio <= 1.0
    if S.ndim == 1:
        S = S.unsqueeze(0)
    sq = S.pow(2).sum(0)  # sum over groups
    cum_energy = sq.cumsum(dim=0)  # cumsum over rank dim
    total_energy = cum_energy[-1]
    threshold = energy_ratio * total_energy
    idx = torch.searchsorted(cum_energy, threshold)
    return idx.item() + 1


# this one is only used for a toy demonstration in a figure
def get_rank_to_keep_from_rank_ratio(X, S: torch.Tensor, rank_ratio: float):
    """
    X is either a matrix or a batch of matrices (each one from a group)
    S is either a vector or a batch of vectors (each one from a group)
    """
    assert 0.0 <= rank_ratio <= 1.0, "rank_ratio must be in [0, 1]"
    if S.ndim == 1:
        S = S.unsqueeze(0)
    k = math.ceil(S.shape[1] * rank_ratio)
    return max(k, 1)


# See comment below on why this is the same function for both params and flops
def get_rank_to_keep_from_uniform_compression_ratio(
    X: torch.Tensor, S: torch.Tensor, ratio: float
) -> int:
    """
    X is either a matrix or a batch of matrices (each one from a group)
    S is either a vector or a batch of vectors (each one from a group)
    """
    assert 0.0 <= ratio <= 1.0
    if S.ndim == 1:
        S = S.unsqueeze(0)
    m, n = X.shape[-2], X.shape[-1]
    cum_ratios = torch.arange(1, S.shape[1] + 1, device=S.device) * (m + n) / (m * n)
    idx = torch.searchsorted(cum_ratios, ratio)
    return idx.item() + 1


rank_to_keep_name_to_fn = {
    "rank_ratio_to_keep": get_rank_to_keep_from_rank_ratio,
    "svals_energy_ratio_to_keep": get_rank_to_keep_from_energy_ratio,
    # it is easy to check that keeping some ratio of params leads to the same ratio of flops
    # and viceversa (for flops, the spatial dimensions cancel out in the ratio)
    "params_ratio_to_keep": get_rank_to_keep_from_uniform_compression_ratio,
    "flops_ratio_to_keep": get_rank_to_keep_from_uniform_compression_ratio,
    "uniform_compression_ratio_to_keep": get_rank_to_keep_from_uniform_compression_ratio,
}


@torch.no_grad()
def to_low_rank_activation_aware_manual(
    model: nn.Module,
    data_or_cache,
    cfg_dict: Dict,
    *,
    inplace: bool = True,
    save_dir: Union[str, Path],
):

    if not inplace:
        model = copy.deepcopy(model)

    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    def _fname_whit(name: str) -> Path:
        return save_dir / (name.replace(".", "__") + "__whit.pt")

    def _fname_fac(name: str) -> Path:
        return save_dir / (name.replace(".", "__") + "__fac.pt")

    def _save_whit(name: str, whit):
        torch.save(whit, _fname_whit(name))

    def _save_fac(name: str, fac):
        torch.save(fac, _fname_fac(name))

    def _load_whit(name: str):
        return torch.load(_fname_whit(name), map_location="cuda", weights_only=True)

    def _load_fac(name: str):
        return torch.load(_fname_fac(name), map_location="cuda", weights_only=True)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(cfg_dict.keys()),
    )

    if isinstance(data_or_cache, dict) and {"acts", "outs"} <= set(
        data_or_cache.keys()
    ):
        cache = data_or_cache
    else:
        cache = collect_activation_cache(model, data_or_cache, cfg_dict.keys())

    acts = cache["acts"]

    for name, module in modules_to_replace:

        if save_dir is not None and _fname_whit(name).exists():
            whittuple = _load_whit(name)
        else:
            whittuple = obtain_whitening_matrix(acts[name], module)
            _save_whit(name, whittuple)

        whit, whitinv = whittuple

        if save_dir is not None and _fname_fac(name).exists():
            U, S, V_T = _load_fac(name)
        else:
            reshaped = get_reshape(module)(module.weight.detach())
            aa = whitinv @ reshaped
            assert reshaped.device.type == "cuda"
            U, S, V_T = torch.linalg.svd(aa, full_matrices=False)
            _save_fac(name, (U.cpu(), S.cpu(), V_T.cpu()))
        torch.cuda.empty_cache()

    def factory_fn(name: str, module: nn.Module):
        whit, whitinv = _load_whit(name)

        U, S, V_T = _load_fac(name)
        fac = (U, S, V_T)
        rule = cfg_dict[name]
        reshaped = get_reshape(module)(module.weight.detach())

        def selector(*args):
            ret = rank_to_keep_name_to_fn[rule["name"]](reshaped, S, rule["value"])
            return ret

        if is_linear(module):
            return factorize_linear_whitened(
                module, selector, whit, whitinv, factors=fac
            )
        if is_conv2d(module):
            return factorize_conv2d_whitened(
                module, selector, whit, whitinv, factors=fac
            )

        return module

    di = {name: module for name, module in modules_to_replace}
    del modules_to_replace
    del acts
    replace_with_factory(model, di, factory_fn)
    return model


def factorize_linear(module, get_rank: Callable, factors=None):
    # again, linears are never grouped
    W = module.weight.T  # shape (in, out)
    if factors is None:
        U, S, V_T = decompose_params(W)
    else:
        U, S, V_T = factors
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(W, rank):
        return module
    U, S, V_T = crop_svd(U, S, V_T, rank)
    W0, W1 = get_factors(U, S, V_T)  # shape (in, rank), (out, rank)
    low_rank_linear = LowRankLinear(
        module.in_features,
        module.out_features,
        rank,
        bias=module.bias is not None,
    ).to(module.weight.device)
    low_rank_linear.w0.data.copy_(W0)
    low_rank_linear.w1.data.copy_(W1.T)
    if module.bias is not None:
        low_rank_linear.bias.data.copy_(module.bias)
    return low_rank_linear


def factorize_conv2d(module: nn.Conv2d, get_rank: Callable, factors=None):
    W = module.weight
    C_o, C_i_by_grp, H_k, W_k = W.shape
    groups = module.groups

    # Reshape to (G, D, C_out_per_group) where D = C_i_by_grp * H_k * W_k
    reshaped = W.reshape(groups, C_o // groups, C_i_by_grp * H_k * W_k).permute(0, 2, 1)

    # Batched SVD
    if factors is None:
        U, S, V_T = decompose_params(reshaped)  # supports batched inputs
    else:
        U, S, V_T = factors

    # Decide rank and early exit if not beneficial
    rank = get_rank(reshaped, U, S, V_T)
    if not should_do_low_rank(reshaped, rank):
        return module

    # Crop SVD to the selected rank (batched)
    U, S, V_T = crop_svd(U, S, V_T, rank)

    # Compute factors per group: W0: (G, D, r), W1: (G, r, C_out_per_group)
    W0, W1 = torch.vmap(get_factors, in_dims=(0, 0, 0))(U, S, V_T)

    # Map back to conv weights expected by LowRankConv2d
    # w1: (C_out, r, 1, 1)
    W1 = W1.transpose(-1, -2).reshape(C_o, rank, 1, 1)
    # w0: (r * groups, C_in/groups, H_k, W_k)
    W0 = W0.transpose(-1, -2).reshape(rank * groups, C_i_by_grp, H_k, W_k)

    # Build the low-rank conv and copy weights/bias
    low_rank_conv2d = LowRankConv2d(
        module.in_channels,
        module.out_channels,
        (H_k, W_k),
        rank,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
    ).to(module.weight.device)
    low_rank_conv2d.w0.data.copy_(W0)
    low_rank_conv2d.w1.data.copy_(W1)
    if module.bias is not None:
        low_rank_conv2d.bias.data.copy_(module.bias)

    return low_rank_conv2d


def to_low_rank_manual(
    model: nn.Module,
    cfg_dict: Dict,
    inplace=True,
):
    # does not whiten
    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(cfg_dict.keys()),
    )

    def factory_fn(name, module):
        rule = cfg_dict[name]
        reshaped = get_reshape(module)(module.weight.detach())
        selector = lambda W, U, S, V_T: rank_to_keep_name_to_fn[rule["name"]](
            reshaped, S, rule["value"]
        )

        if isinstance(module, nn.Linear):
            return factorize_linear(
                module,
                selector,
            )
        elif isinstance(module, nn.Conv2d):
            return factorize_conv2d(
                module,
                selector,
            )
        else:
            return module

    replace_with_factory(
        model,
        {name: module for name, module in modules_to_replace},
        factory_fn,
    )
    return model
