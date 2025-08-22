import copy
import functools
from typing import Callable, Dict, Tuple, Union
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

"""
def maximize_energy(
    cum_energy_vectors, cumulative_cost_vectors, total_cost, minimize=False
):  
    
    print("start")
    print([len(v) for v in cum_energy_vectors])
    # We are given N vectors of cumulative energies and of cumulative vectors. We want to, by selecting a (cumulative)
    # subset of indices from each vector (cumulative in the sense that if j is chosen, all(j' < j) are also chosen), maximize
    # the sum of energies at the selected indices such that the sum of the cumulative costs at the selected indices is less than or equal to the total cost.

    # Let x_{i, j} be a binary variable indicating whether the j-th index in the i-th vector is selected.
    # Then, we want to maximize sum_{i, j} x_{i, j} * cum_energy_vectors[i][j] subject to the constraints:
    # 1. sum_{j} x_{i, j} = 1 for all i
    # 2. sum_{i, j} j * x_{i, j} * cost_vectors[i][j] <= total_cost

    prob = (
        pulp.LpProblem("MaximizeEnergy", pulp.LpMaximize)
        if not minimize
        else pulp.LpProblem("MinimizeEnergy", pulp.LpMinimize)
    )

    selection_vars = {}
    for vec_idx, vec in enumerate(cum_energy_vectors):
        for idx in range(len(vec)):
            selection_vars[(vec_idx, idx)] = pulp.LpVariable(
                f"x_{vec_idx}_{idx}", cat="Binary"
            )
    prob += pulp.lpSum(
        selection_vars[(vec_idx, idx)] * cum_energy_vectors[vec_idx][idx].item()
        for vec_idx, vec in enumerate(cum_energy_vectors)
        for idx in range(len(vec))
    )

    prob += (
        pulp.lpSum(
            selection_vars[(vec_idx, idx)]
            * cumulative_cost_vectors[vec_idx][idx].item()
            for vec_idx, vec in enumerate(cum_energy_vectors)
            for idx in range(len(vec))
        )
        <= total_cost
    )
    for vec_idx, vec in enumerate(cum_energy_vectors):
        prob += (
            pulp.lpSum(selection_vars[(vec_idx, idx)] for idx in range(len(vec))) == 1
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

    selected_indices = {}
    for vec_idx, vec in enumerate(cum_energy_vectors):
        sel = [pulp.value(selection_vars[(vec_idx, idx)]) for idx in range(len(vec))]
        selected_indices[vec_idx] = torch.argmax(torch.tensor(sel)).item() + 1
    print("end")
    return [selected_indices[i] for i in range(len(selected_indices))]
"""


def maximize_energy(
    cum_energy_vectors, cumulative_cost_vectors, total_cost, minimize=False
):
    """
    Solve the multiple-choice knapsack problem via Lagrangian relaxation.
    Returns a list of 1-based selected indices (one per vector), maximizing (or minimizing) total energy
    under the cost budget.
    """

    # print("start")
    # start_time = time.time()
    # Convert inputs to numpy arrays of floats
    def to_array(vec):
        if isinstance(vec, torch.Tensor):
            arr = vec.detach().cpu().numpy()
        else:
            arr = np.array(vec)
        return arr.astype(float)

    energies = [to_array(vec) for vec in cum_energy_vectors]
    costs = [to_array(vec) for vec in cumulative_cost_vectors]
    # num_groups = len(energies)

    low, high = 0.0, 0.0
    for e_vec, c_vec in zip(energies, costs):
        ratios = np.where(c_vec > 0, e_vec / c_vec, 0.0)
        if ratios.size > 0:
            high = max(high, float(np.max(ratios)))

    if high == 0.0:
        high = 1.0

    def compute_selection(lmbda):
        sel_idx = []
        total_c = 0.0
        total_e = 0.0
        for e_i, c_i in zip(energies, costs):
            vals = ((-e_i) if minimize else e_i) - lmbda * c_i
            j = int(np.argmax(vals))
            sel_idx.append(j)
            total_c += float(c_i[j])
            total_e += float(e_i[j])
        return sel_idx, total_c, total_e

    sel0, cost0, _ = compute_selection(low)
    if cost0 <= total_cost:
        return [j + 1 for j in sel0]

    for _ in range(50):
        mid = 0.5 * (low + high)
        _, cost_mid, _ = compute_selection(mid)
        if cost_mid > total_cost:
            low = mid
        else:
            high = mid

    sel_final, _, _ = compute_selection(high)
    # print(f"Time taken: {time.time() - start_time} seconds")
    return [j + 1 for j in sel_final]


def generate_cost_flops_linear(weight_shape: tuple, out_shape: tuple) -> torch.Tensor:
    # A decomposed linear layer has shapes W_0 in [O, R] and W_1 in [R, I], input in [B, I] and output in [B, O]
    # flops(R) = min(B * R * (I + O), B * I * O)
    R = torch.arange(1, min(weight_shape[0], weight_shape[1]) + 1, 1)
    O, I = weight_shape
    B = out_shape[0]
    return B * torch.minimum(R * (I + O), torch.tensor(I * O))


def generate_cost_flops_conv2d(filter_shape: tuple, out_shape: tuple, module):
    if module.groups == 1 or True:
        # A factorized convolution has shape
        # W_0 in [R, C_in, H_k, W_k] and W_1 in [C_out, R, 1, 1]
        # flops_1(R) = B * R * H_out * W_out * C_in * H_k * W_k + B * C_out * R * H_out * W_out = B * R * H_out * W_out * (C_in * H_k * W_k + C_out)
        # flops_2(R) = B * C_out * H_out * W_out * C_in * H_k * W_k
        # flops(R) = min(flops_1(R), flops_2(R))
        R = torch.arange(
            1,
            min(filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3])
            + 1,
            1,
        )
        C_out, C_in, H_k, W_k = filter_shape
        B, H_out, W_out = out_shape[0], out_shape[2], out_shape[3]
        return B * torch.minimum(
            R * H_out * W_out * (C_in * H_k * W_k + C_out),
            torch.tensor(C_out * H_out * W_out * H_k * W_k * C_in),
        )
    else:
        grps = module.groups
        # A grouped convolution has shape
        # W_0 in [R * groups, C_in/groups, H_k, W_k] and W_1 in [C_out, R, 1, 1]
        # flops_1(R) = (H_k * W_k * H_out * R * groups * C_in/groups) * groups
        # flops(R) = min(flops_1(R), flops_2(R))
        R = torch.arange(
            1,
            min(filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3])
            + 1,
            1,
        )
        C_out, C_in_by_grp, H_k, W_k = filter_shape
        B, H_out, W_out = out_shape[0], out_shape[2], out_shape[3]
        G = module.groups
        return B * torch.minimum(
            R * H_out * W_out * (C_in * H_k * W_k + C_out // G),
            torch.tensor(C_out // G * H_out * W_out * H_k * W_k * C_in),
        )


def generate_cost_params_linear(weight_shape: tuple) -> torch.Tensor:
    # A decomposed linear layer has shapes W_0 in [O, R] and W_1 in [R, I]
    # params(R) = min(R * (I + O), I * O)
    r_vec = torch.arange(1, min(weight_shape[0], weight_shape[1]) + 1, 1)
    O, I = weight_shape
    return torch.minimum(
        r_vec * (I + O),
        torch.tensor(I * O),
    )


def generate_cost_params_conv2d(filter_shape: tuple) -> torch.Tensor:
    # A decomposed convolution has shapes W_0 in [R, C_in, H_k, W_k] and W_1 in [C_out, R, 1, 1]
    # params_1(R) = R * (C_in * H_k * W_k + C_out)
    # params_2(R) = C_out * C_in * H_k * W_k
    R = torch.arange(
        1,
        min(filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3]) + 1,
        1,
    )

    C_out, C_in, H_k, W_k = filter_shape
    return torch.minimum(
        R * (C_in * H_k * W_k + C_out),
        torch.tensor(C_out * C_in * H_k * W_k),
    )


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
    U, S, V_T = torch.linalg.svd(w, full_matrices=True)  # complete SVD
    # print("w shape:", w.shape, "U shape:", U.shape, "S shape:", S.shape, "V_T shape:", V_T.shape)
    return U, S, V_T


def crop_svd(U, S, V_T, rank):
    if U.dim() == 3:
        return U[:, :, :rank], S[:, :rank], V_T[:, :rank, :]
    elif U.dim() == 2:
        return U[:, :rank], S[:rank], V_T[:rank, :]


def get_factors(U, S, V_T):
    W0 = U @ torch.diag(torch.sqrt(S))
    W1 = torch.diag(torch.sqrt(S)) @ V_T
    return W0, W1


def should_do_low_rank(W, rank):
    # it can be proved that rank is memory efficient <=> rank is compute efficient
    m, n = W.shape
    cost_base = m * n
    cost_low_rank = (m + n) * rank
    return cost_low_rank < cost_base


def obtain_whitening_matrix(
    acts: torch.Tensor,
    module: nn.Module,
):
    # acts of shape (G, B, D)
    # cusolver seems to have a memory access error?
    # print(acts.shape)
    torch.backends.cuda.preferred_linalg_library("magma")
    eigenvalues, eigenvectors = torch.linalg.eigh(
        acts.cuda().float()
    )  # acts might be in lower precision
    # print(eigenvalues.shape, eigenvectors.shape)
    eigenvalues, eigenvectors = eigenvalues.to(acts.dtype), eigenvectors.to(acts.dtype)
    x_svals = torch.sqrt(eigenvalues)
    V = eigenvectors
    keep = x_svals > 1e-10  # of shape (G, D)
    x_svals = torch.where(keep, x_svals, torch.zeros_like(x_svals))
    x_svals_inv = torch.where(keep, 1 / x_svals, torch.zeros_like(x_svals))
    V = torch.where(
        keep.reshape(keep.shape[0], 1, keep.shape[1]), V, torch.zeros_like(V)
    )
    # print(x_svals.shape, x_svals_inv.shape, V.shape)
    vmap_diag = torch.vmap(torch.diag, in_dims=0)
    # print(f"Whitening matrix for {module.__class__.__name__} has rank {len(x_svals)}")
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
    U = U[0]
    S = S[0]
    V_T = V_T[0]
    data_whitening_matrix = data_whitening_matrix[0]
    data_whitening_matrix_inverse = data_whitening_matrix_inverse[0]
    # print(W.shape, U.shape, S.shape, V_T.shape, rank)
    if not should_do_low_rank(W, rank):
        return module
    U, S, V_T = crop_svd(U, S, V_T, rank)
    # print("cropped U shape:", U.shape, "S shape:", S.shape, "V_T shape:", V_T.shape)
    W0, W1 = get_factors(U, S, V_T)  # shape (in, rank), (out, rank)
    W0 = data_whitening_matrix @ W0

    low_rank_linear = (
        LowRankLinear(
            module.in_features,
            module.out_features,
            rank,
            bias=module.bias is not None,
        )
        .to(module.weight.device)
        .to(module.weight.dtype)
    )
    low_rank_linear.w0.data.copy_(W0)
    low_rank_linear.w1.data.copy_(W1)
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
    C_o, C_i_by_grp, H_k, W_k = W.shape
    groups = module.groups
    # print(module.weight.shape, groups)
    reshaped = W.reshape(groups, C_o // groups, C_i_by_grp * H_k * W_k).permute(
        0, 2, 1
    )  # shape (groups, C_i_by_grp * H_k * W_k, C_o // groups)
    # data_whitening_matrix of shape (G, D', D')
    # data_whitening_matrix_inverse of shape (G, D', D')
    if factors is None:
        U, S, V_T = decompose_params(data_whitening_matrix_inverse @ reshaped)
    else:
        U, S, V_T = factors

    # print(f"U shape: {U.shape}, S shape: {S.shape}, V_T shape: {V_T.shape}")
    rank = get_rank(W, U, S, V_T)
    if False and not should_do_low_rank(reshaped, rank):
        return module
    U, S, V_T = crop_svd(
        U, S, V_T, rank
    )  # [C_i * H_k * W_k, rank], [rank], [rank, C_o]
    W0, W1 = torch.vmap(get_factors, in_dims=(0, 0, 0))(
        U, S, V_T
    )  # [C_i * H_k * W_k, rank], [rank, C_o]
    # print(data_whitening_matrix.shape, W0.shape, W1.shape)
    W0 = data_whitening_matrix @ W0
    W1 = W1.transpose(-1, -2).reshape(C_o, rank, 1, 1)
    W0 = W0.transpose(-1, -2).reshape(rank * groups, C_i_by_grp, H_k, W_k)
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
        .to(module.weight.device)
        .to(module.weight.dtype)
    )
    low_rank_conv2d.w0.data.copy_(W0)
    low_rank_conv2d.w1.data.copy_(W1)
    if module.bias is not None:
        low_rank_conv2d.bias.data.copy_(module.bias)
    return low_rank_conv2d


def _process_act(act, mod):
    if isinstance(mod, nn.Conv2d):
        # Input should be of shape (B, Cin, H, W)
        groups = mod.groups
        assert act.dim() == 4
        im2coled = nn.functional.unfold(
            act,
            kernel_size=mod.kernel_size,
            padding=mod.padding,
            stride=mod.stride,
        )
        im2coled = im2coled.permute(
            0, 2, 1
        )  # shape (B, H_out * W_out, C_in * H_k * W_k, )
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
    return im2coled.float()


# -------------------------
# Public collection utility
# -------------------------


def _move(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move(v, device) for v in obj)


def collect_activation_cache(model: nn.Module, data, keys):
    length = data.size(0) if isinstance(data, torch.Tensor) else len(data)
    loader = [data] if isinstance(data, torch.Tensor) else data
    mods = gather_submodules(model, should_do=keys_passlist_should_do(keys))
    device = next(model.parameters()).device
    acts, outs, hooks = {}, {}, []

    def fn(n, m, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        a = _process_act(x.detach(), m)
        # print(a.shape)
        if acts.get(n) is None:
            acts[n] = torch.zeros(
                a.shape[0], a.shape[2], a.shape[2], device=device, dtype=a.dtype
            )
        acts[n] = acts[n].to(device, non_blocking=True)
        acts[n] += a.transpose(-1, -2) @ a / length
        acts[n] = acts[n].to("cpu")
        outs.setdefault(n, out.detach().cpu())

    for n, m in mods:
        hooks.append(m.register_forward_hook(functools.partial(fn, n)))
    state = model.training
    model.eval()
    with torch.no_grad():
        nbatches = len(loader)
        it = 0
        for batch in loader:
            print("batch {}/{}".format(it + 1, nbatches), end="\r")
            it += 1
            if torch.is_tensor(batch):
                model(batch.to(device))
            elif isinstance(batch, (list, tuple)):
                model(_move(batch[0], device))
            elif isinstance(batch, dict):
                model(
                    **{
                        k: v.to(device)
                        for k, v in batch.items()
                        if k not in {"labels", "label"} and torch.is_tensor(v)
                    }
                )
            else:
                raise TypeError(type(batch))
    model.train(state)
    for h in hooks:
        h.remove()

    return {"acts": acts, "outs": outs, "len_dataset": length}


@torch.no_grad()
def to_low_rank_activation_aware_auto(
    model: nn.Module,
    data_or_cache,
    keys,
    ratio_to_keep: float,
    metric: str = "flops",
    inplace: bool = True,
    *,
    save_dir: Union[str, Path, None] = "./tmp_whitening",
    keep_whiteners_in_mem: bool = False,
    load_existing: bool = True,
    keep_factors_in_mem: bool = False,
):
    if not 0 < ratio_to_keep <= 1:
        raise ValueError("ratio_to_keep must be in (0, 1].")
    if metric not in {"flops", "params", "rank"}:
        raise ValueError(f"Unknown metric '{metric}'.")
    if not inplace:
        model = copy.deepcopy(model)

    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    whit_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    fac_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # ---------- helpers ----------
    def _fname_whit(name: str) -> Path:
        return save_dir / (name.replace(".", "__") + ".pt")

    def _fname_fac(name: str) -> Path:
        return save_dir / (name.replace(".", "__") + "__fac.pt")

    def _save_whit(name: str, whit):
        if save_dir is None:
            return
        torch.save(whit, _fname_whit(name), _use_new_zipfile_serialization=False)

    def _save_fac(name: str, fac):
        if save_dir is None:
            return
        torch.save(fac, _fname_fac(name), _use_new_zipfile_serialization=False)

    def _load_whit(name: str):
        if name in whit_cache:
            return whit_cache[name]
        whit = torch.load(_fname_whit(name), map_location="cpu")
        if keep_whiteners_in_mem:
            whit_cache[name] = whit
        return (whit[0].cuda(), whit[1].cuda()) if torch.cuda.is_available() else whit

    def _load_fac(name: str):
        if name in fac_cache:
            return fac_cache[name]
        fac = torch.load(_fname_fac(name), map_location="cpu")
        if keep_factors_in_mem:
            fac_cache[name] = fac
        return fac

    # ---------- get (or build) activation cache ----------
    if isinstance(data_or_cache, dict) and {"acts", "outs", "len_dataset"} <= set(
        data_or_cache.keys()
    ):
        cache = data_or_cache
    else:
        cache = collect_activation_cache(model, data_or_cache, keys)

    acts, outs, len_dataset = cache["acts"], cache["outs"], cache["len_dataset"]
    modules_to_replace = gather_submodules(
        model, should_do=keys_passlist_should_do(keys)
    )

    cum_energies, ws, out_shapes = [], [], []

    print("[LRA-AUTO] Calculating cumulative energies (cached mode)…")
    for name, module in modules_to_replace:
        # --- whitening matrix ---
        if save_dir is not None and load_existing and _fname_whit(name).exists():
            whit = _load_whit(name)
        else:
            whit = obtain_whitening_matrix(acts[name], module)
            _save_whit(name, whit)
            if keep_whiteners_in_mem:
                whit_cache[name] = whit
        P, W = whit

        # --- SVD factors of whitened weight ---
        if save_dir is not None and load_existing and _fname_fac(name).exists():
            U, S, V_T = _load_fac(name)
            S = S.to(torch.float32)  # ensure numeric stability
        else:
            reshaped = get_reshape(module)(module.weight.detach())
            # print(reshaped.shape, module.weight.shape, W.shape, P.shape)
            aa = W @ reshaped
            U, S, V_T = torch.linalg.svd(aa.float(), full_matrices=False)
            # print(U.shape, S.shape, V_T.shape)
            # print('hehehee')
            _save_fac(name, (U.cpu(), S.cpu(), V_T.cpu()))
            if keep_factors_in_mem:
                fac_cache[name] = (U, S, V_T)

        energy = torch.cumsum((S**2).sum(0), 0)
        energy = energy / energy[-1]
        cum_energies.append(energy)

        ws.append(module.weight.detach())
        out_shapes.append(outs[name].shape)

        if not keep_whiteners_in_mem:
            whit_cache.pop(name, None)
        if not keep_factors_in_mem:
            fac_cache.pop(name, None)
        torch.cuda.empty_cache()

    # ---------- build cost vectors ----------
    if metric == "rank":
        costs = [
            torch.cumsum(torch.arange(1, len(e) + 1, device=e.device), 0)
            for e in cum_energies
        ]
        total_budget = sum(len(e) for e in cum_energies) * ratio_to_keep
    elif metric == "flops":
        make_cost = lambda w, o, mod: (
            generate_cost_flops_linear(w.shape, o, mod)
            if len(o) in {2, 3}
            else generate_cost_flops_conv2d(w.shape, o, mod)
        )
        costs = [
            make_cost(w, o, mod)
            for w, o, mod in zip(ws, out_shapes, modules_to_replace)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    else:  # params
        make_cost = lambda w, o: (
            generate_cost_params_linear(w.shape)
            if len(o) in {2, 3}
            else generate_cost_params_conv2d(w.shape)
        )
        costs = [make_cost(w, o) for w, o in zip(ws, out_shapes)]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep

    costs = [c[: len(e)] for c, e in zip(costs, cum_energies)]

    print("[LRA-AUTO] Selecting ranks (knapsack)…")
    import time

    t0 = time.time()
    selected_indices = maximize_energy(cum_energies, costs, total_budget)
    print(f"[LRA-AUTO] Rank selection done. ({time.time() - t0:.2f}s)")

    selected_per_mod = {n: s for (n, _), s in zip(modules_to_replace, selected_indices)}

    # ---------- replace modules ----------
    def factory_fn(name: str, module: nn.Module):
        # print(f"[LRA-AUTO] Replacing module '{name}' with low-rank version.")
        P, W = _load_whit(name)
        P, W = P.to(module.weight.dtype), W.to(module.weight.dtype)

        U, S, V_T = _load_fac(name)
        U = U.to(module.weight.device, dtype=module.weight.dtype)
        S = S.to(module.weight.device, dtype=module.weight.dtype)
        V_T = V_T.to(module.weight.device, dtype=module.weight.dtype)
        fac = (U, S, V_T)
        # print("YEAH U.shape:", U.shape, "S.shape:", S.shape, "V_T.shape:", V_T.shape)

        selector = lambda *_: selected_per_mod[name]

        if is_linear(module):
            return factorize_linear_whitened(module, selector, P, W, factors=fac)
        if is_conv2d(module):
            return factorize_conv2d_whitened(module, selector, P, W, factors=fac)

        torch.cuda.empty_cache()
        return module

    di = {name: module for name, module in modules_to_replace}
    del modules_to_replace
    del ws
    replace_with_factory(model, di, factory_fn)
    return model


def get_rank_to_keep_from_rank_ratio(
    X: torch.tensor, S: torch.Tensor, rank_ratio: float
):
    # truncates towards 0
    assert 0.0 <= rank_ratio <= 1.0, "rank_ratio must be in [0, 1]"
    k = math.ceil(S.shape[0] * rank_ratio)
    return max(k, 1)


def get_rank_to_keep_from_energy_ratio(
    X: torch.Tensor, S: torch.Tensor, energy_ratio: float
) -> int:
    assert 0.0 <= energy_ratio <= 1.0
    sq = S.pow(2)
    cum_energy = sq.cumsum(dim=0)
    total_energy = cum_energy[-1]
    threshold = energy_ratio * total_energy
    idx = torch.searchsorted(cum_energy, threshold)
    return idx.item() + 1


def get_rank_to_keep_from_param_number_ratio(
    X: torch.Tensor,
    S: torch.Tensor,
    param_number_ratio: float,
):
    assert X.ndim == 2, "X must be 2-dimensional"
    assert S.ndim == 1, "Singular values must be 1-dimensional"
    m, n = X.shape
    # A in R^{m x r}
    # B in R^{r x n}
    # So keeping a rank involves a total of m + n parameters
    params_per_rank_kept = torch.arange(1, S.shape[0] + 1).float() * (m + n)
    rel_params_per_rank_kept = params_per_rank_kept / params_per_rank_kept[-1]
    rank_to_keep = torch.searchsorted(
        rel_params_per_rank_kept, param_number_ratio
    )  # rank_to_keep is the number of ranks to keep
    return rank_to_keep.item() + 1


rank_to_keep_name_to_fn = {
    "rank_ratio_to_keep": get_rank_to_keep_from_rank_ratio,
    "svals_energy_ratio_to_keep": get_rank_to_keep_from_energy_ratio,
    "params_ratio_to_keep": get_rank_to_keep_from_param_number_ratio,
}


def to_low_rank_activation_aware_manual(
    model: nn.Module,
    data_or_cache,
    cfg_dict: Dict,
    inplace=True,
):

    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(cfg_dict.keys()),
    )

    if isinstance(data_or_cache, dict) and {"acts", "outs", "len_dataset"} <= set(
        data_or_cache.keys()
    ):
        cache = data_or_cache
    else:
        cache = collect_activation_cache(model, data_or_cache, cfg_dict.keys())

    acts_auto, _, len_dataset = cache["acts"], cache["outs"], cache["len_dataset"]

    # get the cholesky decomposition of the covariance matrix of each activation im2col'ed in case of conv2d
    whit = {
        name: obtain_whitening_matrix(acts_auto[name] / len_dataset, module)
        for name, module in modules_to_replace
    }

    def factory_fn(name, module):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        if is_linear(module):
            return factorize_linear_whitened(
                module,
                lambda W, U, S, V_T: rank_to_keep_name_to_fn[cfg_dict[name]["name"]](
                    W, S, cfg_dict[name]["value"]
                ),
                whit[name][0],
                whit[name][1],
            )
        elif is_conv2d(module):
            return factorize_conv2d_whitened(
                module,
                lambda W, U, S, V_T: rank_to_keep_name_to_fn[cfg_dict[name]["name"]](
                    W, S, cfg_dict[name]["value"]
                ),
                whit[name][0],
                whit[name][1],
            )
        else:
            return module

    modules_to_replace = {name: module for name, module in modules_to_replace}
    replace_with_factory(model, modules_to_replace, factory_fn)
    return model


@torch.no_grad()
def to_low_rank_activation_aware_manual(
    model: nn.Module,
    data_or_cache,
    cfg_dict: Dict,
    *,
    inplace: bool = True,
    # cache settings -----------------------------------------------------------
    save_dir: Union[str, Path, None] = "./tmp_whitening",
    keep_whiteners_in_mem: bool = False,
    load_existing: bool = True,
    keep_factors_in_mem: bool = False,
):
    # ---------------------------------------------------------------------
    # 0. Sanity checks and basic setup
    # ---------------------------------------------------------------------
    if not inplace:
        model = copy.deepcopy(model)

    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    whit_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    fac_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # filename helpers ----------------------------------------------------
    def _fname_whit(name: str) -> Path:
        return save_dir / (name.replace(".", "__") + ".pt")

    def _fname_fac(name: str) -> Path:
        return save_dir / (name.replace(".", "__") + "__fac.pt")

    # load/store helpers --------------------------------------------------
    def _save_whit(name: str, whit):
        if save_dir is None:
            return
        torch.save(whit, _fname_whit(name), _use_new_zipfile_serialization=False)

    def _save_fac(name: str, fac):
        if save_dir is None:
            return
        torch.save(fac, _fname_fac(name), _use_new_zipfile_serialization=False)

    def _load_whit(name: str):
        if name in whit_cache:
            return whit_cache[name]
        whit = torch.load(_fname_whit(name), map_location="cpu")
        if keep_whiteners_in_mem:
            whit_cache[name] = whit
        return (whit[0].cuda(), whit[1].cuda()) if torch.cuda.is_available() else whit

    def _load_fac(name: str):
        if name in fac_cache:
            return fac_cache[name]
        fac = torch.load(_fname_fac(name), map_location="cpu")
        if keep_factors_in_mem:
            fac_cache[name] = fac
        return fac

    # ---------------------------------------------------------------------
    # 1. Gather sub‑modules and activation cache
    # ---------------------------------------------------------------------
    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(cfg_dict.keys()),
    )

    if isinstance(data_or_cache, dict) and {"acts", "outs", "len_dataset"} <= set(
        data_or_cache.keys()
    ):
        cache = data_or_cache
    else:
        cache = collect_activation_cache(model, data_or_cache, cfg_dict.keys())

    acts, _, len_dataset = cache["acts"], cache["outs"], cache["len_dataset"]

    # ---------------------------------------------------------------------
    # 2. Build / load whitening matrices and SVD factors (once!)
    # ---------------------------------------------------------------------
    print("[LRA‑MANUAL] Preparing whitening matrices and SVD factors…")
    for name, module in modules_to_replace:
        # -- whitening --------------------------------------------------
        if save_dir is not None and load_existing and _fname_whit(name).exists():
            whit = _load_whit(name)
        else:
            whit = obtain_whitening_matrix(acts[name] / len_dataset, module)
            _save_whit(name, whit)
            if keep_whiteners_in_mem:
                whit_cache[name] = whit
        P, W = whit  # noqa: F841 (kept for clarity)

        # -- SVD factors ------------------------------------------------
        if save_dir is not None and load_existing and _fname_fac(name).exists():
            U, S, V_T = _load_fac(name)
            S = S.to(torch.float32)
        else:
            reshaped = get_reshape(module)(module.weight.detach())
            print(reshaped.shape, W.shape)
            aa = W @ reshaped
            U, S, V_T = torch.linalg.svd(aa.float(), full_matrices=False)
            _save_fac(name, (U.cpu(), S.cpu(), V_T.cpu()))
            if keep_factors_in_mem:
                fac_cache[name] = (U, S, V_T)

        # free up memory if requested ----------------------------------
        if not keep_whiteners_in_mem:
            whit_cache.pop(name, None)
        if not keep_factors_in_mem:
            fac_cache.pop(name, None)
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 3. Factory that swaps modules for their low‑rank versions
    # ---------------------------------------------------------------------
    def factory_fn(name: str, module: nn.Module):
        print(f"[LRA‑MANUAL] Replacing module '{name}' with low‑rank version.")
        P, W = _load_whit(name)
        P, W = P.to(module.weight.dtype), W.to(module.weight.dtype)

        U, S, V_T = _load_fac(name)
        U = U.to(module.weight.device, dtype=module.weight.dtype)
        S = S.to(module.weight.device, dtype=module.weight.dtype)
        V_T = V_T.to(module.weight.device, dtype=module.weight.dtype)
        fac = (U, S, V_T)

        # Rank selector according to cfg_dict -------------------------
        rule = cfg_dict[name]
        selector = lambda *_: rank_to_keep_name_to_fn[rule["name"]](
            module.weight, S, rule["value"]
        )

        # Actual replacement -----------------------------------------
        if is_linear(module):
            return factorize_linear_whitened(module, selector, P, W, factors=fac)
        if is_conv2d(module):
            return factorize_conv2d_whitened(module, selector, P, W, factors=fac)
        return module  # fallback (should not happen)

    # ---------------------------------------------------------------------
    # 4. Perform the replacement and return the model
    # ---------------------------------------------------------------------
    di = {name: module for name, module in modules_to_replace}
    del modules_to_replace
    del acts  # free memory
    replace_with_factory(model, di, factory_fn)
    return model


def factorize_linear(module, get_rank: Callable, factors=None):
    W = module.weight.T  # shape (in, out)
    if factors is None:
        U, S, V_T = decompose_params(W)
    else:
        U, S, V_T = factors
    # print(W.shape, U.shape, S.shape, V_T.shape)
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(W, rank):
        return module
    U, S, V_T = crop_svd(U, S, V_T, rank)
    # print("Cropped SVD shapes:", U.shape, S.shape, V_T.shape)
    W0, W1 = get_factors(U, S, V_T)  # shape (in, rank), (out, rank)
    low_rank_linear = LowRankLinear(
        module.in_features,
        module.out_features,
        rank,
        bias=module.bias is not None,
    ).to(module.weight.device)
    low_rank_linear.w0.data.copy_(W0)
    low_rank_linear.w1.data.copy_(W1)
    if module.bias is not None:
        low_rank_linear.bias.data.copy_(module.bias)
    return low_rank_linear


def factorize_conv2d(module, get_rank: Callable, factors=None):
    W = module.weight
    C_o, C_i, H_k, W_k = W.shape
    reshaped = W.reshape(C_o, C_i * H_k * W_k).T
    if factors is None:
        U, S, V_T = decompose_params(reshaped)
    else:
        U, S, V_T = factors
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(reshaped, rank):
        return module
    U, S, V_T = crop_svd(
        U, S, V_T, rank
    )  # [C_i * H_k * W_k, rank], [rank], [rank, C_o]
    W0, W1 = get_factors(U, S, V_T)  # shape (C_i * H_k * W_k, rank), (rank, C_o)
    W1 = W1.T.reshape(C_o, rank, 1, 1)
    W0 = W0.T.reshape(rank, C_i, H_k, W_k)
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
        if isinstance(module, nn.Linear):
            return factorize_linear(
                module,
                lambda W, U, S, V_T: get_rank_to_keep_from_rank_ratio(
                    W, S, cfg_dict[name]["value"]
                ),
            )
        elif isinstance(module, nn.Conv2d):
            return factorize_conv2d(
                module,
                lambda W, U, S, V_T: get_rank_to_keep_from_rank_ratio(
                    W, S, cfg_dict[name]["value"]
                ),
            )
        else:
            return module

    replace_with_factory(
        model,
        {name: module for name, module in modules_to_replace},
        factory_fn,
    )
    return model


def merge_low_rank_layers(
    model: nn.Module,
    inplace: bool = True,
):
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_merge = gather_submodules(
        model,
        should_do=lambda mod, name: isinstance(mod, (LowRankLinear, LowRankConv2d)),
    )

    def factory_fn(name, module):
        return (
            module.to_linear()
            if isinstance(module, LowRankLinear)
            else module.to_conv2d()
        )

    replace_with_factory(
        model,
        {name: module for name, module in modules_to_merge},
        factory_fn,
    )
