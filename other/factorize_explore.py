import copy
import functools
from typing import Callable
import torch
from torch import nn
from .utils import (
    gather_submodules,
    keys_passlist_should_do,
    replace_with_factory,
    is_linear,
    is_conv2d,
)
from .layers import LowRankLinear, LowRankConv2d


def maximize_energy(
    cum_energy_vectors, cumulative_cost_vectors, total_cost, minimize=False
):
    import pulp

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
    return [selected_indices[i] for i in range(len(selected_indices))]


def generate_cost_flops_linear(weight_shape: tuple, out_shape: tuple) -> torch.Tensor:
    # A decomposed linear layer has shapes W_0 in [O, R] and W_1 in [R, I], input in [B, I] and output in [B, O]
    # flops(R) = min(B * R * (I + O), B * I * O)
    R = torch.arange(1, min(weight_shape[0], weight_shape[1]) + 1, 1)
    O, I = weight_shape
    B = out_shape[0]
    return B * torch.minimum(R * (I + O), torch.tensor(I * O))


def generate_cost_flops_conv2d(filter_shape: tuple, out_shape: tuple):
    # A factorized convolution has shape
    # W_0 in [R, C_in, H_k, W_k] and W_1 in [C_out, R, 1, 1]
    # flops_1(R) = B * R * H_out * W_out * C_in * H_k * W_k + B * C_out * R * H_out * W_out = B * R * H_out * W_out * (C_in * H_k * W_k + C_out)
    # flops_2(R) = B * C_out * H_out * W_out * C_in * H_k * W_k
    # flops(R) = min(flops_1(R), flops_2(R))
    R = torch.arange(
        1,
        min(filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3]) + 1,
        1,
    )
    C_out, C_in, H_k, W_k = filter_shape
    B, H_out, W_out = out_shape[0], out_shape[2], out_shape[3]
    return B * torch.minimum(
        R * H_out * W_out * (C_in * H_k * W_k + C_out),
        torch.tensor(C_out * H_out * W_out * H_k * W_k * C_in),
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


def reshape_conv2d(w: torch.Tensor) -> torch.Tensor:
    assert w.dim() == 4, "Weight tensor must be 4D for convolutional layers"
    C_o, C_i, H_k, W_k = w.shape
    return w.reshape(C_o, C_i * H_k * W_k).T  # reshape to [C_o, C_i * H_k * W_k]


def get_reshape(module: nn.Module) -> callable:
    """
    Returns a function to reshape the weights of the module.
    """
    if is_linear(module):
        return reshape_linear
    elif is_conv2d(module):
        return reshape_conv2d
    else:
        raise ValueError("Module should be either Linear or Conv2d")


def decompose_params(w: torch.Tensor):
    U, S, V_T = torch.linalg.svd(w, full_matrices=True)  # complete SVD
    return U, S, V_T


def crop_svd(U, S, V_T, rank):
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


def obtain_whitening_matrix_svd(
    acts: torch.Tensor,
    module: nn.Module,
):
    if isinstance(module, nn.Conv2d):
        assert acts.dim() == 4
        im2coled = nn.functional.unfold(
            acts,
            kernel_size=module.kernel_size,
            padding=module.padding,
            stride=module.stride,
        )
        im2coled = im2coled.permute(0, 2, 1).reshape(
            im2coled.shape[0] * im2coled.shape[2], -1
        )
    elif isinstance(module, nn.Linear):
        assert acts.dim() == 2
        im2coled = acts
    else:
        raise ValueError("Module should be either Conv2d or Linear")

    U, S, Vh = torch.linalg.svd(im2coled, full_matrices=False)
    keep = S > 1e-6
    if not torch.any(keep):
        raise RuntimeError("All singular values ≈ 0; cannot whiten.")

    S_nz = S[keep]
    V_nz = Vh[keep, :].T

    return V_nz @ torch.diag(1 / S_nz), torch.diag(S_nz) @ V_nz.T


def obtain_whitening_matrix_eigh(
    acts: torch.Tensor,
    module: nn.Module,
):

    m = acts.T @ acts
    eigenvalues, eigenvectors = torch.linalg.eig(m)
    eigenvalues, eigenvectors = torch.real(eigenvalues), torch.real(eigenvectors)
    x_svals = torch.sqrt(eigenvalues)
    V = eigenvectors
    keep = x_svals > 1e-10
    x_svals = x_svals[keep]
    V = V[:, keep]
    # eye = torch.eye(600, device=m.device, dtype=m.dtype).cuda()
    # ret1 = V @ torch.diag(1 / x_svals)
    # ret2 = torch.diag(x_svals) @ V.T
    # return eye[:ret1.shape[0], :ret1.shape[1]], eye[:ret2.shape[0], :ret2.shape[1]]
    return V @ torch.diag(1 / x_svals), torch.diag(x_svals) @ V.T


def obtain_whitening_matrix(
    acts: torch.Tensor,
    module: nn.Module,
    method: str = "eigh",
):
    if method == "svd":
        return obtain_whitening_matrix_svd(acts, module)
    elif method == "eigh":
        return obtain_whitening_matrix_eigh(acts, module)
    else:
        raise ValueError("Method must be one of 'cholesky', 'svd', or 'eigh'.")


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
    W0, W1 = get_factors(U, S, V_T)  # shape (in, rank), (out, rank)
    W0 = data_whitening_matrix @ W0

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


def factorize_conv2d_whitened(
    module,
    get_rank: Callable,
    data_whitening_matrix,
    data_whitening_matrix_inverse,
    factors=None,
):
    W = module.weight
    C_o, C_i, H_k, W_k = W.shape
    reshaped = W.reshape(C_o, C_i * H_k * W_k).T
    # print(data_whitening_matrix_inverse @ data_whitening_matrix)
    if factors is None:
        U, S, V_T = decompose_params(data_whitening_matrix_inverse @ reshaped)
    else:
        U, S, V_T = factors
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(reshaped, rank):
        return module
    U, S, V_T = crop_svd(
        U, S, V_T, rank
    )  # [C_i * H_k * W_k, rank], [rank], [rank, C_o]
    W0, W1 = get_factors(U, S, V_T)  # [C_i * H_k * W_k, rank], [rank, C_o]
    W0 = data_whitening_matrix @ W0
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


def to_low_rank_activation_aware_global(
    model: nn.Module,
    dataloader,
    keys,
    ratio_to_keep,
    metric: str = "flops",
    inplace: bool = True,
    data_whitening_impl: str = "eigh",
):
    if not inplace:
        model = copy.deepcopy(model)

    device = next(model.parameters()).device

    acts = {}
    outs = {}
    hooks = []

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )

    def _process_act(act, mod):
        if isinstance(mod, nn.Conv2d):
            # Input should be of shape (B, Cin, H, W)
            assert act.dim() == 4
            im2coled = nn.functional.unfold(
                act,
                kernel_size=mod.kernel_size,
                padding=mod.padding,
                stride=mod.stride,
            )
            im2coled = im2coled.permute(0, 2, 1).reshape(
                im2coled.shape[0] * im2coled.shape[2], -1
            )

        elif isinstance(mod, nn.Linear):
            # Input should be of shape (B, Cin)
            assert act.dim() == 2
            im2coled = act
        return im2coled

    def hook_fn(name, module, input, output):
        x = input[0] if isinstance(input, tuple) else input
        if name not in acts:
            a = _process_act(x.detach(), module)
            acts[name] = a.T @ a
        else:
            a = _process_act(x.detach(), module)
            acts[name] = a.T @ a + acts[name]

        # only need for shape tracking
        if name not in outs:
            outs[name] = output.detach()

    for name, module in modules_to_replace:
        hooks.append(module.register_forward_hook(functools.partial(hook_fn, name)))

    prev_state = model.training
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items()}
                _ = model(**inputs)
            else:
                inputs, _ = batch if len(batch) == 2 else (batch, None)
                inputs = inputs.to(device)
                _ = model(inputs)

    model.train(prev_state)

    for h in hooks:
        h.remove()

    len_dataset = len(dataloader.dataset)
    whit = {
        name: obtain_whitening_matrix(
            acts[name] / len_dataset, module, method=data_whitening_impl
        )
        for name, module in modules_to_replace
    }

    cum_energies = []

    for name, module in modules_to_replace:
        reshaped = get_reshape(module)(module.weight.detach())
        aa = whit[name][1] @ reshaped
        svals = torch.linalg.svdvals(aa)

        energy = torch.cumsum(svals**2, 0)
        energy = energy / energy[-1]  # normalize to [0, 1]
        cum_energies.append(energy)

    ws = [mod.weight.detach() for _, mod in modules_to_replace]
    out_shapes = [outs[name].shape for name, _ in modules_to_replace]

    if metric == "rank":
        costs = [
            torch.cumsum(torch.arange(1, len(e) + 1, device=e.device), 0)
            for e in cum_energies
        ]
        total_budget = sum(len(e) for e in cum_energies) * ratio_to_keep
    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w.shape, oshape)
                if len(oshape) == 2
                else generate_cost_flops_conv2d(w.shape, oshape)
            )
            for w, oshape in zip(ws, out_shapes)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w.shape)
                if len(oshape) == 2
                else generate_cost_params_conv2d(w.shape)
            )
            for w, oshape in zip(ws, out_shapes)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from 'flops', 'params', 'rank'."
        )
    cum_energies = [energy for energy, cost in zip(cum_energies, costs)]
    selected_indices = maximize_energy(cum_energies, costs, total_budget)

    selected_indices_per_module = {
        name: sel for (name, _), sel in zip(modules_to_replace, selected_indices)
    }

    def factory_fn(name, module):
        if is_linear(module):
            return factorize_linear_whitened(
                module,
                lambda W, U, S, V_T: selected_indices_per_module[name],
                whit[name][0],
                whit[name][1],
            )
        elif is_conv2d(module):
            return factorize_conv2d_whitened(
                module,
                lambda W, U, S, V_T: selected_indices_per_module[name],
                whit[name][0],
                whit[name][1],
            )
        else:
            return module

    replace_with_factory(
        model,
        {name: module for name, module in modules_to_replace},
        factory_fn,
    )
    return model
