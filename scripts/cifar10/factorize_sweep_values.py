"""
This is not really a script.
It just contains the different energy and compression ratios for each model.
They were selected so as to give a good spread of points in the final plots.
"""

ratios_comp_all = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    1.00,
]


# ResNet-20
ratios_energy_resnet20 = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.825,
    0.85,
    0.875,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.98,
    0.99999,
]

ratios_energy_act_aware_resnet20 = [
    0.6,
    0.7,
    0.8,
    0.85,
    0.875,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.97,
    0.98,
    0.985,
    0.99,
    0.995,
    0.99999,
]

# ResNet-56
ratios_energy_resnet56 = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.825,
    0.85,
    0.875,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.98,
    0.99999,
]

ratios_energy_act_aware_resnet56 = [
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.875,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.97,
    0.98,
    0.985,
    0.99,
    0.995,
    0.99999,
]


def get_values_for_model_and_mode(model_name: str, mode: str):
    if mode in ["params_auto", "flops_auto", "uniform", "uniform_act_aware"]:
        return ratios_comp_all  # same for all models
    elif mode == "energy_act_aware":
        if model_name == "resnet20":
            return ratios_energy_act_aware_resnet20
        elif model_name == "resnet56":
            return ratios_energy_act_aware_resnet56
    elif mode == "energy":
        if model_name == "resnet20":
            return ratios_energy_resnet20
        elif model_name == "resnet56":
            return ratios_energy_resnet56
    raise ValueError(f"Unsupported model_name {model_name} and/or mode {mode}")
