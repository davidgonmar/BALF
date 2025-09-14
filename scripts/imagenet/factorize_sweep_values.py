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


# ResNet-18
ratios_energy_resnet18 = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.98,
    0.99,
    0.992,
    0.99999,
]

ratios_energy_act_aware_resnet18 = [
    0.8,
    0.85,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.98,
    0.99,
    0.992,
    0.995,
    0.997,
    0.99999,
]

# ResNet-50

ratios_energy_resnet50 = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.98,
    0.99,
    0.992,
    0.99999,
]

ratios_energy_act_aware_resnet50 = [
    0.8,
    0.85,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.97,
    0.98,
    0.985,
    0.99,
    0.992,
    0.995,
    0.997,
    0.99999,
]

# ResNeXt-50 32x4d
ratios_energy_resnext50_32x4d = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.825,
    0.85,
    0.865,
    0.88,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.98,
    0.99,
    0.99999,
]

ratios_energy_act_aware_resnext50_32x4d = [
    0.8,
    0.85,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.97,
    0.975,
    0.98,
    0.985,
    0.9875,
    0.99,
    0.992,
    0.995,
    0.997,
    0.99999,
]

# ResNeXt-101 32x8d
ratios_energy_resnext101_32x8d = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.825,
    0.85,
    0.875,
    0.9,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.985,
    0.99,
    0.99999,
]
ratios_energy_act_aware_resnext101_32x8d = [
    0.8,
    0.85,
    0.9,
    0.92,
    0.94,
    0.96,
    0.97,
    0.975,
    0.98,
    0.985,
    0.9875,
    0.99,
    0.992,
    0.995,
    0.997,
    0.998,
    0.999,
    0.99999,
]

# MobileNet-V2
ratios_energy_mobilenet_v2 = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.825,
    0.85,
    0.875,
    0.9,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.99999,
]

ratios_energy_act_aware_mobilenet_v2 = [
    0.8,
    0.85,
    0.9,
    0.92,
    0.94,
    0.96,
    0.97,
    0.975,
    0.98,
    0.99,
    0.995,
    0.997,
    0.999,
    0.99999,
]

# ViT-B/16
ratios_energy_vit_b_16 = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.825,
    0.85,
    0.875,
    0.9,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99999,
]

ratios_energy_act_aware_vit_b_16 = [
    0.8,
    0.85,
    0.9,
    0.92,
    0.94,
    0.95,
    0.96,
    0.965,
    0.97,
    0.975,
    0.98,
    0.985,
    0.9875,
    0.99,
    0.992,
    0.995,
    0.997,
    0.99999,
    0.999999,
]

# DeiT-B/16
ratios_energy_deit_b_16 = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.825,
    0.85,
    0.875,
    0.9,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99999,
]

ratios_energy_act_aware_deit_b_16 = [
    0.8,
    0.85,
    0.9,
    0.92,
    0.94,
    0.95,
    0.96,
    0.965,
    0.97,
    0.975,
    0.98,
    0.985,
    0.9875,
    0.99,
    0.992,
    0.995,
    0.997,
    0.99999,
    0.999999,
]


def get_values_for_model_and_mode(model_name: str, mode: str):
    if mode in ["params_auto", "flops_auto"]:
        return ratios_comp_all  # same for all models
    elif mode == "energy_act_aware":
        if model_name == "resnet18":
            return ratios_energy_act_aware_resnet18
        elif model_name == "resnet50":
            return ratios_energy_act_aware_resnet50
        elif model_name == "resnext50_32x4d":
            return ratios_energy_act_aware_resnext50_32x4d
        elif model_name == "resnext101_32x8d":
            return ratios_energy_act_aware_resnext101_32x8d
        elif model_name == "mobilenet_v2":
            return ratios_energy_act_aware_mobilenet_v2
        elif model_name == "vit_b_16":
            return ratios_energy_act_aware_vit_b_16
        elif model_name == "deit_b_16":
            return ratios_energy_act_aware_deit_b_16
    elif mode == "energy":
        if model_name == "resnet18":
            return ratios_energy_resnet18
        elif model_name == "resnet50":
            return ratios_energy_resnet50
        elif model_name == "resnext50_32x4d":
            return ratios_energy_resnext50_32x4d
        elif model_name == "resnext101_32x8d":
            return ratios_energy_resnext101_32x8d
        elif model_name == "mobilenet_v2":
            return ratios_energy_mobilenet_v2
        elif model_name == "vit_b_16":
            return ratios_energy_vit_b_16
        elif model_name == "deit_b_16":
            return ratios_energy_deit_b_16

    raise ValueError(f"Unsupported model_name {model_name} and/or mode {mode}")
