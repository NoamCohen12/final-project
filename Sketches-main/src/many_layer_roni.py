import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from collections import OrderedDict


def get_quant_params(tensor, grid_min, grid_max, clamp=False, clamp_percent=0.05):
    if clamp:
        flat_tensor = tensor.flatten()
        k = int(clamp_percent * flat_tensor.numel())
        sorted_tensor, _ = torch.sort(flat_tensor)
        rmin = sorted_tensor[k]
        rmax = sorted_tensor[-k - 1]
    else:
        rmin, rmax = tensor.min(), tensor.max()
    scale = (rmax - rmin) / (grid_max - grid_min)
    zero_point = grid_min - torch.round(rmin / scale)
    return scale, zero_point


def quantize_tensor(tensor, scale, zero_point, grid_min, grid_max):
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor.clamp_(grid_min, grid_max)
    return q_tensor


def dequantize_tensor(q_tensor, scale, zero_point):
    return scale * (q_tensor - zero_point)


def quantize_model(model, grid_min, grid_max, clamp=False):
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            with torch.no_grad():
                weight = layer.weight.data
                scale, zero_point = get_quant_params(weight, grid_min, grid_max, clamp=clamp)
                q_weight = quantize_tensor(weight, scale, zero_point, grid_min, grid_max)
                dq_weight = dequantize_tensor(q_weight, scale, zero_point)
                layer.weight.data = dq_weight


def compute_errors(original, quantized):
    rel_error = torch.abs(original - quantized) / (torch.abs(original) + 1e-6)
    return rel_error.mean().item(), rel_error.max().item()


def run_quant_experiment(input_tensor, grid_name, clamp=False):
    bits_range = list(range(8, 17))
    mean_errors, max_errors = [], []

    for bits in bits_range:
        if grid_name == 'int':
            grid_min = -(2 ** (bits - 1)) + 1
            grid_max = (2 ** (bits - 1)) - 1
        elif grid_name == 'F2P_sr_h2':
            grid_min = -2 ** (bits - 2)
            grid_max = 2 ** (bits - 2) - 1
        elif grid_name == 'F2P_li_h2':
            grid_min = 0
            grid_max = 2 ** bits - 1
        else:
            raise ValueError("Unknown grid name")

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        original_output = test_model_output(model, input_tensor)

        quantize_model(model, grid_min, grid_max, clamp=clamp)
        quant_output = test_model_output(model, input_tensor)

        mean_err, max_err = compute_errors(original_output, quant_output)
        mean_errors.append(mean_err)
        max_errors.append(max_err)

    return bits_range, mean_errors, max_errors


def test_model_output(model, input_tensor):
    model.eval()
    with torch.no_grad():
        return model(input_tensor)


def plot_errors(results, title_suffix=""):
    plt.figure()
    for label, (bits, means, _) in results.items():
        plt.plot(bits, means, label=label)
    plt.title(f"Mean Relative Error {title_suffix}")
    plt.xlabel("Bits")
    plt.ylabel("Mean Error")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    for label, (bits, _, maxs) in results.items():
        plt.plot(bits, maxs, label=label)
    plt.title(f"Max Relative Error {title_suffix}")
    plt.xlabel("Bits")
    plt.ylabel("Max Error")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    torch.set_float32_matmul_precision('high')
    input_tensor = torch.randn(1, 3, 224, 224)

    for clamp in [False, True]:
        results = {}
        clamp_label = "with Clamp 5%" if clamp else "no Clamp"
        print(f"\n🧪 Running experiment ({clamp_label})")
        for grid_type in ['int', 'F2P_sr_h2', 'F2P_li_h2']:
            print(f"🔧 Testing grid: {grid_type}")
            bits, mean_errors, max_errors = run_quant_experiment(input_tensor, grid_type, clamp=clamp)
            results[f"{grid_type} ({clamp_label})"] = (bits, mean_errors, max_errors)

        plot_errors(results, title_suffix=f"({clamp_label})")


if __name__ == "__main__":

    main()