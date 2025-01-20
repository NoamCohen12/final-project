import copy
import json
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
import pandas as pd
from collections import defaultdict
from datetime import datetime
import sys
from typing import Dict, List, Tuple
import Quantizer  # Make sure this is in your path

# Constants
ASSETS_PATH = r"/"


def random_tensor(tensor) -> torch.Tensor:
    """
    Performs full randomization on a given tensor, including a dequantize-like step
    """
    if tensor is None or torch.isnan(tensor).any():
        print("Warning: Input tensor contains NaN values or is None")
        return tensor

    tensor_flat = tensor.cpu().numpy().flatten()

    try:
        random_quantized = np.random.uniform(low=tensor_flat.min(), high=tensor_flat.max(), size=tensor_flat.shape)
        randomized_dequantized = random_quantized + np.random.normal(loc=0, scale=0.1, size=random_quantized.shape)
        mse_error = np.mean((tensor_flat - randomized_dequantized) ** 2)

    except Exception as e:
        print(f"Randomization error: {e}")
        return tensor

    randomized_tensor = torch.from_numpy(randomized_dequantized).view(tensor.shape).to(tensor.device,
                                                                                       dtype=torch.float32)

    if torch.isnan(randomized_tensor).any():
        print("Error: NaN values detected in randomized tensor")
        return tensor

    return randomized_tensor


def getImageList(folderPath) -> list:
    """
    Gets list of image paths from a folder and its subfolders
    """
    imageList = []
    for filename in os.listdir(folderPath):
        subfolderPath = os.path.join(folderPath, filename)
        if os.path.isdir(subfolderPath):
            for image_path in os.listdir(subfolderPath):
                imageList.append(os.path.join(subfolderPath, image_path))
    return imageList


def calculate_statistics(predictions: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of predictions.
    """
    predictions = np.array(predictions)
    mean = np.mean(predictions)
    std = np.std(predictions)

    # Calculate 99% confidence interval
    confidence = 0.99
    degrees_of_freedom = len(predictions) - 1
    t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    margin_of_error = t_value * (std / np.sqrt(len(predictions)))
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def check_precision(prediction_list, prediction_list_quant) -> float:
    """
    Calculate precision for multiclass/multioutput predictions
    """
    class_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for true_label, pred_label in zip(prediction_list, prediction_list_quant):
        true_label = tuple(true_label) if isinstance(true_label, list) else (true_label,)
        pred_label = tuple(pred_label) if isinstance(pred_label, list) else (pred_label,)

        if pred_label == true_label:
            class_counts[true_label]['tp'] += 1
        else:
            class_counts[true_label]['fn'] += 1
            class_counts[pred_label]['fp'] += 1

    total_precision = 0.0
    total_classes = 0
    for class_label, counts in class_counts.items():
        tp = counts['tp']
        fp = counts['fp']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        total_precision += precision
        total_classes += 1

    avg_precision = total_precision / total_classes if total_classes > 0 else 0.0
    return avg_precision


def check_recall(prediction_list, prediction_list_quant) -> float:
    """
    Calculate recall for multiclass/multioutput predictions
    """
    class_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for true_label, pred_label in zip(prediction_list, prediction_list_quant):
        true_label = tuple(true_label) if isinstance(true_label, list) else (true_label,)
        pred_label = tuple(pred_label) if isinstance(pred_label, list) else (pred_label,)

        if pred_label == true_label:
            class_counts[true_label]['tp'] += 1
        else:
            class_counts[true_label]['fn'] += 1
            class_counts[pred_label]['fp'] += 1

    total_recall = 0.0
    total_classes = 0
    for class_label, counts in class_counts.items():
        tp = counts['tp']
        fn = counts['fn']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        total_recall += recall
        total_classes += 1

    avg_recall = total_recall / total_classes if total_classes > 0 else 0.0
    return avg_recall


def write_results_to_file(statistics: str, model_name: str) -> None:
    """
    Write results to a file with timestamp
    """
    try:
        report_directory = os.path.join(os.getcwd(), "reports")
        if not os.path.exists(report_directory):
            os.makedirs(report_directory)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"results_{model_name}_{timestamp}.txt"
        file_path = os.path.join(report_directory, file_name)

        with open(file_path, "w") as file:
            file.write(f"Results for {model_name}\n")
            file.write("--------------------------------------------\n")
            file.write(statistics)

        print(f"Results written to {file_path} successfully.")
    except Exception as e:
        print(f"Failed to write to file: {e}")


def process_layer_parameter(parameter, quantization_type, cntrSize, signed, verbose=True, flavor=""):
    """
    Process a single layer parameter for quantization
    """
    if parameter is None:
        return

    if parameter.dtype == torch.float64:
        parameter.data = parameter.data.float()

    match quantization_type:
        case "RANDOM":
            parameter.data = random_tensor(parameter.data)
        case "INT":
            grid = generate_grid(cntrSize, signed)
            parameter.data = quantize_tensor(parameter.data, grid, verbose, flavor)
        case "F2P":
            if not flavor:
                raise ValueError("Flavor is required for F2P quantization.")
            grid = Quantizer.getAllValsFxp(
                fxpSettingStr=flavor,
                cntrSize=cntrSize,
                verbose=[],
                signed=signed
            )
            parameter.data = quantize_tensor(parameter.data, grid, verbose, flavor)
        case _:
            raise ValueError(f"Invalid quantization type '{quantization_type}'")


def quantize_all_layers(model, quantization_type="RANDOM", signed=True, cntrSize=8, verbose=False,
                        flavor="") -> torch.nn.Module:
    """
    Quantize all layers of the model
    """
    print(f"Quantizing model with {quantization_type}")
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            if verbose:
                print(f"Quantizing weights for layer: {name}")
            process_layer_parameter(layer.weight, quantization_type, cntrSize, signed, verbose, flavor)
            process_layer_parameter(layer.bias, quantization_type, cntrSize, signed, verbose, flavor)
    return model


def quantize_model_switch(model, method="None", flavor="F2P_li_h2") -> torch.nn.Module:
    """
    Switch for different quantization methods
    """
    quantization_methods = {
        "None": lambda m: m,
        "INT8": lambda m: quantize_all_layers(m, quantization_type="INT", cntrSize=8),
        "INT16": lambda m: quantize_all_layers(m, quantization_type="INT", cntrSize=16),
        "RANDOM": lambda m: quantize_all_layers(m, quantization_type="RANDOM"),
    }

    if method.startswith("F2P_"):
        return quantize_all_layers(model, quantization_type="F2P", cntrSize=8, flavor=method)

    return quantization_methods.get(method, lambda m: m)(model)


def predict_image(model, image_path, device='cpu') -> tuple:
    """
    Predict image class using the model
    """
    model = model.to(device).float()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device).float()

    if torch.isnan(input_tensor).any():
        print("Error: Input tensor contains NaN values")
        return None, None, None

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    if torch.isnan(output).any():
        print("Error: Output contains NaN values")
        return None, None, None

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    if torch.isnan(probabilities).any():
        print("Error: Probabilities contain NaN values")
        return None, None, None

    class_index = torch.argmax(probabilities).item()

    with open("../../../../Desktop/final-project-main-2/Sketches-main/src/imagenet_class_index.json") as labels_file:
        labels = json.load(labels_file)

    label = labels[str(class_index)]

    return label, class_index, probabilities


def generate_grid(cntrSize, signed):
    """
    Generate quantization grid
    """
    if signed:
        return np.array(range(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1), 1))
    return np.array(range(2 ** cntrSize))


def quantize_tensor(tensor, grid, verbose=False, type=""):
    """
    Quantize and dequantize a tensor using a given grid
    """
    flattened = tensor.view(-1).numpy()
    quantized, scale, zero_point = Quantizer.quantize(flattened, grid=grid)
    if verbose:
        print(f"type {type}, Quantized: {quantized}, Scale: {scale}, Zero Point: {zero_point}")
    dequantized = Quantizer.dequantize(quantized, scale, z=zero_point)
    return torch.from_numpy(dequantized).view(tensor.shape).to(tensor.device)


def evaluate_model(model: torch.nn.Module, image_paths: List[str], device: str, quantization_method: str = "None") -> \
Dict[str, float]:
    """
    Evaluate model performance with comprehensive metrics
    """
    model = model.to(device)
    model.eval()

    predictions = []
    probabilities = []

    # Get ground truth predictions from non-quantized model if this is a quantized model
    if quantization_method != "None":
        # Create original model for ground truth
        if isinstance(model, type(resnet18())):
            original_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            original_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        original_model = original_model.to(device)
        original_model.eval()

        true_labels = []
        with torch.no_grad():
            for image_path in image_paths:
                label, class_index, _ = predict_image(original_model, image_path, device)
                if label:
                    true_labels.append(class_index)
    else:
        # For non-quantized model, use its own predictions as ground truth
        true_labels = []

    # Get predictions from the evaluated model
    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            label, class_index, probs = predict_image(model, image_path, device)
            if label:
                predictions.append(class_index)
                probabilities.append(probs.max().item())
                if quantization_method == "None":
                    true_labels.append(class_index)

    # Calculate statistics
    stats_dict = calculate_statistics(probabilities)

    # Calculate F1 score
    if len(true_labels) > 0 and len(predictions) > 0:
        try:
            f1 = f1_score(true_labels, predictions, average='weighted')
            stats_dict['f1_score'] = f1
            print(f"F1 Score for {quantization_method}: {f1:.4f}")
        except Exception as e:
            print(f"Error calculating F1 score: {e}")
            stats_dict['f1_score'] = 0.0
    else:
        print("Warning: No predictions or true labels available for F1 score calculation")
        stats_dict['f1_score'] = 0.0

    return stats_dict


def compare_models(image_paths: List[str], device: str = "cpu",
                   quantization_methods: List[str] = ["None", "INT8", "INT16", "F2P_li_h2"]) -> pd.DataFrame:
    """
    Compare ResNet18 and ResNet50 across different quantization methods
    """
    results = []

    models = {
        'ResNet18': resnet18(weights=ResNet18_Weights.DEFAULT),
        'ResNet50': resnet50(weights=ResNet50_Weights.DEFAULT)
    }

    for model_name, base_model in models.items():
        print(f"\nEvaluating {model_name}...")

        for method in quantization_methods:
            print(f"\nQuantization method: {method}")

            # Create a deep copy of the model for quantization
            if method != "None":
                model = copy.deepcopy(base_model)
                quantized_model = quantize_model_switch(model, method=method)
            else:
                quantized_model = base_model

            metrics = evaluate_model(quantized_model, image_paths, device, method)

            results.append({
                'Model': model_name,
                'Quantization': method,
                'Mean Confidence': f"{metrics['mean']:.4f}",
                'Std Dev': f"{metrics['std']:.4f}",
                'CI (99%)': f"({metrics['ci_lower']:.4f}, {metrics['ci_upper']:.4f})",
                'F1 Score': f"{metrics['f1_score']:.4f}"
            })

            print(f"Completed evaluation of {model_name} with {method} quantization")

    return pd.DataFrame(results)

def write_comparison_results(df: pd.DataFrame, filename: str = "model_comparison_results.txt"):
    """
    Write comparison results to a file
    """
    with open(filename, 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string())
        f.write("\n\n")




def generate_detailed_analysis(df: pd.DataFrame) -> None:
    """
    Generate and save detailed statistical analysis of the results
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    analysis_file = f"detailed_analysis_{timestamp}.txt"

    with open(analysis_file, 'w') as f:
        f.write("Detailed Statistical Analysis\n")
        f.write("=" * 50 + "\n\n")

        # Model comparison
        f.write("1. Model Performance Comparison\n")
        f.write("-" * 30 + "\n")
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            f.write(f"\n{model} Analysis:\n")
            f.write(f"Average Mean Confidence: {model_data['Mean Confidence'].astype(float).mean():.4f}\n")
            f.write(
                f"Best Quantization Method: {model_data.iloc[model_data['Mean Confidence'].astype(float).argmax()]['Quantization']}\n")

        # Quantization method comparison
        f.write("\n2. Quantization Method Comparison\n")
        f.write("-" * 30 + "\n")
        for quant in df['Quantization'].unique():
            quant_data = df[df['Quantization'] == quant]
            f.write(f"\n{quant} Analysis:\n")
            f.write(f"Average F1 Score: {quant_data['F1 Score'].replace('N/A', '0').astype(float).mean():.4f}\n")

        # Best configurations
        f.write("\n3. Best Configurations\n")
        f.write("-" * 30 + "\n")
        best_f1 = df.loc[df['F1 Score'].replace('N/A', '0').astype(float).idxmax()]
        best_confidence = df.loc[df['Mean Confidence'].astype(float).idxmax()]

        f.write(f"\nBest F1 Score Configuration:\n")
        f.write(f"Model: {best_f1['Model']}\n")
        f.write(f"Quantization: {best_f1['Quantization']}\n")
        f.write(f"F1 Score: {best_f1['F1 Score']}\n")

        f.write(f"\nBest Confidence Configuration:\n")
        f.write(f"Model: {best_confidence['Model']}\n")
        f.write(f"Quantization: {best_confidence['Quantization']}\n")
        f.write(f"Mean Confidence: {best_confidence['Mean Confidence']}\n")

    print(f"Detailed analysis saved to {analysis_file}")


def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print a summary of model architecture and parameters
    """
    print(f"\nModel Summary:")
    print("-" * 50)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print layer information
    print("\nLayer Information:")
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {module.__class__.__name__}, Parameters: {params:,}")


def visualize_results(df: pd.DataFrame) -> None:
    """
    Create visualizations of the results using matplotlib
    Note: This requires matplotlib to be installed
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use('seaborn')

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Mean Confidence by Model and Quantization
        sns.barplot(
            data=df,
            x='Quantization',
            y='Mean Confidence',
            hue='Model',
            ax=axes[0]
        )
        axes[0].set_title('Mean Confidence by Model and Quantization Method')
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: F1 Scores by Model and Quantization
        df_f1 = df.copy()
        df_f1['F1 Score'] = df_f1['F1 Score'].replace('N/A', '0').astype(float)
        sns.barplot(
            data=df_f1,
            x='Quantization',
            y='F1 Score',
            hue='Model',
            ax=axes[1]
        )
        axes[1].set_title('F1 Scores by Model and Quantization Method')
        axes[1].tick_params(axis='x', rotation=45)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('model_comparison_results.png')
        print("Visualization saved as model_comparison_results.png")

    except ImportError:
        print("Matplotlib and/or seaborn not installed. Skipping visualization.")

def main():
    """
    Main execution function
    """
    # Set your images path here
    images_path = r"../../../../Desktop/final-project-main-2/100 animals"  # Update this path

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Path {images_path} does not exist")

    # Get list of images
    image_list = getImageList(images_path)

    # Perform model comparison
    print("Starting comprehensive model comparison...")
    comparison_df = compare_models(
        image_list[:100],  # Limit to first 100 images for faster testing
        device="cpu",
        quantization_methods=["None", "INT8", "INT16", "F2P_li_h2", "F2P_lr_h2", "F2P_sr_h2", "F2P_si_h2", "RANDOM"]
    )

    # Print results
    print("\nComparison Results:")
    print(comparison_df)

    # Save detailed results
    write_comparison_results(comparison_df)

    # Generate additional statistical analysis
    print("\nGenerating detailed statistical analysis...")
    generate_detailed_analysis(comparison_df)

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    main()
