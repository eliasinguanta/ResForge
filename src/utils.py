#best model currently
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import pynvml
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=str, default="70")
    parser.add_argument('--batch_sizes', type=str, default="256")
    parser.add_argument('--learning_rates', type=str, default="0.01")
    parser.add_argument('--frozen_layers', type=str, default="0")
    parser.add_argument('--distillation', type=str, default='True', help='Enable distillation: True or False')
    parser.add_argument('--representative_memory', type=str, default='True', help='Enable representative memory: True or False')
    args = parser.parse_args()

    # Convert strings to lists
    args.epochs = [int(x) for x in args.epochs.split(",")]
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    args.learning_rates = [float(x) for x in args.learning_rates.split(",")]
    args.frozen_layers = [int(x) for x in args.frozen_layers.split(",")]
    args.distillation = args.distillation.lower() == 'true'
    args.representative_memory = args.representative_memory.lower() == 'true'

    return args



def log_gpu_stats(writer, epoch):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # in W

        utilization = util.gpu
        mem_used = int(mem.used / 1024**2)   # MB
        mem_total = int(mem.total / 1024**2) # MB

        writer.add_scalar(f'GPU{i}/utilization_percent', utilization, epoch)
        writer.add_scalar(f'GPU{i}/memory_used_MB', mem_used, epoch)
        writer.add_scalar(f'GPU{i}/memory_total_MB', mem_total, epoch)
        writer.add_scalar(f'GPU{i}/temperature_C', temp, epoch)
        writer.add_scalar(f'GPU{i}/power_draw_W', power, epoch)

        print(f"[SM_METRIC]gpu{i}_utilization_percent={utilization}")
        print(f"[SM_METRIC]gpu{i}_memory_used_MB={mem_used}")
        print(f"[SM_METRIC]gpu{i}_memory_total_MB={mem_total}")
        print(f"[SM_METRIC]gpu{i}_temperature_C={temp}")
        print(f"[SM_METRIC]gpu{i}_power_draw_W={power:.2f}")

    pynvml.nvmlShutdown()

def get_training_metrics(epoch, all_labels, all_preds, running_loss, running_ce_loss, running_kd_loss, train_emissions, cum_train_emissions, dataset_size):
    if torch.cuda.is_available():
        # Optional: Du kannst hier gpu stats sammeln und zurückgeben, falls nötig
        pass

    train_precision = precision_score(all_labels, all_preds, average='macro')
    train_recall = recall_score(all_labels, all_preds, average='macro')
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    epoch_loss = running_loss / dataset_size
    epoch_ce_loss = running_ce_loss / dataset_size
    epoch_kd_loss = running_kd_loss / dataset_size

    return {
        "epoch": epoch,
        "train_loss": epoch_loss,
        "train_ce_loss": epoch_ce_loss,
        "train_kd_loss": epoch_kd_loss,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_co2": train_emissions,
        "train_co2_cum": cum_train_emissions
    }


def get_test_metrics(epoch, all_labels, all_preds,
                all_labels_old, all_preds_old,
                all_labels_new, all_preds_new,
                test_emissions, cum_test_emissions, train_emissions):

    total = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100 * correct / total

    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    test_f1 = f1_score(all_labels, all_preds, average='macro')

    acc_old = None
    acc_new = None

    if all_labels_old:
        acc_old = 100 * sum(p == l for p, l in zip(all_preds_old, all_labels_old)) / len(all_labels_old)

    if all_labels_new:
        acc_new = 100 * sum(p == l for p, l in zip(all_preds_new, all_labels_new)) / len(all_labels_new)

    return {
        "epoch": epoch,
        "test_accuracy": accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_acc_old": acc_old,
        "test_acc_new": acc_new,
        "test_co2": test_emissions,
        "test_co2_cum": cum_test_emissions,
        "test_co2_train_cum": train_emissions
    }

def prepare_data(train_dataset, test_dataset, num_classes, new_classes, exemplar_set, batch_size, representive_memory):
    # New classes
    new_class_range = range(num_classes - new_classes, num_classes)
    new_class_indices = [i for i, (_, label) in enumerate(train_dataset) if label in new_class_range]

    # Add exemplars
    exemplar_indices = []
    if representive_memory:
        for indices in exemplar_set.values():
            exemplar_indices.extend(indices)

    # Combine class indicies and delete duplicates
    train_indices = list(set(new_class_indices + exemplar_indices))
    train_subset = Subset(train_dataset, train_indices)

    # Test classes (new and old classes)
    test_class_range = range(0, num_classes)
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in test_class_range]
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# extract model output (without fc layer)
def extract_features(model, images):
    x = images
    for name, module in model.named_children():
        if name == "fc":
            break
        x = module(x)
    x = torch.flatten(x, 1)
    x = F.normalize(x, p=2, dim=1)
    return x