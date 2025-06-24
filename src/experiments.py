
import os
import torch
import numpy as np
import random
import csv
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from trainer import IncrementalTrainer
from itertools import product


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def grid_search(args, seed=123, steps=5):
    # Get output directory from environment or use default /tmp/output
    log_base = os.environ.get('SM_OUTPUT_DATA_DIR', '/tmp/output')
    grid_results = {}
    # Set global random seed for reproducibility
    set_global_seed(seed)

    # Iterate over all combinations of epochs, learning rates, batch sizes, and frozen layers
    for epochs, lr, bs, freeze in product(args.epochs, args.learning_rates, args.batch_sizes, args.frozen_layers):
        # Create a unique combintion name for logging
        combo_name = f"ep_{epochs}_lr_{lr}_bs_{bs}_frozen_{freeze}"
        if args.representative_memory:
            combo_name += "_rm"
        if args.distillation:
            combo_name += "_dl"

        print(f"Benchmarking: {combo_name}")

        # Tensorboard setup (logging)
        writer = SummaryWriter(log_dir=os.path.join(log_base, "tensorboard", f"{combo_name}"))

        # Initialize the trainer with current hyperparameters
        trainer = IncrementalTrainer(
            epochs=epochs,
            learning_rate=lr,
            batch_size=bs,
            frozen_layers=freeze,
            pretrained=True, # we have decided to keep it hardcoded
            representative_memory=args.representative_memory,
            distillation=args.distillation
        )

        all_test_metrics = defaultdict(list)
        all_train_metrics = defaultdict(list)
        epoch_offset = 0 #TODO: can be deleted the trainer keeps globaly track of the epoch

        # Run incremental training steps
        for step in range(steps):
            if step > 0:
                trainer.expand_model(10)

            # Train the modell on the new classes and collect the metrics
            train_metrics, test_metrics = trainer.train()

            # Logging (Tensorboard)
            for i, (train_m, test_m) in enumerate(zip(train_metrics, test_metrics)):
                epoch = epoch_offset + i 
                all_train_metrics[epoch].append(train_m)
                all_test_metrics[epoch].append(test_m)
                for key, val in train_m.items():
                    if val is not None:
                        writer.add_scalar(f"train/{key}", val, epoch)
                for key, val in test_m.items():
                    if val is not None:
                        writer.add_scalar(f"test/{key}", val, epoch)

            # Update epoch offset for next incremental step
            epoch_offset += len(test_metrics)

        # Save all collected metrics to a CSV file for this combination
        csv_path = os.path.join(log_base, f"metrics_{combo_name}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            # Collect all metric keys across epochs for CSV columns
            all_keys = set()
            for epoch in all_train_metrics:
                all_keys.update(all_train_metrics[epoch][0].keys())
                all_keys.update({f"test_{k}" for k in all_test_metrics[epoch][0].keys()})
            fieldnames = ['epoch'] + sorted(all_keys)
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer_csv.writeheader()

            # Write row per epoch with train and test metrics
            for epoch in sorted(all_train_metrics.keys()):
                row = {'epoch': epoch}
                for k, v in all_train_metrics[epoch][0].items():
                    row[k] = v
                for k, v in all_test_metrics[epoch][0].items():
                    row[f"test_{k}"] = v
                writer_csv.writerow(row)

        writer.flush()
        writer.close()

        #store the model
        torch.save(trainer.model.state_dict(), os.path.join(log_base, f"model_{combo_name}.pth"))

        grid_results[combo_name] = all_test_metrics
    return grid_results


