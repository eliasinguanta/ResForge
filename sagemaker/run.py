import sagemaker
from sagemaker.pytorch import PyTorch
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, required=True)
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--max_runtime_hours', type=float, default=10.9)
    return parser.parse_args()

def main():
    args = parse_args()

    print("Starte SageMaker Session...")
    session = sagemaker.Session()

    # 'ml.g4dn.xlarge'
    estimator = PyTorch(
        entry_point='main.py',
        source_dir='src',
        role=args.role,
        framework_version='2.1.0',
        py_version='py310',
        instance_type='ml.g4dn.xlarge',
        instance_count=1,
        max_run=int(args.max_runtime_hours * 3600),
        output_path=f's3://{args.bucket}/models',
        output_data_dir=f's3://{args.bucket}/models/training-{int(time.time())}',
        hyperparameters = {
            'epochs': '70',
            'batch_sizes': '256',
            'learning_rates': '0.05',
            'frozen_layers': '0,2,4,6',
            'distillation': 'True',
            'representative_memory': 'True',
        },
        sagemaker_session=session,
    )

    print("Starte Training...")
    estimator.fit()
    print("Training beendet.")

if __name__ == '__main__':
    main()

#python3 scripts/train_sagemaker.py \
#  --role arn:aws:iam::386757133985:role/resforge-training-role \
#  --bucket resforge-k02l28nu