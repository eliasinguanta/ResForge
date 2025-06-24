# ResForge Terraform Configuration

This Terraform configuration creates the essential AWS resources needed for training the ResNet18 model.

## Created Resources

- S3 Bucket for datasets and models
- IAM Role and Policy for training
- Basic permissions for SageMaker training jobs

## Prerequisites

1. AWS CLI installed and configured
2. AWS Credentials with sufficient permissions
3. Terraform installed

## Installation

1. Configure AWS CLI:
```bash
aws configure
```

2. Initialize Terraform:
```bash
terraform init
```

3. Create Terraform plan:
```bash
terraform plan
```

4. Deploy infrastructure:
```bash
terraform apply
```

## Using the Infrastructure

After deployment, Terraform will output:
- The S3 bucket name where you can upload your datasets
- The IAM role ARN to use in your training configuration

## Destroy Infrastructure

To remove all created resources:
```bash
terraform destroy
```

## Important Notes

- S3 Bucket has to be empty to delete it