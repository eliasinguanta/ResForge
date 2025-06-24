terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "eu-central-1" 
}

variable "aws_region" {
  type    = string
  default = "eu-central-1"
}


# S3 Bucket for datasets and models
resource "aws_s3_bucket" "resforge_bucket" {
  bucket = "resforge-${random_string.suffix.result}"

  tags = {
    Name        = "ResForge Data Bucket"
    Environment = "development"
  }
}

# Random suffix for bucket name
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 Bucket versioning
resource "aws_s3_bucket_versioning" "resforge_versioning" {
  bucket = aws_s3_bucket.resforge_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# IAM Role for training
resource "aws_iam_role" "training_role" {
  name = "resforge-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_full_access" {
  role       = aws_iam_role.training_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# IAM Policy for training
resource "aws_iam_role_policy" "training_policy" {
  name = "resforge-training-policy"
  role = aws_iam_role.training_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.resforge_bucket.arn,
          "${aws_s3_bucket.resforge_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "sagemaker:*",
          "servicecatalog:*",
          "codecommit:*",
          "codebuild:*",
          "codepipeline:*",
          "iam:PassRole",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:DescribeLogGroups",
          "sagemaker:listSpaces",
          "sagemaker:listApps"
        ],
        Resource = "*"
      }
    ]
  })
}





# Default VPC
data "aws_vpc" "default" {
  default = true
}

# Default Subnets (z. B. public)
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}


resource "aws_sagemaker_domain" "studio" {
  domain_name = "resforge-studio-domain"
  auth_mode   = "IAM"

  vpc_id     = data.aws_vpc.default.id
  subnet_ids = data.aws_subnets.default.ids

  default_user_settings {
    execution_role = aws_iam_role.training_role.arn
  }

  app_network_access_type = "PublicInternetOnly"
}


resource "aws_sagemaker_user_profile" "studio_user" {
  domain_id = aws_sagemaker_domain.studio.id
  user_profile_name = "resforge-elias"

  user_settings {
    execution_role = aws_iam_role.training_role.arn
  }
}


# Output the bucket name and role ARN
output "bucket_name" {
  value = aws_s3_bucket.resforge_bucket.id
}

output "training_role_arn" {
  value = aws_iam_role.training_role.arn
}

output "studio_domain_url" {
  value = "https://${aws_sagemaker_domain.studio.home_efs_file_system_id}.studio.eu-central-1.sagemaker.aws/tree"
}

