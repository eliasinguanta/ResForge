import os
import boto3
import subprocess
import tarfile
import argparse

def extract_tar_gz_files(local_dir):
    # Walk through all directories and files in the given local directory
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            # Check if the file is a .tar.gz archive
            if file.endswith('.tar.gz'):
                file_path = os.path.join(root, file)
                print(f"Extracting archive: {file_path}")
                
                # Open and extract the contents of the .tar.gz file
                with tarfile.open(file_path, "r:gz") as tar:
                    tar.extractall(path=root)


def download_tensorboard_logs(bucket_name, s3_prefix, local_dir="./tensorboard_logs"):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    # Iterate over all objects in the specified S3 prefix
    for obj in bucket.objects.filter(Prefix=s3_prefix):
        # Skip "folders" (S3 keys that end with '/')
        if obj.key.endswith('/'):
            continue

        # Determine the local file path where the S3 object will be saved
        target = os.path.join(local_dir, obj.key[len(s3_prefix):].lstrip('/'))

        # Create the necessary local directories if they don't exist
        os.makedirs(os.path.dirname(target), exist_ok=True)

        # Download the file from S3 to the local target path
        bucket.download_file(obj.key, target)


def show():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download and launch TensorBoard from S3 logs")
    parser.add_argument("--prefix", required=True, help="S3 prefix to download logs from")
    parser.add_argument("--bucket", default="resforge-k02l28nu", help="S3 bucket name")
    parser.add_argument("--local_dir", default="./tensorboard_logs", help="Local directory to store logs")
    args = parser.parse_args()

    # Ensure the local directory exists
    os.makedirs(args.local_dir, exist_ok=True)

    # Download TensorBoard logs from the specified S3 bucket and prefix
    download_tensorboard_logs(args.bucket, args.prefix, args.local_dir)

    # Extract all .tar.gz archives (if any) in the downloaded logs
    extract_tar_gz_files(args.local_dir)

    # Start TensorBoard using the downloaded log directory
    subprocess.run(['tensorboard', '--logdir', args.local_dir, '--host', '0.0.0.0'])

    
if __name__ == "__main__":
    show()
