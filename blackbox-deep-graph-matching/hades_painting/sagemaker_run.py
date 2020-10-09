from sagemaker.pytorch import PyTorch


def main():
    instance_type = "ml.p3.2xlarge"
    train_data_path = "s3://rnd-ocr/tyler/painting_data"

    output_path = "s3://rnd-ocr/tyler/painting/output"
    code_location = "s3://rnd-ocr/tyler/painting/source"
    checkpoint_location = "s3://rnd-ocr/tyler/painting/checkpoint_cosine/add_noise"
    role = "arn:aws:iam::533155507761:role/service-role/AmazonSageMaker-ExecutionRole-20190312T160681"
    source_dir = "."

    pytorch_estimator = PyTorch(
        entry_point="sagemaker_entry.py",
        source_dir=source_dir,
        code_location=code_location,
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_location,
        role=role,
        train_instance_type=instance_type,
        train_instance_count=1,
        train_volume_size=500,
        base_job_name="tyler-sagemaker-geek",
        train_max_run=10*86400,
        framework_version="1.1.0",
        py_version="py3",
        train_use_spot_instances=True,
        train_max_wait=11*86400,
        hyperparameters={"config": "sagemaker_config.ini"})
    pytorch_estimator.fit({"train": train_data_path})


if __name__ == "__main__":
    main()
