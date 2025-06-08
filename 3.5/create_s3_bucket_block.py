import os
import boto3
from time import sleep
from prefect_aws import S3Bucket, AwsCredentials
from pydantic import SecretStr

## IMPORTANT: Dont hardcode the credentials in the program
##            Rather use environment variables.
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']


def create_aws_creds_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=SecretStr(AWS_SECRET_ACCESS_KEY)
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)

def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="zoomcamp-mlops-prefect-bucket", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True)
    # my_s3_bucket_obj.save(name="s3-bucket-block", overwrite=True)

if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()