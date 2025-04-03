import boto3
import os

s3 = boto3.client('s3')

def create_bucket(bucket_name):
    res = s3.list_buckets()
    existing_buckets = [buck["Name"] for buck in res["Buckets"]]
    if bucket_name in existing_buckets:
        print(f"{bucket_name} already exists in your account. ")
        return
    s3.create_bucket(Bucket=bucket_name)
    print(f"Bucket {bucket_name} is created")


def upload_dir(bucket_name, directory_path, s3_prefix):
    for root, dirs, files in os.walk(directory_path):
        # print(root, dirs, files)
        # print()
        for file in files:
            file_path = os.path.join(root, file).replace("\\", "/")
            print("file_path", file_path)
            relpath = os.path.relpath(file_path, directory_path) # relative path
            print("relpath", relpath)
            s3_key = os.path.join(s3_prefix, relpath).replace("\\", "/")
            print("s3_key", s3_key)
            
            s3.upload_file(file_path, bucket_name, s3_key)
            print()


def download_dir(bucket_name, local_path, s3_prefix):
    # os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                print("s3_key", s3_key)

                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                print("local_file", local_file)

                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file)
                print()


def upload_image_to_s3(bucket_name, file_name, s3_prefix="ml-images", object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)
    
    full_object_name = os.path.join(s3_prefix, object_name).replace("\\", "/")
    print("full_object_name", full_object_name)
    s3.upload_file(file_name, bucket_name, full_object_name)

    # get url of the uploaded image
    response = s3.generate_presigned_url("get_object", 
                                         Params={
                                             "Bucket": bucket_name, 
                                             "Key": full_object_name
                                         }, 
                                         ExpiresIn=3600 # the image url will be expiring in 1 hour
                                         )
    return response