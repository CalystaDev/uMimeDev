from google.cloud import storage
import os

# Ensure the environment variable is set
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not credentials_path:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set.")

# Try to connect to Google Cloud Storage
storage_client = storage.Client()
buckets = list(storage_client.list_buckets())
print("Buckets in your project:")
for bucket in buckets:
    print(bucket.name)
