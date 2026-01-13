"""
GCS utility functions for downloading and uploading images.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from google.cloud import storage
from google.oauth2 import service_account


class GCSUtils:
    """Helper class for Google Cloud Storage operations."""
    
    def __init__(self, service_account_path: Optional[str] = None):
        """
        Initialize GCS client.
        
        Args:
            service_account_path: Path to GCS service account JSON file.
                                 If None, uses default credentials.
        """
        if service_account_path and os.path.exists(service_account_path):
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path
            )
            self.client = storage.Client(credentials=credentials)
        else:
            # Use default credentials (e.g., from environment)
            self.client = storage.Client()
    
    def parse_gcs_uri(self, gcs_uri: str) -> Tuple[str, str]:
        """
        Parse GCS URI into bucket and blob name.
        
        Args:
            gcs_uri: GCS URI in format gs://bucket-name/path/to/file
            
        Returns:
            Tuple of (bucket_name, blob_name)
        """
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}. Must start with 'gs://'")
        
        uri_parts = gcs_uri[5:].split("/", 1)
        bucket_name = uri_parts[0]
        blob_name = uri_parts[1] if len(uri_parts) > 1 else ""
        
        return bucket_name, blob_name
    
    def download_file(self, gcs_uri: str, local_path: str) -> str:
        """
        Download a file from GCS to local path.
        
        Args:
            gcs_uri: GCS URI of the file to download
            local_path: Local file path to save the file
            
        Returns:
            Local file path
        """
        bucket_name, blob_name = self.parse_gcs_uri(gcs_uri)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        blob.download_to_filename(local_path)
        return local_path
    
    def upload_file(self, local_path: str, gcs_uri: str, content_type: str = "image/png") -> str:
        """
        Upload a local file to GCS.
        
        Args:
            local_path: Local file path to upload
            gcs_uri: GCS URI where the file should be uploaded
            content_type: MIME type of the file
            
        Returns:
            GCS URI of the uploaded file
        """
        bucket_name, blob_name = self.parse_gcs_uri(gcs_uri)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_path, content_type=content_type)
        return gcs_uri
    
    def generate_output_uri(self, input_uri: str, suffix: str = "_tryon") -> str:
        """
        Generate an output GCS URI based on input URI.
        
        Args:
            input_uri: Input GCS URI
            suffix: Suffix to add before file extension
            
        Returns:
            Generated output GCS URI
        """
        bucket_name, blob_name = self.parse_gcs_uri(input_uri)
        
        # Add suffix before file extension
        path_parts = blob_name.rsplit(".", 1)
        if len(path_parts) == 2:
            new_blob_name = f"{path_parts[0]}{suffix}.{path_parts[1]}"
        else:
            new_blob_name = f"{blob_name}{suffix}"
        
        return f"gs://{bucket_name}/{new_blob_name}"

