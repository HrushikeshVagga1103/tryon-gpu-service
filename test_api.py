"""
Test script for CatVTON API
"""
import requests
import json
import os
from typing import Optional


class CatVTONTester:
    """Test client for CatVTON API."""
    
    def __init__(self, base_url: str, api_key: str):
        """
        Initialize test client.
        
        Args:
            base_url: Base URL of the API (e.g., 'http://localhost:8000' or 'https://xxxxx-8000.proxy.runpod.net')
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> dict:
        """Check service health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def root(self) -> dict:
        """Get root endpoint."""
        response = requests.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    def try_on(
        self,
        person_image_uri: str,
        garment_image_uri: str,
        output_uri: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8
    ) -> dict:
        """
        Test try-on endpoint.
        
        Args:
            person_image_uri: GCS URI for person image
            garment_image_uri: GCS URI for garment image
            output_uri: Optional output GCS URI
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
        
        Returns:
            Response JSON
        """
        payload = {
            "person_image_uri": person_image_uri,
            "garment_image_uri": garment_image_uri,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength
        }
        
        if output_uri:
            payload["output_uri"] = output_uri
        
        response = requests.post(
            f"{self.base_url}/tryon",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CatVTON API")
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("API_KEY"),
        required=False,
        help="API key (or set API_KEY env var)"
    )
    parser.add_argument(
        "--person-uri",
        type=str,
        required=True,
        help="GCS URI for person image"
    )
    parser.add_argument(
        "--garment-uri",
        type=str,
        required=True,
        help="GCS URI for garment image"
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        default=None,
        help="Optional output GCS URI"
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only run health check"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API key is required. Set --api-key or API_KEY env var.")
        return
    
    tester = CatVTONTester(args.url, args.api_key)
    
    print("=" * 60)
    print("CatVTON API Test")
    print("=" * 60)
    
    # Health check
    print("\n1. Health Check...")
    try:
        health = tester.health_check()
        print(f"✓ Health: {json.dumps(health, indent=2)}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return
    
    if args.health_only:
        return
    
    # Root endpoint
    print("\n2. Root Endpoint...")
    try:
        root = tester.root()
        print(f"✓ Root: {json.dumps(root, indent=2)}")
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
    
    # Try-on
    print("\n3. Try-On Request...")
    print(f"   Person: {args.person_uri}")
    print(f"   Garment: {args.garment_uri}")
    if args.output_uri:
        print(f"   Output: {args.output_uri}")
    
    try:
        result = tester.try_on(
            person_image_uri=args.person_uri,
            garment_image_uri=args.garment_uri,
            output_uri=args.output_uri
        )
        print(f"✓ Try-on successful!")
        print(f"   Result: {json.dumps(result, indent=2)}")
    except requests.exceptions.HTTPError as e:
        print(f"✗ Try-on failed: {e}")
        if e.response.text:
            print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"✗ Try-on failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

