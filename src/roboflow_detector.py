"""Utility for running Roboflow inference on a single image."""
from __future__ import annotations

import argparse
import os

from inference_sdk import InferenceHTTPClient

API_URL = "https://serverless.roboflow.com"
DEFAULT_MODEL = "people_basketball_hoops/4"
DEFAULT_API_KEY = "lqmyOyZUfklTXFZjOa6R"


def get_client(api_key: str | None = None) -> InferenceHTTPClient:
    """Create an inference client with the provided or environment API key."""
    key = api_key or os.getenv("ROBOFLOW_API_KEY", DEFAULT_API_KEY)
    return InferenceHTTPClient(api_url=API_URL, api_key=key)


def infer_image(image_path: str, model_id: str = DEFAULT_MODEL, api_key: str | None = None) -> dict:
    """Run inference on ``image_path`` and return the raw JSON result."""
    client = get_client(api_key)
    return client.infer(image_path, model_id=model_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Roboflow inference on an image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="Roboflow model ID")
    parser.add_argument("--api-key", help="Override API key (or use ROBOFLOW_API_KEY env)")
    args = parser.parse_args()

    result = infer_image(args.image_path, model_id=args.model_id, api_key=args.api_key)
    print(result)


if __name__ == "__main__":
    main()
