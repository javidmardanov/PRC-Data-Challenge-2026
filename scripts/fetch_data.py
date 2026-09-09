"""Download PRC Data Challenge objects from the OpenSky S3 endpoint.

Credentials are read from the repository-root ``credentials.json`` and are never
printed. Existing files with the expected size are skipped, so reruns are safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIALS = ROOT / "credentials.json"
DEFAULT_OUTPUT = ROOT / "data" / "raw"


def endpoint_from_console_url(url: str) -> str:
    """Convert the MinIO web-console host to its S3 API host when necessary."""
    parsed = urlsplit(url)
    host = parsed.netloc.replace("s3-console.opensky-network.org", "s3.opensky-network.org")
    return urlunsplit((parsed.scheme or "https", host, "", "", "")).rstrip("/")


def client(credentials_path: Path):
    values = json.loads(credentials_path.read_text(encoding="utf-8"))
    return boto3.client(
        "s3",
        endpoint_url=endpoint_from_console_url(values["url"]),
        aws_access_key_id=values["accessKey"],
        aws_secret_access_key=values["secretKey"],
        config=Config(signature_version="s3v4", s3={"addressing_style": values.get("path", "auto")}),
        region_name="us-east-1",
    )


def list_objects(s3, bucket: str) -> list[dict]:
    paginator = s3.get_paginator("list_objects_v2")
    return [obj for page in paginator.paginate(Bucket=bucket) for obj in page.get("Contents", [])]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bucket", help="Bucket to inventory/download; defaults to all visible buckets")
    parser.add_argument("--download", action="store_true", help="Download every object in the selected bucket(s)")
    parser.add_argument("--manifest", type=Path, help="Write a JSON manifest (including local SHA-256 after download)")
    args = parser.parse_args()

    endpoint = endpoint_from_console_url(json.loads(args.credentials.read_text(encoding="utf-8"))["url"])
    s3 = client(args.credentials)
    buckets = [args.bucket] if args.bucket else [item["Name"] for item in s3.list_buckets().get("Buckets", [])]
    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_endpoint": endpoint,
        "buckets": {},
    }
    transfer = TransferConfig(max_concurrency=8, multipart_threshold=16 * 1024 * 1024)

    for bucket in buckets:
        objects = list_objects(s3, bucket)
        manifest["buckets"][bucket] = []
        print(f"{bucket}/ ({len(objects)} objects)")
        for obj in objects:
            key = obj["Key"]
            size = int(obj["Size"])
            print(f"  {size:>12,}  {key}")
            record = {
                "key": key,
                "size_bytes": size,
                "etag": str(obj.get("ETag", "")).strip('"'),
                "last_modified": obj["LastModified"].isoformat(),
            }
            if args.download:
                destination = args.output / bucket / Path(key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() or destination.stat().st_size != size:
                    s3.download_file(bucket, key, str(destination), Config=transfer)
                record["local_path"] = str(destination.relative_to(ROOT))
                record["sha256"] = sha256(destination)
            manifest["buckets"][bucket].append(record)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
