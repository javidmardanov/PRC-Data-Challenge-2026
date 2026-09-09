"""Validate and upload one versioned prediction file to the challenge team bucket."""
import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from check_solution import check_submission
from fetch_data import client, DEFAULT_CREDENTIALS, list_objects, sha256

BUCKET='prc-2026-elegant-alligator'


def recent_submissions(objects, cutoff):
    # The scorer writes *_result.json and *_persist.json beside each submission.
    # These generated receipts are storage objects, not additional submissions.
    return sum(o['Key'].lower().endswith('.parquet') and o['LastModified']>=cutoff
               for o in objects)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('file',type=Path)
    p.add_argument('--upload',action='store_true')
    args=p.parse_args()
    version=re.fullmatch(r'elegant-alligator_v([1-9]\d*)\.parquet',args.file.name)
    if not version:
        raise ValueError('Use elegant-alligator_vN.parquet, N a positive integer')
    check_submission(args.file)
    digest=sha256(args.file)
    if not args.upload:
        print('Validated locally; add --upload to submit. SHA256:',digest)
        return
    s3=client(DEFAULT_CREDENTIALS)
    objects=list_objects(s3,BUCKET)
    if any(o['Key']==args.file.name for o in objects):
        raise ValueError('This version already exists; no overwrite or duplicate upload')
    previous=[int(m.group(1)) for o in objects
        if (m:=re.fullmatch(r'elegant-alligator_v([1-9]\d*)\.parquet',o['Key']))]
    if previous and int(version.group(1)) <= max(previous):
        raise ValueError('Use a version number greater than every existing submission')
    # A rolling 24h cap is conservative when the server's reset timezone is unclear.
    cutoff=datetime.now(timezone.utc)-timedelta(days=1)
    if recent_submissions(objects, cutoff)>=5:
        raise ValueError('Five uploads already present from the last 24 hours')
    if sum(o['Size'] for o in objects)+args.file.stat().st_size > 1_000_000_000:
        raise ValueError('Upload would exceed the 1 GB bucket limit')
    with args.file.open('rb') as stream:
        s3.put_object(Bucket=BUCKET,Key=args.file.name,Body=stream,IfNoneMatch='*',
            ContentType='application/octet-stream',Metadata={'sha256':digest})
    uploaded=s3.head_object(Bucket=BUCKET,Key=args.file.name)
    if uploaded['ContentLength']!=args.file.stat().st_size or uploaded.get('Metadata',{}).get('sha256')!=digest:
        raise IOError('Uploaded size or stored SHA-256 metadata differs from the local submission')
    record={'bucket':BUCKET,'key':args.file.name,'sha256':digest,
        'uploaded_at':datetime.now(timezone.utc).isoformat(),'bytes':uploaded['ContentLength']}
    args.file.with_suffix('.upload.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
