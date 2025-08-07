# Code by Jonathan Ahern (jahern@ucsd.edu) Last updated 2025-09-07
# Code included here was made with the help of ChatGPT and GitHub Copilot.

# import libraries
from urllib.parse import urlparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import requests
import pandas as pd
import nibabel as nib
import io
import gzip

# find the latest version of a given OpenNeuro datset
def get_latest_snapshot(dataset_id: str):
    """
    Fetch the latest snapshot tag for a given OpenNeuro dataset ID.

    Parameters
    ----------
    dataset_id : str
        The OpenNeuro dataset ID (e.g., "ds000001").
    
    Returns
    -------
    str
        The latest snapshot tag (version) for the dataset.

    """
    graphql_url = "https://openneuro.org/crn/graphql" # GraphQL endpoint for OpenNeuro
    
    # GraphQL query to fetch dataset snapshots
    query = """
      query ($id: ID!) {
        dataset(id: $id) {
          snapshots {
            tag
          }
        }
      }
    """

    # Make the GraphQL request to fetch snapshots
    res = requests.post(graphql_url, json={"query": query, "variables": {"id": dataset_id}})
    res.raise_for_status() # Check for HTTP errors
    snaps = res.json()["data"]["dataset"]["snapshots"] # Extract snapshots
    if not snaps: # Check if there are any snapshots
        raise Exception("No snapshots found.")
    latest = sorted([s["tag"] for s in snaps], reverse=True)[0] # Get the latest tag
    return latest

# get the list of JSON files in a given OpenNeuro dataset
def list_s3_json_files(dataset_id):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    prefix = f"{dataset_id}/"

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    json_files = {}

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json'):
                url = f"https://s3.amazonaws.com/{bucket}/{key}"
                json_files[key] = url

    print(f"Found {len(json_files)} JSON files.")
    for example in list(json_files.keys())[:10]:
        print(" -", example)

    return json_files


# flatten json objects into a single-level dictionary
def flatten_json(y):
    """
    Flatten a nested JSON object into a single-level dictionary with dot notation keys.

    Parameters
    ----------
    y : dict
        The nested JSON object to flatten.
    
    Returns
    -------
    dict
        A flattened dictionary with keys in dot notation.
    """

    out = {} # Initialize output dictionary

    # Recursive function to flatten the JSON
    def _flatten(x, name=""):
        if isinstance(x, dict):
            for k,v in x.items():
                _flatten(v, name + k + ".")
        elif isinstance(x, list):
            for i,v in enumerate(x):
                _flatten(v, name + str(i) + ".")
        else:
            out[name[:-1]] = x
    _flatten(y)

    return out

# make a Pandas DataFrame from the data in JSON files in an OpenNeuro dataset
def jsons_to_dataframe(openneuro_url: str):
    """
    Load JSON files from an OpenNeuro dataset URL into a Pandas DataFrame.

    Parameters
    ----------
    openneuro_url : str
        The OpenNeuro dataset URL (e.g., "https://openneuro.org/datasets/ds000001/versions/1.0.0").
    
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the flattened JSON data, indexed by file name.
    """

    parsed = urlparse(openneuro_url)
    parts = parsed.path.strip("/").split("/")
    dataset_id = parts[1]
    # version is not used for S3 listing, so no need to parse it here

    print(f"Listing JSON files for dataset: {dataset_id} from S3 (no versioning)")

    # Call your S3 JSON file lister WITHOUT version arg
    json_urls = list_s3_json_files(dataset_id)  # <-- Removed version argument here

    print(f"Found {len(json_urls)} JSON files.")

    records = []
    for fname, url in tqdm(json_urls.items(), desc="Downloading JSONs"):
        r = requests.get(url)
        r.raise_for_status()
        flat = flatten_json(r.json())
        flat["__file__"] = fname
        records.append(flat)

    df = pd.DataFrame(records).set_index("__file__")
    return df

# list all .nii.gz files in a given OpenNeuro dataset (via public S3)
def list_s3_niigz_files(dataset_id):
    """
    List all .nii.gz files in a given OpenNeuro dataset (via public S3).

    Parameters
    ----------
    dataset_id : str
        The OpenNeuro dataset ID (e.g., "ds000001").
    
    Returns
    -------
    dict
        A dictionary mapping file paths to their download URLs.
    """
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    prefix = f"{dataset_id}/"

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    niigz_files = {}

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.nii.gz'):
                url = f"https://s3.amazonaws.com/{bucket}/{key}"
                niigz_files[key] = url

    print(f"Found {len(niigz_files)} NIfTI files (.nii.gz).")
    for example in list(niigz_files.keys())[:10]:
        print(" -", example)

    return niigz_files

def get_nii_shape_from_url(url):
    """
    Fetch a NIfTI file from a URL, decompress it if gzipped, and return its shape.

    Parameters
    ----------
    url : str
        The s3 URL of the NIfTI file (can be gzipped).

    Returns
    -------
    tuple
        The shape of the NIfTI image (dimensions).
    """
    response = requests.get(url)
    response.raise_for_status()

    # Decompress gzip content
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
        decompressed = gz.read()

    # Use FileHolder with decompressed bytes for header and image
    file_holder = nib.FileHolder(fileobj=io.BytesIO(decompressed))
    img = nib.Nifti1Image.from_file_map({'header': file_holder, 'image': file_holder})

    return img.shape