#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timedelta, UTC
from functools import lru_cache
from pathlib import Path
from typing import IO

from tqdm import tqdm

from iqdbc.car.tests.routes import routes as test_car_models_routes
from iqpilot.selfdrive.test.process_replay.test_processes import source_segments as replay_segments

TOKEN_PATH = Path("/data/azure_token")


@lru_cache
def get_azure_credential():
  if "AZURE_TOKEN" in os.environ:
    return os.environ["AZURE_TOKEN"]
  if TOKEN_PATH.is_file():
    return TOKEN_PATH.read_text().strip()
  from azure.identity import AzureCliCredential
  return AzureCliCredential()


@lru_cache
def get_container_sas(account_name: str, container_name: str):
  from azure.storage.blob import BlobServiceClient, ContainerSasPermissions, generate_container_sas
  start_time = datetime.now(UTC).replace(tzinfo=None)
  expiry_time = start_time + timedelta(hours=1)
  blob_service = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=get_azure_credential())
  return generate_container_sas(account_name, container_name,
                                user_delegation_key=blob_service.get_user_delegation_key(start_time, expiry_time),
                                permission=ContainerSasPermissions(read=True, write=True, list=True), expiry=expiry_time)


class AzureContainer:
  def __init__(self, account, container):
    self.ACCOUNT = account
    self.CONTAINER = container

  @property
  def ACCOUNT_URL(self) -> str:
    return f"https://{self.ACCOUNT}.blob.core.windows.net"

  @property
  def BASE_URL(self) -> str:
    return f"{self.ACCOUNT_URL}/{self.CONTAINER}/"

  def get_client_and_key(self):
    from azure.storage.blob import ContainerClient
    return ContainerClient(self.ACCOUNT_URL, self.CONTAINER, credential=get_azure_credential()), get_container_sas(self.ACCOUNT, self.CONTAINER)

  def upload_bytes(self, data: bytes | IO, blob_name: str, overwrite=False) -> str:
    from azure.storage.blob import BlobClient
    client = BlobClient(account_url=self.ACCOUNT_URL, container_name=self.CONTAINER, blob_name=blob_name, credential=get_azure_credential())
    client.upload_blob(data, overwrite=overwrite)
    return self.BASE_URL + blob_name

  def upload_file(self, path: str | os.PathLike, blob_name: str, overwrite=False) -> str:
    with open(path, "rb") as f:
      return self.upload_bytes(f, blob_name, overwrite)


DataCIContainer = AzureContainer("commadataci", "commadataci")
DataProdContainer = AzureContainer("commadata2", "commadata2")
OpenpilotCIContainer = AzureContainer("commadataci", "openpilotci")

SOURCES: list[AzureContainer] = [
  DataProdContainer,
  DataCIContainer
]

DEST = OpenpilotCIContainer

def upload_route(path: str, exclude_patterns: Iterable[str] | None = None) -> None:
  if exclude_patterns is None:
    exclude_patterns = [r'dcamera\.hevc']

  r, n = path.rsplit("--", 1)
  r = '/'.join(r.split('/')[-2:])  # strip out anything extra in the path
  destpath = f"{r}/{n}"
  for file in os.listdir(path):
    if any(re.search(pattern, file) for pattern in exclude_patterns):
      continue
    DEST.upload_file(os.path.join(path, file), f"{destpath}/{file}")


def sync_to_ci_public(route: str) -> bool:
  dest_container, dest_key = DEST.get_client_and_key()
  key_prefix = route.replace('|', '/')
  dongle_id = key_prefix.split('/')[0]

  if next(dest_container.list_blob_names(name_starts_with=key_prefix), None) is not None:
    return True

  print(f"Uploading {route}")
  for source_container in SOURCES:
    # assumes az login has been run
    print(f"Trying {source_container.ACCOUNT}/{source_container.CONTAINER}")
    _, source_key = source_container.get_client_and_key()
    cmd = [
      "azcopy",
      "copy",
      f"{source_container.BASE_URL}{key_prefix}?{source_key}",
      f"{DEST.BASE_URL}{dongle_id}?{dest_key}",
      "--recursive=true",
      "--overwrite=false",
      "--exclude-pattern=*/dcamera.hevc",
    ]

    try:
      result = subprocess.call(cmd, stdout=subprocess.DEVNULL)
      if result == 0:
        print("Success")
        return True
    except subprocess.CalledProcessError:
      print("Failed")

  return False


if __name__ == "__main__":
  failed_routes = []

  to_sync = sys.argv[1:]

  if not len(to_sync):
    # sync routes from the car tests routes and process replay
    to_sync.extend([rt.route for rt in test_car_models_routes])
    to_sync.extend([s[1].rsplit('--', 1)[0] for s in replay_segments])

  for r in tqdm(to_sync):
    if not sync_to_ci_public(r):
      failed_routes.append(r)

  if len(failed_routes):
    print("failed routes:", failed_routes)
