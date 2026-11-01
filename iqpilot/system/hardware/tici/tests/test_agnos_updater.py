import json
import os
import requests

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MANIFESTS = [
  os.path.join(TEST_DIR, "../agnos.json"),
  os.path.join(TEST_DIR, "../agnos_tici_15_1.json"),
]

IMAGE_HOST = "git.konn3kt.com"

XZ_MAGIC = b"\xfd7zXZ\x00"
LFS_POINTER_MAGIC = b"version https://git-lfs"


class TestAgnosUpdater:

  def test_manifest(self):
    for manifest in MANIFESTS:
      with open(manifest) as f:
        m = json.load(f)

      for img in m:
        assert img['url'].split('/')[2] == IMAGE_HOST
        if not img['sparse']:
          assert img['hash'] == img['hash_raw']

        s = requests.Session()
        s.trust_env = False
        r = s.get(img['url'], timeout=10, stream=True,
                  headers={"User-Agent": "IQOS-Updater"})
        if r.status_code in (401, 403, 404):
          continue
        head = next(r.iter_content(chunk_size=256), b"") or b""
        assert not head.startswith(XZ_MAGIC), f"{img['name']}: anonymous request served image content"
        assert not head.startswith(LFS_POINTER_MAGIC), f"{img['name']}: anonymous request served the LFS pointer"
