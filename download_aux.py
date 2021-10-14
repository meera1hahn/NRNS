import os
import gdown
import zipfile

RESNET_LINKS = [
    (
        "models/resnet18_places365.pth.tar",
        "https://drive.google.com/uc?id=1lNONueGpvM8J8PpLuN9K-uzr2xNZH_OL",
    )
]
NOISE_LINKS = [
    (
        "models/noise_models.zip",
        "https://drive.google.com/uc?id=1RrxbdszQKqJVezJI3hwf2H5QR6Q6Exc1",
    )
]
GIBSON_TEST_LINKS = [
    (
        "data/topo_nav/gibson/gibson_episodes.zip",
        "https://drive.google.com/uc?id=1ta_3BUpTD39R-KZrNabQMaQ-t_8_0eIA",
    )
]
MP3D_TEST_LINKS = [
    (
        "data/topo_nav/mp3d/mp3d_episodes.zip",
        "https://drive.google.com/uc?id=1VNYyjz72tYEvWencb8krMRDrVvv67n26",
    )
]
GIBSON_MODELS_LINKS = [
    (
        "models/gibson_models.zip",
        "https://drive.google.com/uc?id=1oq_7baTERTVHV9guPlxVWrtH-UsFtmRe",
    )
]


def _download_drive_url_to_file(url, path, _iszipfile=False):
    print(f"downloading {url}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    output = path
    gdown.download(url, output, quiet=False)
    print(f"downloading {url}... done!")
    if _iszipfile:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path[: path.rindex("/") + 1])
        os.remove(path)
        print(f"unzip {path}... done!")


if __name__ == "__main__":
    os.makedirs("models/", exist_ok=True)
    os.makedirs("logs/", exist_ok=True)
    os.makedirs("logs/tensorboard/", exist_ok=True)
    os.makedirs("logs/submitit/", exist_ok=True)
    os.makedirs("logs/submitit/log_test/", exist_ok=True)
    os.makedirs("data/scene_datasets/", exist_ok=True)
    os.makedirs("data/topo_nav/", exist_ok=True)
    os.makedirs("data/topo_nav/gibson/", exist_ok=True)
    os.makedirs("data/topo_nav/mp3d/", exist_ok=True)

    for path, url in RESNET_LINKS:
        _download_drive_url_to_file(url, path, _iszipfile=False)
    for path, url in NOISE_LINKS:
        _download_drive_url_to_file(url, path, _iszipfile=True)
    for path, url in GIBSON_TEST_LINKS:
        _download_drive_url_to_file(url, path, _iszipfile=True)
    for path, url in MP3D_TEST_LINKS:
        _download_drive_url_to_file(url, path, _iszipfile=True)
    for path, url in GIBSON_MODELS_LINKS:
        _download_drive_url_to_file(url, path, _iszipfile=True)
