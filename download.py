import requests
from kaggle import api
import os
from utils.data import load_dataset
from PIL import Image
import logging
from multiprocessing import Pool
import io

logging.basicConfig(level=20)

COMPETITION = "imaterialist-challenge-furniture-2018"
DATASETS = ["train", "test", "validation"]


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def download_image(image_id, image_url, save_dir, force=False):
    success = True
    try:
        logging.info("Downloading Image From: %s" % image_url)
        img_path = os.path.join(save_dir, str(image_id) + '.jpg')
        if not os.path.exists(img_path) or force:
            req = requests.get(image_url, timeout=10)
            req.raise_for_status()
            img = Image.open(io.BytesIO(req.content))
            if not img.format == 'JPEG':
                img = img.convert('RGB')
            img.save(img_path)
        else:
            logging.info("Image %s Exists, Skip Download !" % img_path)
    except Exception as err:
        logging.error(str(image_id) + ": " + str(err))
        success = False
    return success


def download_dataset_images(image_dir, image_list, image_annotations=None, force=False):
    reqs = []
    for i, image in enumerate(image_list):
        image_url = image['url'][0]
        image_id = image['image_id']
        if image_annotations:
            image_label = image_annotations[i]['label_id']
            assert image_id == image_annotations[i]['image_id'], "Image Id Is Not Match (%s vs %s)" % \
                                                                 (image_id, image_annotations[i]['image_id'])
            image_save_dir = os.path.join(image_dir,  str(image_label))
            if not os.path.exists(image_save_dir):
                os.mkdir(image_save_dir)
        else:
            image_save_dir = image_dir
        reqs.append((image_id, image_url, image_save_dir, force))

    with Pool(processes=os.cpu_count()*2) as pool:
        res = pool.starmap_async(download_image, reqs)
        downloaded = sum(res.get())

    return downloaded


def download_dataset(dataset_dir, force=False):
    api.competitionDownloadFiles(COMPETITION, path=dataset_dir, quiet=False)

    for dataset in DATASETS:
        image_dir, image_list, image_annotations = load_dataset(dataset, dataset_dir)

        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        downloaded = download_dataset_images(image_dir, image_list, image_annotations, force)

        logging.info("Finished Download %s Dataset, Total %d Images !" % (dataset, downloaded))


if __name__=="__main__":
    dataset_dir = "dataset"
    download_dataset(dataset_dir)
