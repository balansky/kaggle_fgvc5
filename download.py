import requests
from kaggle import api
import os
from utils.data import load_dataset
from PIL import Image
import logging
from multiprocessing import Queue
import concurrent.futures
import argparse
import io

logging.basicConfig(level=20)

COMPETITION = "imaterialist-challenge-furniture-2018"
DATASETS = ["train", "test", "validation"]


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def fillup_request_queue(image_dir, image_list, image_annotations=None):

    reqs = Queue()

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
        reqs.put((image_id, image_url, image_save_dir))

    return reqs


def download_images(q, force=False):

    successes = 0

    while not q.empty():
        image_id, image_url, save_dir = q.get()
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
            successes += 1
        except Exception as err:
            logging.error(str(image_id) + ": " + str(err))

    return successes



def download_dataset(req_queue, force=False):

    downloaded = 0
    concurrent_nums = os.cpu_count()*2

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_nums) as executor:
        futures = [executor.submit(download_images, req_queue, force) for _ in range(concurrent_nums)]
        for future in concurrent.futures.as_completed(futures):
            downloaded += future.result()

    return downloaded


def main(dataset_dir, force=False):

    total = 0

    api.competitionDownloadFiles(COMPETITION, path=dataset_dir, quiet=False)

    for dataset in DATASETS:
        image_dir, image_list, image_annotations = load_dataset(dataset, dataset_dir)

        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        req_queue = fillup_request_queue(image_dir, image_list, image_annotations)
        downloaded = download_dataset(req_queue, force)
        total += downloaded
        logging.info("Finished Download %s Dataset, Total %d Images !" % (dataset, downloaded))

    logging.info("Total Downloaded %d Images, Exit!" % total)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--force',
        default=False,
        action='store_true'
    )
    args, unparsed = parser.parse_known_args()
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        make_folder(dataset_dir)
    main(dataset_dir, args.force)
