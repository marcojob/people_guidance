import os
import sys
import zipfile
from pathlib import Path

import urllib.request

data_url = 'https://www.eth3d.net/data/slam'


# Reads the content at the given URL and returns it as text.
def make_url_request(url):
    url_object = None
    url_object = urllib.request.urlopen(url)

    result = ''

    block_size = 8192
    while True:
        buffer = url_object.read(block_size)
        if not buffer:
            break

        result += buffer.decode("utf-8")

    return result


# Downloads the given URL to a file in the given directory. Returns the
# path to the downloaded file.
# In part adapted from: https://stackoverflow.com/questions/22676
def download_file(url, dest_dir_path):
    file_name = url.split('/')[-1]
    dest_file_path = os.path.join(dest_dir_path, file_name)

    url_object = urllib.request.urlopen(url)

    with open(dest_file_path, 'wb') as outfile:
        meta = url_object.info()
        file_size = 0
        if sys.version_info[0] == 2:
            file_size = int(meta.getheaders("Content-Length")[0])
        else:
            file_size = int(meta["Content-Length"])
        print("Downloading: %s (size [bytes]: %s)" % (url, file_size))

        file_size_downloaded = 0
        block_size = 8192
        while True:
            buffer = url_object.read(block_size)
            if not buffer:
                break

            file_size_downloaded += len(buffer)
            outfile.write(buffer)

            sys.stdout.write(
                "%d / %d  (%3f%%)\r" % (file_size_downloaded, file_size, file_size_downloaded * 100. / file_size))
            sys.stdout.flush()

    return dest_file_path


# Unzips the given zip file into the given directory.
def unzip_file(file_path, unzip_dir_path):
    zip_ref = zipfile.ZipFile(open(file_path, 'rb'))
    zip_ref.extractall(unzip_dir_path)
    zip_ref.close()


# Downloads a zip file and directly unzips it.
def download_and_unzip_file(url, unzip_dir_path):
    archive_path = download_file(url, unzip_dir_path)
    unzip_file(archive_path, unzip_dir_path)
    os.remove(archive_path)


# Performs a request to get the list of all training datasets.
def get_training_dataset_list():
    text_list = make_url_request(data_url + '/dataset_list_training.txt')
    return text_list.split('\n')


# Performs a request to get the list of all test datasets.
def get_test_dataset_list():
    text_list = make_url_request(data_url + '/dataset_list_test.txt')
    return text_list.split('\n')


def download_dataset_for_conversion(dataset):
    training_datasets = get_training_dataset_list()
    test_datasets = get_test_dataset_list()

    download_mono = True
    download_stereo = False
    download_rgbd = True
    download_imu = True
    download_raw = False
    download_raw_calibration = False

    is_training_dataset = (dataset in training_datasets)
    if not is_training_dataset and not (dataset in test_datasets):
        raise ValueError('Dataset not found in training or test dataset list, skipping: ' + dataset)

    dpath = Path(__file__).parent.resolve()
    dpath.mkdir(exist_ok=True)
    datasets_path = str(dpath)

    if download_mono:
        download_and_unzip_file(
            data_url + '/datasets/' + dataset + '_mono.zip',
            datasets_path)

    if download_stereo:
        download_and_unzip_file(
            data_url + '/datasets/' + dataset + '_stereo.zip',
            datasets_path)

    if download_rgbd:
        download_and_unzip_file(
            data_url + '/datasets/' + dataset + '_rgbd.zip',
            datasets_path)

    if download_imu:
        download_and_unzip_file(
            data_url + '/datasets/' + dataset + '_imu.zip',
            datasets_path)

    if download_raw:
        download_and_unzip_file(
            data_url + '/datasets/' + dataset + '_raw.zip',
            datasets_path)

        if download_raw_calibration and is_training_dataset:
            calibration_dataset_path = os.path.join(datasets_path, dataset, 'calibration_dataset.txt')
            if not os.path.isfile(calibration_dataset_path):
                print('Error: No calibration_dataset.txt found for raw dataset: ' + dataset)
            else:
                calibration_dataset_name = ''
                with open(calibration_dataset_path, 'rb') as infile:
                    calibration_dataset_name = infile.read().decode('UTF-8').rstrip('\n')

                download_and_unzip_file(
                    data_url + '/calibration/' + calibration_dataset_name + '.zip',
                    datasets_path)

    print('Finished.')
