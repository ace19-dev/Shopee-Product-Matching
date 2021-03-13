import os
import cv2
import shutil

import torch


def get_all_filenames(dir_path):
    """	경로 하위에 있는 모든 파일 목록 조회
    :param
     path: 테스트 이미지 경로
    :return: 하위 디렉토리 내 전체 파일 목록
    """
    filenames = []

    for path, dirs, files in os.walk(dir_path):
        if files:
            for filename in files:
                filenames.append(os.path.join(path, filename))

    return filenames


def get_all_filenames_with_ext(dir_path, ext):
    """	경로 하위에 있는 모든 파일 목록 조회
    :param
     path: 테스트 이미지 경로
    :return: 하위 디렉토리 내 전체 파일 목록
    """
    filenames = []

    for path, dirs, files in os.walk(dir_path):
        if files:
            for filename in files:
                if filename.endswith(ext.lower()) or filename.endswith(ext.upper()):
                    filenames.append(os.path.join(path, filename))

    return filenames


def image_resize_on_the_path(path, ratio=0.8):
    images = get_all_filenames(path)
    for image in images:
        img = cv2.resize(cv2.imread(image), dsize=(0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(image, img)
