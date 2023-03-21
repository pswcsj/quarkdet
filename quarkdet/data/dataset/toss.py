import os
import torch
import numpy as np
import cv2
from .base import BaseDataset
from quarkdet.util import distance2bbox, bbox2distance, overlay_bbox_cv
import random
import math
from quarkdet.util import cfg
__author__ = 'tylin'
__version__ = '2.0'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import numpy as np
import copy
import itertools
from . import mask as maskUtils
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class TOSS:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                #imgToAnns dictionary에서 해당 ann의 image_id key에 대응되는 value로 ann을 추가
                imgToAnns[ann['image_id']].append(ann)
                #anns dictionary에서 해당 ann의 id key에 대응되는 value로 ann 설정
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            # 각 category에 객체는 {"supercategory": "person", "id": 1, "name": "person"} 이런식으로 저장돼있음
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    #annotation 파일에 있는 id 목록을 모두 받아와 리스트로 반환
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
        ids = [ann['id'] for ann in anns]
        return ids

    #categories 데이터에 적혀있는 순서대로 category의 id를 가져옴
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            # {"supercategory": "person", "id": 1, "name": "person"} 각 category별 이런 정보를 가진 게 반환
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

class TossDataset(BaseDataset):

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url': 'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = TOSS(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds()) #category 객체로부터 category id 목록을 받아옴
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)} # 각각의 category를 label과 대응
        # "1": {"supercategory": "person", "id": 1, "name": "person"}, ... 이런 각 category별 정보를 받아옴
        self.cats = self.coco_api.loadCats(self.cat_ids)
        # imgs는 밑과 같은 정보가 배열로 저장돼 있음
        # 397133: {
        #     "license": 4,
        #     "file_name": "000000397133.jpg",
        #     "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        #     "height": 427,
        #     "width": 640,
        #     "date_captured": "2013-11-14 17:02:52",
        #     "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        #     "id": 397133
        # },...
        self.img_ids = sorted(self.coco_api.imgs.keys())  # image의 id들을 순서대로 저장
        img_info = self.coco_api.loadImgs(self.img_ids) #img_id 순서대로 img_info롤 불러와 저장함
        return img_info

    def show(self, meta, class_names):

        all_box = meta['gt_bboxes']

        img = meta['img'].astype(np.float32) / 255
        for i, box in enumerate(all_box):
            x0 = box[0]
            y0 = box[1]
            x1 = box[2]
            y1 = box[3]
            color = (0, 255, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        return img

    # 'file_name', 'height', 'width', 'id'가 있는 딕셔너리를 반환
    def get_per_img_info(self, idx):
        # img_info
        # [{
        #   'file_name': '000000000139.jpg',
        #   'height': 426,
        #   'width': 640,
        #   'id': 139},
        #  ]
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        id = img_info['id']
        if not isinstance(id, int):
            raise TypeError('Image id must be int.')
        info = {'file_name': file_name,
                'height': height,
                'width': width,
                'id': id}
        return info

    # bboxes, labels, bboxes_ignore 가 정의되어 있는 딕셔너리 반환
    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx] #이미지들의 id가 저장된 img_ids로부터 idx에 맞는 img_id를 받아옴

        #img_id에 대응되는 annotation id를 받아옴
        ann_ids = self.coco_api.getAnnIds([img_id])
        #해당 annotation id로부터 annotation 파일을 받아옴
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        # 해당 idx의 annotation을 순회
        for ann in anns:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']  # top left x, top left y, width, height
            # area는 segmentation area를 의미함
            if w < 1 or h < 1:  # 매우 작으면 넘어간다는 뜻
                continue
            # annotation에 있는 id가 cat_ids 여기 기준 sorted(self.coco_api.getCatIds())에 없으면 continue
            if ann['category_id'] not in self.cat_ids:
                continue
            # bbox를 top left x, top left y, bottom right x, bottom right y로 변환
            bbox = [x1, y1, x1 + w, y1 + h]


            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])

        # bounding box 배열이 정의돼 있으면
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        # 정의돼있지 않으면 0으로 초기
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        return annotation

    # __getitem__를 호출했을 때 호출되는 함수
    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """

        # 'file_name', 'height', 'width', 'id'가 있는 딕셔너리를 반환
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)  # img 폴더 path와 file_name을 합쳐 image_path를 만듦
        img = cv2.imread(image_path)  # image_path에서 이미지 읽어와 img 변수에 저장
        if img is None:
            print('image {} read failed.'.format(image_path))
            raise FileNotFoundError('Cant load image! Please check image path!')
        # bboxes, labels, bboxes_ignore 가 정의되어 있는 딕셔너리
        ann = self.get_img_annotation(idx)
        # print("img_ids:",len(self.img_ids))

        # img는 PIL로 읽은 이미지 파일, img_info는 'file_name', 'width', 'height', 'id'가 있는 딕셔너리,
        # gt_bboxes, gt_labels는 bounding box coor과 label이 저장된 배열
        # bbox는 left top x, left top y, right bottom x, right bottom y
        meta = dict(img=img,
                    img_info=img_info,
                    gt_bboxes=ann['bboxes'],
                    gt_labels=ann['labels'])
        # print("original meta:",meta)  opencv H x W xC

        # img_test=self.show(meta, cfg.class_names)
        # cv2.imshow('img_test:', img_test)
        # cv2.waitKey(0)

        meta = self.pipeline(meta, self.input_size)  # pipeline을 따라 data 수정을 거침
        # PIL, cv2의 이미지 순서는 (H, W, C)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))  # H x W x C to C x H x W
        return meta

    # def get_val_data(self, idx):
    #     """
    #     Currently no difference from get_train_data.
    #     Not support TTA(testing time augmentation) yet.
    #     :param idx:
    #     :return:
    #     """
    #     # TODO: support TTA
    #     return self.get_train_data(idx)

    def get_val_data(self, idx):
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print('image {} read failed.'.format(image_path))
            raise FileNotFoundError('Cant load image! Please check image path!')
        ann = self.get_img_annotation(idx)

        meta = dict(img=img,
                    img_info=img_info,
                    gt_bboxes=ann['bboxes'],
                    gt_labels=ann['labels'])
        if self.use_instance_mask:
            meta['gt_masks'] = ann['masks']
        if self.use_keypoint:
            meta['gt_keypoints'] = ann['keypoints']

        meta = self.pipeline(meta, self.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))
        return meta

    # 增加可视化的工作

    def load_image(self, idx):
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)

        return img  # .astype(np.float32)

    def load_annotations(self, idx):

        annotations_ids = self.coco_api.getAnnIds(
            imgIds=self.img_ids[idx], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco_api.loadAnns(annotations_ids)
        for _, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            if a['bbox'][2] > 0 and a['bbox'][3] > 0:
                annotation[0, :4] = a['bbox']
                annotation[0, 4] = a['category_id']

                annotations = np.append(annotations, annotation, axis=0)

        # transform from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max] # bbox = [x1, y1, x1 + w, y1 + h]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    #  [ 95.  47. 113. 115.   1.]
#  [ 24.   1.  76. 150.   1.]
#  [  9.   3.  30. 117.   1.]
#  [ 82.  58.  87.  77.   1.]
#  [ 91.  53.  97.  72.   1.]
#  [144.  50. 149.  64.   1.]
#  [123.  56. 126.  70.   1.]
#  [105.  54. 108.  61.   1.]
#  [ 61.  25.  82. 153.  41.]
#  [ 99.  80. 120.  91.  41.]
#  [ 11.  59.  16.  71.   1.]
#  [ 90.  55.  92.  65.   1.]
#  [ 53.  89.  62.  99.  41.]
#  [ 86.  55.  88.  60.   1.]
#  [129.  53. 133.  56.   3.]
#  [126.  53. 129.  57.   3.]
#  [ 82.  75.  85.  77.  41.]
#  [135.  53. 137.  60.   1.]
#  [277.  90. 286. 115.  43.]
#  [118. 251. 122. 253.  34.]
#  [201. 213. 259. 286.  19.]

# gt box 大小 将被过滤掉的框如下
# [105.  54. 108.  61.   1.]
# [ 86.  55.  88.  60.   1.]
# [129.  53. 133.  56.   3.]
# [126.  53. 129.  57.   3.]
# [ 82.  75.  85.  77.  41.]
# [135.  53. 137.  60.   1.]
# [118. 251. 122. 253.  34.]