import copy
from .coco import CocoDataset
from .toss import TossDataset

# cfg의 속성들
# name: 데이터셋 이름
# img_path: 이미지 경로
# ann_path: 어노테이션 경로
# input_size: [w,h] 이미지 사이즈
# keep_ratio: True 비율을 지킬지
# pipeline: 데이터 어그멘테이션으로 뭘 넣을지를 넣는 거 같음
def build_dataset(cfg, mode):
    # dataset의 cfg를 deep copy(새로운 메모리 공간을 만들어 내용이 똑같은 새로운 자료를 만듦)
    dataset_cfg = copy.deepcopy(cfg)
    if dataset_cfg['name'] == 'coco':
        dataset_cfg.pop('name')  # 데이터셋 이름 부분을 제외시킨다
        # mode(train or test), 그리고 cfg에 있는 내용들을 인자로 넘겨 데이터셋을 만든다
        return CocoDataset(mode=mode, **dataset_cfg)
    elif dataset_cfg['name'] == 'toss':
        dataset_cfg.pop('name')
        return TossDataset(mode = mode, **dataset_cfg)
