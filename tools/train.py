from torchsummary import summary
import sys

sys.path.append("./")
from quarkdet.evaluator import build_evaluator
from quarkdet.model.detector import build_model
from quarkdet.data.dataset import build_dataset
from quarkdet.data.collate import custom_collate_function
from quarkdet.trainer import build_trainer
from quarkdet.util import mkdir, Logger, cfg, load_config
import os
import torch
import logging
import argparse
import numpy as np
import torch.distributed as dist


# args를 리턴하는 함수
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    args = parser.parse_args()
    return args


# seed 값에 따라 랜덤으로 생성되는 값이 데결정되는데 seed를 어떤 값으로 초기화하여 랜덤으로 생성되는 값을 결정함
def init_seeds(seed=0):
    """
    manually set a random seed for numpy, torch and cuda
    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    # cuda에서 만들어지는 초기화 값의 seed를 결정
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_fn_coco(batch):
    return tuple(zip(*batch))


def main(args):
    # config의 경로를 args.config에서 가져와 cfg에 저장함
    load_config(cfg, args.config)

    # local_rank를 설정함(local rank는 한 컴퓨터 내에서 몇번째 gpu를 사용할 것인지를 나타냄)
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)

    #logger 초기화
    # cfg.save_dir은 quarkdet의 경우 workspace/ghostnet으로 설정돼 있음
    logger = Logger(local_rank, cfg.save_dir)
    if args.seed is not None:
        logger.log('Set random seed to {}'.format(args.seed))
        init_seeds(args.seed)

    logger.log('Creating model...')
    model = build_model(cfg.model)

    print("model:", model)

    # pre_dict = model.state_dict() #按键值对将模型参数加载到pre_dict
    # for k, v in pre_dict.items(): # 打印模型参数
    # for k, v in pre_dict.items(): #打印模型每层命名
    #     print ('%-50s%s' %(k,v.shape))

    # summary(model, (3, 320, 320))

    logger.log('Setting up data...')
    # build_dataset은 dataset object를 만드는 함수

    # train_dataset의 경우 cfg.data.train을 인자로 줌

    # name: 데이터셋 이름
    # img_path: 이미지 경로
    # ann_path: 어노테이션 경로
    # input_size: [w,h] 이미지 사이즈
    # keep_ratio: True 비율을 지킬지
    # pipeline: 데이터 어그멘테이션으로 뭘 넣을지를 넣는 거 같음
    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'test')

    if len(cfg.device.gpu_ids) > 1:
        print('rank = ', local_rank)
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % num_gpus)
        dist.init_process_group(backend='nccl')
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        # train_dataloader를 정의함
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                       num_workers=cfg.device.workers_per_gpu, pin_memory=True,
                                                       collate_fn=custom_collate_function, sampler=train_sampler,
                                                       drop_last=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                       shuffle=True, collate_fn=custom_collate_function,
                                                       pin_memory=True, drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                 pin_memory=True, collate_fn=custom_collate_function, drop_last=True)
    # trainer 정의

    # local_rank
    trainer = build_trainer(local_rank, cfg, model, logger)

    if cfg.schedule.resume:
        trainer.resume(cfg)
        if 'load_model' in cfg.schedule:
            trainer.load_model(cfg)

    #cfg는 config 파일. 현재는 CocoEvalDetector이 선택됨. val_dataset은
    evaluator = build_evaluator(cfg, val_dataset)

    logger.log('Starting training...')
    trainer.run(train_dataloader, val_dataloader, evaluator)


if __name__ == '__main__':
    args = parse_args()
    main(args)
