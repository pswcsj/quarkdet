from .coco_detection import CocoDetectionEvaluator


def build_evaluator(cfg, dataset):
    #dataset은 dataset 객체
    if cfg.evaluator.name == 'CocoDetectionEvaluator':
        return CocoDetectionEvaluator(dataset)
    else:
        raise NotImplementedError
