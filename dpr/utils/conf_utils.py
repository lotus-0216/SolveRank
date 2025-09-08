import glob
import logging
import os

from dpr.data.biencoder_data import JsonQADataset

logger = logging.getLogger(__name__)

class BiencoderDatasetsCfg:
    def __init__(self, cfg):
        ds_cfg = cfg.datasets
        self.train_datasets_names = cfg.train_datasets
        logger.info("train_datasets: %s", self.train_datasets_names)
        self.train_datasets = self._init_datasets(self.train_datasets_names, ds_cfg)
        self.dev_datasets_names = cfg.dev_datasets
        logger.info("dev_datasets: %s", self.dev_datasets_names)
        self.dev_datasets = self._init_datasets(self.dev_datasets_names, ds_cfg)
        self.sampling_rates = cfg.train_sampling_rates

    def _init_datasets(self, datasets_names, ds_cfg):
        if isinstance(datasets_names, str):
            return [self._init_dataset(datasets_names, ds_cfg)]
        elif datasets_names:
            return [self._init_dataset(ds_name, ds_cfg) for ds_name in datasets_names]
        else:
            return []

    def _init_dataset(self, name, ds_cfg):
        if os.path.exists(name):
            return JsonQADataset(name)
        elif glob.glob(name):
            files = glob.glob(name)
            return [self._init_dataset(f, ds_cfg) for f in files]
        if name not in ds_cfg:
            raise RuntimeError(f"Can't find dataset location/config for: {name}")
        return ds_cfg[name]

def get_pretrained_model_cfg():
    """
    新增函数，直接返回本地模型路径
    """
    return './bert-base-uncased'
