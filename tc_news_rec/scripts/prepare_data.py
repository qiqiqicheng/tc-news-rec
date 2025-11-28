from omegaconf import DictConfig, OmegaConf
import hydra

from tc_news_rec.data.preprocessor import DataProcessor
from tc_news_rec.utils.logger import RankedLogger

log = RankedLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="prepare_data.yaml"
)
def main(cfg: DictConfig):
    log.info(f"{OmegaConf.to_yaml(cfg.data.data_preprocessor)}")
    
    log.info(f"Instantiating datamodule <{cfg.data.data_preprocessor._target_}>")
    preprocessor: DataProcessor = hydra.utils.instantiate(cfg.data.data_preprocessor)

    preprocessor.process()
    
if __name__ == "__main__":
    main()