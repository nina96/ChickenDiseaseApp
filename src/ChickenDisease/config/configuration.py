from src.ChickenDisease.constants import * 
import os
from src.ChickenDisease.utils.common import read_yaml,create_directories, get_size
from src.ChickenDisease.entity.config_entity import TrainingConfig, PrepareBaseModelConfig, PrepareCallbackConfig, EvaluationConfig

class ConfigurationManger:
    def __init__(
            self,
            config_filepath= CONFIG_FILE_PATH,
            param_filepath= PARAM_FILE_PATH):
            
            self.config= read_yaml(config_filepath)
            self.param= read_yaml(param_filepath)

            create_directories([self.config.artifacts_root])

    
    def get_prepare_base_model_config(self)-> PrepareBaseModelConfig:
          config=self.config.prepare_base_model
          create_directories([config.root_dir])

          prepare_base_model_config= PrepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path=Path(config.base_model_path),
                updated_base_model_path=Path(config.updated_model_path),
                params_image_size=self.param.IMAGE_SIZE,
                params_learning_rate=self.param.LEARNING_RATE,
                params_include_top=self.param.INCLUDE_TOP,
                params_weights=self.param.WEIGHTS,
                params_classes=self.param.CLASSES,
                params_batch_size=self.param.BATCH_SIZE
          )

          return prepare_base_model_config
    
    def get_prepare_callback_config(self)->PrepareCallbackConfig:
          config= self.config.prepare_callbacks
          model_ckpt_dir= os.path.dirname(config.checkpoint_model_filepath)
          create_directories([
                    Path(model_ckpt_dir),
                    Path(config.tensorboard_root_log_dir)
          ])

          prepare_callback_config = PrepareCallbackConfig(
                root_dir= Path(config.root_dir),
                tensorboard_root_log_dir= Path(config.tensorboard_root_log_dir),
                checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
          
          )

          return prepare_callback_config
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.data_ingestion
        create_directories([ Path(training.root_dir)])


        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            training_data=Path(training_data.local_data_file),
            updated_base_model_path= Path(prepare_base_model.updated_model_path),
            params_epoch=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        
        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/chickens.h5",
            training_data="data/pathwithname.csv",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config