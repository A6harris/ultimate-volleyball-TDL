"""
Module for registering the TD Learning trainer with ML-Agents.
This allows the TD Learning trainer to be used with the mlagents-learn CLI.
"""

from mlagents.trainers.trainer import Trainer
from td_trainer import TDTrainer
from mlagents.trainers.settings import TrainerSettings, TrainerType
from mlagents.trainers.exception import UnityTrainerException
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)

# Register the TD Trainer type
TrainerType.TD = "td"


def _validate_trainer_settings(trainer_settings: TrainerSettings) -> None:
    """
    Validates the trainer settings for TD learning.
    """
    # Validate required hyperparameters
    if not hasattr(trainer_settings.hyperparameters, "epsilon"):
        raise UnityTrainerException(
            "epsilon is a required hyperparameter for TD trainer."
        )
    if not hasattr(trainer_settings.hyperparameters, "epsilon_decay"):
        raise UnityTrainerException(
            "epsilon_decay is a required hyperparameter for TD trainer."
        )
    if not hasattr(trainer_settings.hyperparameters, "epsilon_min"):
        raise UnityTrainerException(
            "epsilon_min is a required hyperparameter for TD trainer."
        )
    if not hasattr(trainer_settings.hyperparameters, "steps_per_update"):
        raise UnityTrainerException(
            "steps_per_update is a required hyperparameter for TD trainer."
        )
    if not hasattr(trainer_settings.hyperparameters, "target_update_interval"):
        raise UnityTrainerException(
            "target_update_interval is a required hyperparameter for TD trainer."
        )
    logger.info("TD hyperparameters validated successfully!")


def get_td_trainer(trainer_settings: TrainerSettings, **kwargs) -> Trainer:
    """
    Returns a TD Trainer instance.
    """
    _validate_trainer_settings(trainer_settings)
    return TDTrainer(**kwargs, trainer_settings=trainer_settings)