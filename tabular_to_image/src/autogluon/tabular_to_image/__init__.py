import logging

from autogluon.core.dataset import TabularDataset
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.tabular_to_image.utils_pro.utils_tool import Utils_pro
from autogluon.tabular_to_image.models_zoo.models import ModelsZoo
from autogluon.DeepInsight_auto.pyDeepInsight import ImageTransformer,LogScaler


try:
    from .version import __version__
except ImportError:
    pass

from autogluon.tabular_to_image.prediction.predictions import ImagePredictions #.predictor import TabularPredictor

logging.basicConfig(format='%(message)s')  # just print message in logs
