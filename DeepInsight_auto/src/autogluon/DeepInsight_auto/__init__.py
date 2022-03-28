import logging
from autogluon.core.dataset import TabularDataset
from autogluon.common.features.feature_metadata import FeatureMetadata
#from autogluon.tabular_to_image.utils_pro.utils_pro import Utils_pro
#from autogluon.tabular_to_image.models_zoo.models_zoo import ModelsZoo
from autogluon.DeepInsight_auto.pyDeepInsight.image_transformer import ImageTransformer,LogScaler


try:
    from .version import __version__
except ImportError:
    pass

from autogluon.DeepInsight_auto.pyDeepInsight.image_transformer import ImageTransformer,LogScaler

logging.basicConfig(format='%(message)s')  # just print message in logs


from .version import __version__
