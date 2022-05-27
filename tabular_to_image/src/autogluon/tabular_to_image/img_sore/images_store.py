import copy
import logging
import os
import time
from collections import defaultdict
from typing import List, Union, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import psutil
import torchvision
from torchvision import datasets, models, transforms
import torch

from autogluon.common.features.feature_metadata import FeatureMetadata

from .utils import process_hyperparameters
from ..augmentation.distill_utils import format_distillation_labels, augment_data
from ..constants import AG_ARGS, BINARY, MULTICLASS, REGRESSION, REFIT_FULL_NAME, REFIT_FULL_SUFFIX
from ..models import AbstractModel, BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel, GreedyWeightedEnsembleModel, SimpleWeightedEnsembleModel
from ..scheduler.scheduler_factory import scheduler_factory
from ..utils import default_holdout_frac, get_pred_from_proba, generate_train_test_split, infer_eval_metric, compute_permutation_feature_importance, extract_column, compute_weighted_metric
from ..utils.exceptions import TimeLimitExceeded, NotEnoughMemoryError, NoValidFeatures, NoGPUError, NotEnoughCudaMemoryError
from ..utils.loaders import load_pkl
from ..utils.savers import save_json, save_pkl
from ..utils.feature_selection import FeatureSelector

logger = logging.getLogger(__name__)


# FIXME: Below is major defect!
#  Weird interaction for metrics like AUC during bagging.
#  If kfold = 5, scores are 0.9, 0.85, 0.8, 0.75, and 0.7, the score is not 0.8! It is much lower because probs are combined together and AUC is recalculated
#  Do we want this to happen? Should we calculate score by 5 separate scores and then averaging instead?

# TODO: Dynamic model loading for ensemble models during prediction, only load more models if prediction is uncertain. This dynamically reduces inference time.
# TODO: Try midstack Semi-Supervised. Just take final models and re-train them, use bagged preds for SS rows. This would be very cheap and easy to try.
# TODO: Move to autogluon.core
class Store:
    trainer_file_name = 'trainer.pkl'
    trainer_info_name = 'info.pkl'
    trainer_info_json_name = 'info.json'
    distill_stackname = 'distill'  # name of stack-level for distilled student models

    def __init__(self, path: str,  low_memory=False, save_data=False):
        self.path = path
        self.save_data = save_data
        self.low_memory = low_memory

        self.is_data_saved = False
        
        self.X_train_img_saved = False
        self.y_train_saved=False
        
        self.X_val_img_saved = False
        self.y_val_saved=False
        
        self.X_test_img_saved = False
        self.y_test_saved=False

       
    # path_root is the directory containing learner.pkl
    @property
    def path_root(self) -> str:
        return self.path.rsplit(os.path.sep, maxsplit=2)[0] + os.path.sep

    @property
    def path_utils(self) -> str:
        return self.path_root + 'utils' + os.path.sep

    @property
    def path_data(self) -> str:
        return self.path_utils + 'data' + os.path.sep

    def load_X(self):
        if self._X_saved:
            path = self.path_data + 'X.pkl'
            return load_pkl.load(path=path)
        return None

    def load_X_val(self):
        if self._X_val_saved:
            path = self.path_data + 'X_val.pkl'
            return load_pkl.load(path=path)
        return None

    def load_y(self):
        if self._y_saved:
            path = self.path_data + 'y.pkl'
            return load_pkl.load(path=path)
        return None

    def load_y_val(self):
        if self._y_val_saved:
            path = self.path_data + 'y_val.pkl'
            return load_pkl.load(path=path)
        return None

    def load_data(self):
        X = self.load_X()
        y = self.load_y()
        X_val = self.load_X_val()
        y_val = self.load_y_val()

        return X, y, X_val, y_val

    def save_train(self, X_train_img,y_train):
        path = self.path_data + '/train'
        train={'X_train_img':X_train_img,'y_train' :y_train}
        torch.save(train, path)
        self.X_train_img_saved = True
        self.y_train_saved=True

    def save_val(self, X_val_img,y_val):
        path = self.path_data + '/val'
        val={'X_val_img':X_val_img,'y_val' :y_val}
        torch.save(val, path)
        self.X_val_img_saved = True
        self.y_val_saved=True
    
    def save_test(self, X_test_img,y_test):
        path = self.path_data + '/test'
        test={'X_test_img':X_test_img,'y_test' :y_test}
        torch.save(test, path)
        self.X_test_img_saved = True
        self.y_test_saved=True     


    def set_contexts(self, path_context):
        self.path, model_paths = self.create_contexts(path_context)
        for model, path in model_paths.items():
            self.set_model_attribute(model=model, attribute='path', val=path)

    def create_contexts(self, path_context: str) -> (str, dict):
        path = path_context
        model_paths = self.get_models_attribute_dict(attribute='path')
        for model, prev_path in model_paths.items():
            model_local_path = prev_path.split(self.path, 1)[1]
            new_path = path + model_local_path
            model_paths[model] = new_path

        return path, model_paths

    def save_model(self, model, reduce_memory=True):
        # TODO: In future perhaps give option for the reduce_memory_size arguments, perhaps trainer level variables specified by user?
        if reduce_memory:
            model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if self.low_memory:
            model.save()
        else:
            self.models[model.name] = model

    def save(self):
        models = self.models
        if self.low_memory:
            self.models = {}
        save_pkl.save(path=self.path + self.trainer_file_name, object=self)
        if self.low_memory:
            self.models = models

     # TODO: model_name change to model in params
    def load_model(self, model_name: str, path: str = None, model_type=None) -> AbstractModel:
        if isinstance(model_name, AbstractModel):
            return model_name
        if model_name in self.models.keys():
            return self.models[model_name]
        else:
            if path is None:
                path = self.get_model_attribute(model=model_name, attribute='path')
            if model_type is None:
                model_type = self.get_model_attribute(model=model_name, attribute='type')
            return model_type.load(path=path, reset_paths=self.reset_paths)


    def reduce_memory_size(self, data_files, remove_data=True, requires_save=True,**kwargs):
        if remove_data and self.is_data_saved:
            try:
                del data_files
            except FileNotFoundError:
                pass
            if requires_save:
                self.is_data_saved = False
            try:
                del data_files
            except OSError:
                pass
            
        models = self.get_model_names()
        for model in models:
            model = self.load_model(model)
            model.reduce_memory_size(remove_fit_stack=remove_fit_stack, remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, reduce_children=reduce_children, **kwargs)
            if requires_save:
                self.save_model(model, reduce_memory=False)
        if requires_save:
            self.save()

    # TODO: Also enable deletion of models which didn't succeed in training (files may still be persisted)
    #  This includes the original HPO fold for stacking
    # Deletes specified models from trainer and from disk (if delete_from_disk=True).
    def delete_models(self, models_to_keep=None, models_to_delete=None, allow_delete_cascade=False, delete_from_disk=True, dry_run=True):
        if models_to_keep is not None and models_to_delete is not None:
            raise ValueError('Exactly one of [models_to_keep, models_to_delete] must be set.')
        if models_to_keep is not None:
            if not isinstance(models_to_keep, list):
                models_to_keep = [models_to_keep]
            minimum_model_set = set()
            for model in models_to_keep:
                minimum_model_set.update(self.get_minimum_model_set(model))
            minimum_model_set = list(minimum_model_set)
            models_to_remove = [model for model in self.get_model_names() if model not in minimum_model_set]
        elif models_to_delete is not None:
            if not isinstance(models_to_delete, list):
                models_to_delete = [models_to_delete]
            minimum_model_set = set(models_to_delete)
            minimum_model_set_orig = copy.deepcopy(minimum_model_set)
            for model in models_to_delete:
                minimum_model_set.update(nx.algorithms.dag.descendants(self.model_graph, model))
            if not allow_delete_cascade:
                if minimum_model_set != minimum_model_set_orig:
                    raise AssertionError('models_to_delete contains models which cause a delete cascade due to other models being dependent on them. Set allow_delete_cascade=True to enable the deletion.')
            minimum_model_set = list(minimum_model_set)
            models_to_remove = [model for model in self.get_model_names() if model in minimum_model_set]
        else:
            raise ValueError('Exactly one of [models_to_keep, models_to_delete] must be set.')

        if dry_run:
            logger.log(30, f'Dry run enabled, AutoGluon would have deleted the following models: {models_to_remove}')
            if delete_from_disk:
                for model in models_to_remove:
                    model = self.load_model(model)
                    logger.log(30, f'\tDirectory {model.path} would have been deleted.')
            logger.log(30, f'To perform the deletion, set dry_run=False')
            return

        if delete_from_disk:
            for model in models_to_remove:
                model = self.load_model(model)
                model.delete_from_disk()

        self.model_graph.remove_nodes_from(models_to_remove)
        for model in models_to_remove:
            if model in self.models:
                self.models.pop(model)

        models_kept = self.get_model_names()

        if self.model_best is not None and self.model_best not in models_kept:
            try:
                self.model_best = self.get_model_best()
            except AssertionError:
                self.model_best = None

        # TODO: Delete from all the other model dicts
        self.save()

    @classmethod
    def load(cls, path, reset_paths=False):
        load_path = path + cls.trainer_file_name
        if not reset_paths:
            return load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
            obj.reset_paths = reset_paths
            return obj

    @classmethod
    def load_info(cls, path, reset_paths=False, load_model_if_required=True):
        load_path = path + cls.trainer_info_name
        try:
            return load_pkl.load(path=load_path)
        except:
            if load_model_if_required:
                trainer = cls.load(path=path, reset_paths=reset_paths)
                return trainer.get_info()
            else:
                raise

    def save_info(self, include_model_info=False):
        info = self.get_info(include_model_info=include_model_info)

        save_pkl.save(path=self.path + self.trainer_info_name, object=info)
        save_json.save(path=self.path + self.trainer_info_json_name, obj=info)
        return info
