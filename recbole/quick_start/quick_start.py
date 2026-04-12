# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import logging
import os
from logging import getLogger
from datetime import datetime

import torch
import pickle

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color


def _ensure_hr_nd_5_10(config):
    """Ensure evaluation always includes Hit/NDCG at topk 5 and 10."""
    metrics = config['metrics']
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics_lower = {m.lower(): m for m in metrics}
    if 'hit' not in metrics_lower:
        metrics.append('Hit')
    if 'ndcg' not in metrics_lower:
        metrics.append('NDCG')
    config['metrics'] = metrics

    topk = config['topk']
    if isinstance(topk, int):
        topk = [topk]
    topk = sorted(set(topk + [5, 10]))
    config['topk'] = topk


def _sort_metric_key(item):
    key = item[0]
    if '@' in key:
        metric_name, topk = key.split('@', 1)
        try:
            topk = int(topk)
        except ValueError:
            topk = 10 ** 9
        return metric_name.lower(), topk, key
    return key.lower(), 10 ** 9, key


def _format_metric_lines(result_dict):
    if result_dict is None:
        return ['  (empty)']
    lines = []
    for key, value in sorted(result_dict.items(), key=_sort_metric_key):
        lines.append(f'  {key}: {float(value):.6f}')
    return lines


def _save_experiment_result(config, best_valid_score, best_valid_result, test_result):
    result_dir = os.path.join('log', 'results')
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{config['model']}_{config['dataset']}.txt")
    time_str = datetime.now().isoformat(timespec='seconds')
    record_lines = [
        f'time: {time_str}',
        f"model: {config['model']}",
        f"dataset: {config['dataset']}",
        f"topk: {config['topk']}",
        f"metrics: {config['metrics']}",
        f'best_valid_score: {float(best_valid_score):.6f}',
        'best_valid_result:',
        *_format_metric_lines(best_valid_result),
        'test_result:',
        *_format_metric_lines(test_result),
        '',
    ]
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(record_lines))
    return result_file


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    _ensure_hr_nd_5_10(config)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    if config['train_stage'] == 'pretrain':
        saved = False

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    result_file = _save_experiment_result(config, best_valid_score, best_valid_result, test_result)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    logger.info(set_color('saved result', 'yellow') + f': {result_file}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    _ensure_hr_nd_5_10(config)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    try:
        checkpoint = torch.load(model_file, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_file)
    except pickle.UnpicklingError:
        checkpoint = torch.load(model_file, weights_only=False)
    # checkpoint = torch.load(model_file, map_location=torch.device('cpu')) # for visualization
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
