from genericpath import exists
import os
import json
import logging
from datetime import datetime

from .config import DataArguments, MiscArgument, ModelArguments, TrainingArguments, AnalysisArguments, BaselineArguments


def prepare_dirs_and_logger(
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        analysis_args: AnalysisArguments,
        baseline_args: BaselineArguments
) -> None:
    # os.chdir(os.path.dirname(__file__))
    os.chdir(misc_args.root_dir)
    path = os.getcwd()

    if not os.path.exists(misc_args.log_dir):
        os.makedirs(misc_args.log_dir)

    # if not os.path.exists(analysis_args.analysis_result_dir):
    #     os.makedirs(analysis_args.analysis_result_dir)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # data path


def get_time(format_string="%m%d_%H%M%S"):
    return datetime.now().strftime(format_string)


def save_config(config):

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % config.param_path)

    with open(config.param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        fp.write('\n')
