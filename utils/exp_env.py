import git
import time
import json
import shutil
import socket
import getpass
import logging
import textwrap
import tensorflow as tf
from pathlib import Path
from omegaconf import OmegaConf


from definitions import ROOT_DIR


def parse_configs(configs, overwrite_config):
    if len(configs) == 1:
        config = OmegaConf.load(str(configs[0]))
    else:
        config = OmegaConf.merge(*[OmegaConf.load(str(cfg)) for cfg in configs])
    if overwrite_config is not None:
        config.merge_with(OmegaConf.from_cli(overwrite_config))
    OmegaConf.set_readonly(config, True)
    OmegaConf.set_struct(config, True)
    return config


def extract_git_info(repo_dir):
    repo = git.Repo(repo_dir)
    head = repo.head
    branch_name = head.ref.name
    commit = head.object.hexsha
    g = git.Git(repo_dir)
    git_log = g.log()
    diff = g.diff()
    return commit, branch_name, git_log, diff


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


class ExpEnv:

    def __init__(self, args, configs, overwrite_config=None):
        if isinstance(configs, str) or isinstance(configs, Path):
            configs = [configs]

        self.args = args
        self.config_paths = configs
        self.config = parse_configs(configs, overwrite_config)

        self.output_dir = Path(self.config.env.output_dir) / self.config.exp_name
        self.run_dir = self.output_dir / '_'.join([time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()),
                                                   getpass.getuser(), socket.gethostname()])
        self.checkpoints_dir = self.run_dir / 'checkpoints'
        self._mk_dirs()
        for cfg in self.config_paths:
            shutil.copy(cfg, self.output_dir)
        with open(self.output_dir / 'merged_config.yml', 'w') as f:
            f.write(self.config.pretty())

        self._save_git_info()

        self.tf_writer_logdir = str(self.run_dir / 'tb_log')
        self.tf_writer = tf.summary.create_file_writer(self.tf_writer_logdir)
        with self.tf_writer.as_default():
            tf.summary.text('configs', textwrap.indent(self.config.pretty(), '    '), step=0)
        tf.summary.flush()
        self._save_args(self.args)

    def _mk_dirs(self):
        if self.output_dir.exists():
            raise ValueError(f'The directory {self.output_dir} already exists!')
        logging.info(f'Creating output directory: {self.output_dir}')
        self.output_dir.mkdir()

        if self.run_dir.exists():
            raise ValueError(f'Run directory {self.run_dir} already exists!')
        logging.info(f'Creating run directory: {self.run_dir}')
        self.run_dir.mkdir()

        if self.checkpoints_dir.exists():
            raise ValueError(f'Checkpoints directory {self.checkpoints_dir} already exists!')
        logging.info(f'Creating checkpoint directory: {self.checkpoints_dir}')
        self.checkpoints_dir.mkdir()

    def _save_args(self, args):
        args_dict = {}
        for k, v in vars(args).items():
            if isinstance(v, Path):
                args_dict[k] = str(v)
            else:
                args_dict[k] = v

        with open(self.output_dir / 'args.json', 'w') as f:
            json.dump(args_dict, f, indent=4)

        with self.tf_writer.as_default():
            tf.summary.text('args', textwrap.indent(json.dumps(args_dict, indent=4), '    '), step=0)
        tf.summary.flush()

    def _save_git_info(self):
        commit, branch_name, git_log, diff = extract_git_info(ROOT_DIR)

        with open(self.output_dir / 'git_status.txt', 'w') as f:
            f.write(commit + '\n')
            f.write('branch: ' + branch_name + '\n\n\n')
            f.write('------log:\n' + git_log)

        with open(self.output_dir / 'git_diff.txt', 'w') as f:
            f.write(diff)


class SimpleEnv:
    def __init__(self, args, output_dir, configs=None, overwrite_config=None, other_repos=None):
        self.args = args
        self.output_dir = output_dir
        self.repos = [ROOT_DIR]
        self.config = None
        if configs is None:
            self.config = None
            self.config_paths = []
        else:
            self.config = parse_configs(configs, overwrite_config)
            self.config_paths = configs
        if other_repos is not None:
            self.repos.extend(other_repos)
        self.run_dir = self.output_dir / '_'.join([time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()),
                                                   getpass.getuser(), socket.gethostname()])
        logging.info(f'Creating run directory: {self.run_dir}')
        self.run_dir.mkdir(parents=True)

        for cfg in self.config_paths:
            shutil.copy(cfg, self.run_dir)
        if self.config:
            with open(self.run_dir / 'merged_config.yml', 'w') as f:
                f.write(self.config.pretty())

        self._save_args(self.args)
        self._save_git_info()

    def _save_args(self, args):
        args_dict = {}
        for k, v in vars(args).items():
            if isinstance(v, Path):
                args_dict[k] = str(v)
            else:
                if is_jsonable(v):
                    args_dict[k] = v
                else:
                    logging.warning(f"Argument '{k}':'{v}' is not jsonable, converting to str: '{str(v)}'")
                    args_dict[k] = str(v)

        with open(self.run_dir / 'args.json', 'w') as f:
            json.dump(args_dict, f, indent=4)

    def _save_git_info(self):
        repo_names = set()
        for repo in self.repos:
            commit, branch_name, git_log, diff = extract_git_info(repo)

            repo_name = repo.stem
            while repo_name in repo_names:
                logging.warning(f'Multiple repos with name: {repo_name}')
                repo_name += '_t'

            repo_names.add(repo_name)

            with open(self.run_dir / f'git_{repo_name}_status.txt', 'w') as f:
                f.write(commit + '\n')
                f.write('branch: ' + branch_name + '\n\n\n')
                f.write('------log:\n' + git_log)

            with open(self.run_dir / f'git_{repo_name}_diff.txt', 'w') as f:
                f.write(diff)

