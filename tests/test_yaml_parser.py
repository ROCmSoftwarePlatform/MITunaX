###############################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################

import os
import sys
import yaml
import subprocess

sys.path.append("../tuna")
sys.path.append("tuna")

this_path = os.path.dirname(__file__)

from tuna.yaml_parser import parse_yaml
from tuna.example.example_lib import Example
from tuna.miopen.miopen_lib import MIOpen
from tuna.libraries import Library


def test_yaml_parser():
  miopen_yaml1 = "{0}/../tuna/miopen/yaml_files/sample1.yaml".format(this_path)
  miopen_yaml2 = "{0}/../tuna/miopen/yaml_files/sample2.yaml".format(this_path)
  miopen_yaml3 = "{0}/../tuna/miopen/yaml_files/sample3.yaml".format(this_path)
  miopen_yaml4 = "{0}/../tuna/miopen/yaml_files/sample4.yaml".format(this_path)
  example_yaml = "{0}/../tuna/example/sample.yaml".format(this_path)

  parse_miopen_yaml1(miopen_yaml1, Library('miopen'))
  parse_miopen_yaml2(miopen_yaml2, Library('miopen'))
  parse_miopen_yaml3(miopen_yaml3, Library('miopen'))
  parse_miopen_yaml4(miopen_yaml4, Library('miopen'))
  parse_example_yaml(example_yaml, Library('example'))

  multiple_yamls()


def multiple_yamls():
  yaml_sample = "{0}/yaml_sample.yaml".format(this_path)
  go_fish = "{0}/../tuna/go_fish.py miopen --yaml {1}".format(
      this_path, yaml_sample)
  subp_fail = False

  try:
    subprocess.run(go_fish, shell=True, check=True)
  except subprocess.CalledProcessError as subp_err:
    print(f"Subprocess error: {subp_err}")
    subp_fail = True
    print(f"{subp_err}")

  assert subp_fail == False


def parse_miopen_yaml1(miopen_yaml, miopen):
  yaml_files = parse_yaml(miopen_yaml, miopen)
  assert len(yaml_files) == 6

  yaml_dicts = []
  #reading in initial yaml file split in 2 yaml files
  for yfile in yaml_files:
    with open(yfile, encoding="utf8") as stream:
      yaml_dict = yaml.safe_load(stream)
      yaml_dicts.append(yaml_dict)

  dict1 = {
      'add_tables': True,
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }

  dict2 = {
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'init_session': True,
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }

  dict3 = {
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'import_configs': {
          'file_name': '../utils/configs/conv_configs.txt',
          'framework': 'Pytorch',
          'fw_version': 1,
          'md_version': 1,
          'model': 'Alexnet',
          'tag': 'someTag'
      },
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }

  dict4 = {
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'fin_steps': False,
      'load_job': {
          'solvers': "[('', None)]",
          'tunable': False,
          'fin_steps': False,
          'tag': 'someTag',
      },
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }
  dict5 = {
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'export_db': {
          'file_name': 'fdb_nhwc.txt'
      },
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }
  dict6 = {
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'update_golden': {
          'base_golden_v': 1.0,
          'golden_v': 1.1
      },
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }

  print(yaml_dicts[4])
  print(dict5)
  assert (yaml_dicts[0] == dict1)
  assert (yaml_dicts[1] == dict2)
  assert (yaml_dicts[2] == dict3)
  assert (yaml_dicts[3] == dict4)
  assert (yaml_dicts[4] == dict5)
  assert (yaml_dicts[5] == dict6)


def parse_miopen_yaml2(miopen_yaml, miopen):
  yaml_files = parse_yaml(miopen_yaml, miopen)
  assert len(yaml_files) == 1

  yaml_dicts = []
  for yfile in yaml_files:
    with open(yfile, encoding="utf8") as stream:
      yaml_dict = yaml.safe_load(stream)
      yaml_dicts.append(yaml_dict)

  dict1 = {
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'import_configs': {
          'add_model': 'Alexnet',
          'md_version': 1
      },
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }
  assert (yaml_dicts[0] == dict1)


def parse_miopen_yaml3(miopen_yaml, miopen):
  yaml_files = parse_yaml(miopen_yaml, miopen)
  assert len(yaml_files) == 1

  yaml_dicts = []
  for yfile in yaml_files:
    with open(yfile, encoding="utf8") as stream:
      yaml_dict = yaml.safe_load(stream)
      yaml_dicts.append(yaml_dict)

  dict1 = {
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'import_configs': {
          'add_framework': 'Pytorch',
          'fw_version': 1
      },
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }
  assert (yaml_dicts[0] == dict1)


def parse_miopen_yaml4(miopen_yaml, miopen):
  yaml_files = parse_yaml(miopen_yaml, miopen)
  assert len(yaml_files) == 1

  yaml_dicts = []
  for yfile in yaml_files:
    with open(yfile, encoding="utf8") as stream:
      yaml_dict = yaml.safe_load(stream)
      yaml_dicts.append(yaml_dict)

  dict4 = {
      'arch': 'gfx908',
      'config_type': 'convolution',
      'docker_name': 'my_docker_name',
      'label': 'Example',
      'load_job': {
          'solvers': "[('', None)]",
          'tunable': False,
          'fin_steps': False,
          'tag': 'someTag',
      },
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }
  assert (yaml_dicts[0] == dict4)


def parse_example_yaml(example_yaml, example):
  yaml_files = parse_yaml(example_yaml, example)
  assert len(yaml_files) == 2

  yaml_dicts = []
  #reading in initial yaml file split in 2 yaml files
  for yfile in yaml_files:
    with open(yfile, encoding="utf8") as stream:
      yaml_dict = yaml.safe_load(stream)
      yaml_dicts.append(yaml_dict)

  dict1 = {
      'arch': 'gfx908',
      'docker_name': 'my_docker_name',
      'init_session': True,
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }
  dict2 = {
      'arch': 'gfx908',
      'docker_name': 'my_docker_name',
      'execute': True,
      'label': 'Example',
      'num_cu': 120,
      'remote_machine': False,
      'restart_machine': False,
      'session_id': 1
  }

  assert (yaml_dicts[0] == dict1)
  assert (yaml_dicts[1] == dict2)
