#!/usr/bin/env python3
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
"""Module to represent MIOpen subcommands parsers"""
import jsonargparse
from tuna.parse_args import TunaArgs, setup_arg_parser
from tuna.miopen.db.benchmark import FrameworkEnum, ModelEnum
from tuna.miopen.utils.metadata import ALG_SLV_MAP


def get_import_cfg_parser(
    with_yaml: bool = True) -> jsonargparse.ArgumentParser:
  """Return parser for import_configs subcommand"""

  parser = setup_arg_parser(
      'Import MIOpenDriver commands and MIOpen performance DB entries.',
      [TunaArgs.VERSION, TunaArgs.CONFIG_TYPE],
      with_yaml=with_yaml)

  group = parser.add_mutually_exclusive_group()

  group.add_argument(
      '--add_framework',
      dest='add_framework',
      choices=[frm.value for frm in FrameworkEnum],
      help='Populate framework table with new framework and version')
  group.add_argument('--add_model',
                     dest='add_model',
                     choices=[model.value for model in ModelEnum],
                     help='Populate table with new model and version')
  group.add_argument('--print_models',
                     dest='print_models',
                     action='store_true',
                     help='Print models from table')
  group.add_argument('--add_benchmark',
                     dest='add_benchmark',
                     action='store_true',
                     help='Insert new benchmark')

  parser.add_argument('-b',
                      '--batches',
                      type=str,
                      dest='batches',
                      help='Batch sizes to iterate over in the given configs')
  parser.add_argument('--batchsize',
                      dest='batchsize',
                      type=int,
                      default=None,
                      required=False,
                      help='Specify model batchsize')
  parser.add_argument(
      '-c',
      '--command',
      type=str,
      dest='command',
      default=None,
      help='Command override: run a different command on the imported configs',
      choices=[None, 'conv', 'convfp16', 'convbfp16'])
  parser.add_argument('-d',
                      '--driver',
                      dest='driver',
                      type=str,
                      default=None,
                      help='Specify driver cmd')
  parser.add_argument('-f',
                      '--file_name',
                      type=str,
                      dest='file_name',
                      help='File to import')
  parser.add_argument('-F',
                      '--framework',
                      dest='framework',
                      choices=[frm.value for frm in FrameworkEnum],
                      help='Specify framework.')
  parser.add_argument('--fw_version',
                      dest='fw_version',
                      type=int,
                      default=None,
                      required=False,
                      help='Specify framework version')
  parser.add_argument('-g',
                      '--gpu_count',
                      dest='gpu_count',
                      type=int,
                      default=None,
                      required=False,
                      help='Specify number of gpus the benchmark runs on')
  parser.add_argument(
      '--mark_recurrent',
      dest='mark_recurrent',
      action="store_true",
      help='Indicate whether you want the configs to be marked as recurrent')
  parser.add_argument('-m',
                      '--model',
                      dest='model',
                      choices=[model.value for model in ModelEnum],
                      help='Specify model')
  parser.add_argument('--md_version',
                      dest='md_version',
                      type=int,
                      default=None,
                      required=False,
                      help='Specify model version')
  parser.add_argument('-t',
                      '--tag',
                      type=str,
                      dest='tag',
                      help='Tag to mark the origin of this \
                      config, if config not present it will insert. No wildcard columns for \
                      tagging.')
  parser.add_argument(
      '-T',
      '--tag_only',
      action='store_true',
      dest='tag_only',
      help=
      'Tag to mark the origin of this config but skips the insert new config \
                      step in case the config does not exist in the table. Wildcard columns \
                      allowed for tagging')

  return parser


def get_load_job_parser(with_yaml: bool = True) -> jsonargparse.ArgumentParser:
  "Return parser for load_job subcommand"

  #pylint: disable=duplicate-code
  parser = setup_arg_parser('Insert jobs into MySQL db by tag from" \
      " config_tags table.', [TunaArgs.VERSION, TunaArgs.CONFIG_TYPE],
                            with_yaml=with_yaml)
  config_filter = parser.add_mutually_exclusive_group(required=True)
  solver_filter = parser.add_mutually_exclusive_group()
  config_filter.add_argument(
      '-t',
      '--tag',
      type=str,
      dest='tag',
      default=None,
      help='All configs with this tag will be added to the job table. \
                        By default adds jobs with no solver specified (all solvers).'
  )
  config_filter.add_argument('--all_configs',
                             dest='all_configs',
                             action='store_true',
                             help='Add all convolution jobs')
  solver_filter.add_argument(
      '-A',
      '--algo',
      type=str,
      dest='algo',
      default=None,
      help='Add job for each applicable solver+config in Algorithm.',
      choices=ALG_SLV_MAP.keys())
  solver_filter.add_argument('-s',
                           '--solvers',
                           type=str,
                           dest='solvers',
                           default=None,
                           help='add jobs with only these solvers '\
                             '(can be a comma separated list)')
  parser.add_argument(
      '-d',
      '--only_dynamic',
      dest='only_dynamic',
      action='store_true',
      help='Use with --tag to create a job for dynamic solvers only.')
  parser.add_argument('--tunable',
                      dest='tunable',
                      action='store_true',
                      help='Use to add only tunable solvers.')
  parser.add_argument('-c',
                      '--cmd',
                      type=str,
                      dest='cmd',
                      default=None,
                      required=False,
                      help='Command precision for config',
                      choices=['conv', 'convfp16', 'convbfp16'])
  parser.add_argument('-l',
                      '--label',
                      type=str,
                      dest='label',
                      required=True,
                      help='Label to annontate the jobs.',
                      default='new')
  parser.add_argument('--fin_steps',
                      dest='fin_steps',
                      type=str,
                      default='not_fin')
  parser.add_argument(
      '--session_id',
      required=True,
      dest='session_id',
      type=int,
      help=
      'Session ID to be used as tuning tracker. Allows to correlate DB results to tuning sessions'
  )

  return parser


def get_export_db_parser(with_yaml: bool = True) -> jsonargparse.ArgumentParser:
  """Return parser for export db subcommand"""
  parser = setup_arg_parser('Convert MYSQL find_db to text find_dbs' \
  'architecture', [TunaArgs.ARCH, TunaArgs.NUM_CU, TunaArgs.VERSION, TunaArgs.CONFIG_TYPE],
          with_yaml=with_yaml)

  group_ver = parser.add_mutually_exclusive_group(required=True)
  group_ver.add_argument(
      '--session_id',
      dest='session_id',
      type=int,
      help=
      'Session ID to be used as tuning tracker. Allows to correlate DB results to tuning sessions'
  )
  group_ver.add_argument(
      '--golden_v',
      dest='golden_v',
      type=int,
      help='export from the golden table using this version number')

  parser.add_argument('--config_tag',
                      dest='config_tag',
                      type=str,
                      help='import configs based on config tag',
                      default=None)
  parser.add_argument('-c',
                      '--opencl',
                      dest='opencl',
                      action='store_true',
                      help='Use OpenCL extension',
                      default=False)
  parser.add_argument('--filename',
                      dest='filename',
                      help='Custom filename for DB dump',
                      default=None)

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('-k',
                     '--kern_db',
                     dest='kern_db',
                     action='store_true',
                     help='Serialize Kernel Database',
                     default=False)
  group.add_argument('-f',
                     '--find_db',
                     dest='find_db',
                     action='store_true',
                     help='Serialize Find Database',
                     default=False)
  group.add_argument('-p',
                     '--perf_db',
                     dest='perf_db',
                     action='store_true',
                     help='Serialize Perf Database',
                     default=False)
  return parser


def get_update_golden_parser(
    with_yaml: bool = True) -> jsonargparse.ArgumentParser:
  "Return parser for update golden subcommand"
  parser = setup_arg_parser('Populate golden table based on session_id.',
                            [TunaArgs.CONFIG_TYPE],
                            with_yaml=with_yaml)
  parser.add_argument('--golden_v',
                      dest='golden_v',
                      type=int,
                      default=None,
                      required=True,
                      help='target golden miopen version to write')
  parser.add_argument('--base_golden_v',
                      dest='base_golden_v',
                      type=int,
                      default=None,
                      required=False,
                      help='previous golden miopen version for initialization')
  parser.add_argument('--session_id',
                      required=False,
                      dest='session_id',
                      type=int,
                      help='Tuning session to be imported to golden table.')
  parser.add_argument('-o',
                      '--overwrite',
                      dest='overwrite',
                      action='store_true',
                      default=False,
                      help='Write over existing golden version.')
  parser.add_argument('--create_perf_table',
                      dest='create_perf_table',
                      action='store_true',
                      default=False,
                      help='Create performance table.')
  return parser
