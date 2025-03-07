#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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
"""! @brief Script to launch tuning jobs, or execute commands on available machines"""

import os
import argparse
import sys
import logging
from typing import Dict, List, Any, Union
from tuna.utils.logger import setup_logger
from tuna.libraries import Library
from tuna.lib_utils import get_library
from tuna.miopen.miopen_lib import MIOpen
from tuna.example.example_lib import Example
from tuna.yaml_parser import parse_yaml
from tuna.parse_args import clean_args

# Setup logging
LOGGER: logging.Logger = setup_logger('go_fish')


def parse_args() -> Dict[str, Any]:
  """Function to parse arguments"""
  args: argparse.Namespace
  parser = argparse.ArgumentParser(add_help=False)

  parser.add_argument('lib',
                      nargs='?',
                      default=Library.MIOPEN,
                      type=Library,
                      help="Specify library to run",
                      choices=Library)

  parser.add_argument('--yaml',
                      dest='yaml',
                      default=None,
                      help='Path to yaml input file')

  args, _ = parser.parse_known_args()

  if '--yaml' in sys.argv and len(sys.argv) > 4:
    parser.error('Command line arguments not accepted with yaml file')

  return vars(args)


def main() -> bool:
  """Main function to start Tuna"""

  args: Dict[str, Any]
  args = parse_args()
  clean_args()

  #case no yaml file
  library: Union[Example, MIOpen]
  yaml_files: List[str]
  library = get_library(args)
  yaml_files = [args['yaml']]

  #case with yaml file
  if args['yaml']:
    yaml_files = parse_yaml(args['yaml'], args['lib'])

  job_batch_size = 1000
  if 'TUNA_CELERY_JOB_BATCH_SIZE' in os.environ:
    job_batch_size = int(os.environ['TUNA_CELERY_JOB_BATCH_SIZE'])

  try:
    for yaml_file in yaml_files:
      args['yaml_file'] = yaml_file
      if args['yaml_file']:
        sys.argv[2] = yaml_file
        LOGGER.info("Executing with yaml file: %s", yaml_file)

      if library.has_tunable_operation():
        #Celery operations
        library.tune(job_batch_size=job_batch_size)
      else:
        #non-celery operations
        #returns a list of workers/processes it started
        worker_lst = library.run()
        if worker_lst is None:
          continue

        for worker in worker_lst:
          worker.join()
          LOGGER.warning('Process finished')
  except KeyboardInterrupt:
    LOGGER.warning('Interrupt signal caught')

  return True


if __name__ == '__main__':
  main()
