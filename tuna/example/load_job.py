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
"""
Script for adding jobs to the MySQL database
"""

import logging
import argparse

from typing import Type
from sqlalchemy.exc import IntegrityError
from tuna.utils.logger import setup_logger
from tuna.parse_args import TunaArgs, setup_arg_parser
from tuna.utils.db_utility import connect_db
from tuna.dbBase.sql_alchemy import DbSession
from tuna.example.tables import ExampleDBTables
from tuna.example.example_tables import Job

LOGGER: logging.Logger = setup_logger('example_load_jobs')


def parse_args() -> argparse.Namespace:
  """ Argument input for the module """
  #pylint: disable=duplicate-code
  parser = setup_arg_parser(
      'Insert jobs into MySQL db',
      [TunaArgs.VERSION, TunaArgs.ARCH, TunaArgs.NUM_CU, TunaArgs.SESSION_ID])
  parser.add_argument('-l',
                      '--label',
                      type=str,
                      dest='label',
                      required=True,
                      help='Label to annotate the jobs.',
                      default='new')

  args: argparse.Namespace = parser.parse_args()
  if not args.session_id:
    parser.error('session_id must be specified')

  return args


def add_jobs(args: argparse.Namespace, dbt: Type[ExampleDBTables]) -> int:
  """
    Function uses args.label & args.session_id to create new jobs into
    job_table.
  """
  counts = 0
  with DbSession() as session:
    try:
      job: Job = dbt.job_table()
      job.state = 'new'
      job.valid = 1
      job.reason = args.label
      job.session = args.session_id
      session.add(job)
      session.commit()
      counts += 1
    except IntegrityError as err:
      session.rollback()
      LOGGER.warning('Integrity Error while adding new job: %s', err)

  return counts


def main():
  """ main """
  args = parse_args()
  connect_db()

  dbt = ExampleDBTables(session_id=None)
  cnt = add_jobs(args, dbt)

  print(f"New jobs added: {cnt}")


if __name__ == '__main__':
  main()
