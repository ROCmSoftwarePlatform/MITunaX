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
"""Utility module for helper functions"""

import random
import string
from time import sleep
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Query

from tuna.utils.logger import setup_logger
from tuna.dbBase.sql_alchemy import DbSession
from tuna.machine import Machine
from tuna.miopen.db.solver import get_solver_ids
from tuna.utils.utility import check_qts
from tuna.miopen.utils.metadata import MYSQL_LOCK_WAIT_TIMEOUT
from tuna.miopen.utils.metadata import FUSION_DEFAULTS, CONV_2D_DEFAULTS, CONV_3D_DEFAULTS
from tuna.utils.metadata import NUM_SQL_RETRIES
from tuna.utils.db_utility import gen_update_query, session_retry

LOGGER = setup_logger('helper')


#NOTE:remove pylint flag after driver implementation throughout code
#pylint: disable=duplicate-code
def get_qts_machine_data(mid, mhost, slogger):
  """Get machine local QTS info"""
  with DbSession() as session:
    query = session.query(Machine).filter(Machine.id == mid)
    rows = query.all()

    if rows:
      machine_obj = rows[0][0]
      machine_cfg = machine_obj.to_dict()
      machine_cfg['logger'] = slogger
      if check_qts(mhost):
        machine_cfg['hostname'] = machine_cfg['local_ip']
        machine_cfg['port'] = machine_cfg['local_port']
    else:
      slogger.warning('No machine found mid %u', mid)
      return None

    return machine_cfg


def print_solvers():
  """Pretty print solvers list"""
  slv_dict = get_solver_ids()
  for name, sid in slv_dict.items():
    print(f"{sid:>4} - {name}")


# fill in the missing columns with defaults to avoid duplicate entries
def config_set_defaults(fds):
  """Setting config DB defaults to avoid duplicates through SELECT"""
  if 'conv' in fds['cmd']:
    if 'spatial_dim' in fds and fds['spatial_dim'] == 3:
      fds = set_defaults(fds, CONV_3D_DEFAULTS)
    else:
      fds = set_defaults(fds, CONV_2D_DEFAULTS)
  elif 'cbainfer' in fds['cmd']:
    fds = set_defaults(fds, FUSION_DEFAULTS)
  elif 'bnorm' in fds['cmd']:
    fds = set_defaults(fds, BN_DEFAULTS)


def set_defaults(fds, defaults):
  """Set fds defaults"""
  for k, val in defaults.items():
    if k not in fds:
      fds[k] = val

  return fds


def valid_cfg_dims(in_perf_cfg):
  """prune 3rd dimension off 2d config"""
  perf_cfg = in_perf_cfg.copy()
  keys = list(perf_cfg.keys())[:]

  if 'spatial_dim' not in perf_cfg or perf_cfg['spatial_dim'] < 3:
    for key in keys:
      if key.endswith("_d"):
        perf_cfg[key] = CONV_2D_DEFAULTS[key]

  return perf_cfg


def prune_cfg_dims(in_perf_cfg):
  """prune 3rd dimension off 2d config"""
  perf_cfg = in_perf_cfg.copy()
  keys = list(perf_cfg.keys())[:]

  if 'spatial_dim' not in perf_cfg or perf_cfg['spatial_dim'] < 3:
    for key in keys:
      if key.endswith("_d"):
        perf_cfg.pop(key)

  return perf_cfg


def mysqldb_insert_dict(table, in_dict, filter_dict):
  """insert/update dict obj to mysql table"""
  with DbSession() as session:
    for idx in range(NUM_SQL_RETRIES):
      try:
        query = session.query(table).filter_by(**filter_dict)
        query = query.order_by(table.id.asc())
        res = query.first()

        if res:
          LOGGER.warning("%s update : %s", table.__tablename__, in_dict)
          query = session.query(table).filter(table.id == res.id)
          query.update(in_dict, synchronize_session='fetch')
          session.commit()
          return True, res.id

        LOGGER.warning("%s insert : %s", table.__tablename__, in_dict)
        entry = table(**in_dict)
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return True, entry.id

      except OperationalError as error:
        handle_op_error(LOGGER, error)
      except IntegrityError as error:
        session.rollback()
        LOGGER.warning('insert failed (%s) attempt %s, retrying ... ', error,
                       idx)
        sleep(5)

    return False, -1


def mysqldb_overwrite_table(table, dict_list, filter_cols):
  """insert/update rows to mysql"""
  insert_ids = []
  ret = False
  for i, in_dict in enumerate(dict_list):
    filter_dict = {}
    for col in filter_cols:
      filter_dict[col] = in_dict[col]

    ret, ins_id = mysqldb_insert_dict(table, in_dict, filter_dict)
    if ret:
      insert_ids.append(ins_id)
    else:
      return False, insert_ids

    if i % 1000 == 0:
      LOGGER.info('Inserting sql... %s', i)

  return ret, insert_ids


def handle_op_error(logger, error):
  """error handling for sql OperationalError"""
  if error.orig.args[0] == MYSQL_LOCK_WAIT_TIMEOUT:
    logger.warning('Db contention, sleeping ...')
    sleep(random.randint(1, 30))
  else:
    raise error


def get_db_id(db_elems, config_table):
  """Return unique DB id for config dict"""
  cid = None
  query = Query(config_table.id).filter_by(**db_elems)
  with DbSession() as session:
    try:
      res = session.execute(query).fetchall()
    except IntegrityError as err:
      session.rollback()
      LOGGER.error("Error occurred: %s \n", err)
      return False
    if res:
      cid = res[0][0]
  return cid


def set_job_state(session, job, dbt, state, increment_retries=False, result=""):
  """Update job state for builder/evaluator job_set_attr: List[str]"""

  LOGGER.info('Setting job id %s state to %s', job.id, state)
  job_set_attr = ['state', 'gpu_id']
  job.state = state
  if result:
    job_set_attr.append('result')
    job.result = result
  if increment_retries:
    job_set_attr.append('retries')
    job.retries += 1

  #pylint: disable=duplicate-code
  if '_start' in state:
    job_set_attr.append('cache_loc')
    cache: str = '~/.cache/miopen_'
    blurr: str = ''.join(
        random.choice(string.ascii_lowercase) for i in range(10))
    cache_loc: str = cache + blurr
    job.cache_loc = cache_loc
  #pylint: enable=duplicate-code

  query: str = gen_update_query(job, job_set_attr, dbt.job_table.__tablename__)

  def callback() -> bool:
    session.execute(query)
    session.commit()
    return True

  assert session_retry(session, callback, lambda x: x(), LOGGER)
  return True
