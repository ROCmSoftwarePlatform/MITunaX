#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE.
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################
"""Utility module for parsing fin json results"""
import functools
from sqlalchemy.exc import OperationalError

from tuna.utils.logger import setup_logger
from tuna.dbBase.sql_alchemy import DbSession
from tuna.utils.utility import SimpleDict
from tuna.utils.db_utility import session_retry, gen_select_objs
from tuna.utils.db_utility import gen_update_query, gen_insert_query
from tuna.miopen.worker.fin_utils import get_fin_slv_status
from tuna.miopen.utils.parsing import parse_pdb_key
from tuna.miopen.db.solver import get_solver_ids

LOGGER = setup_logger('parse_results')


def __update_fdb_w_kernels(  #pylint: disable=too-many-arguments,too-many-locals
    session: DbSession,
    fin_json,
    config,
    session_id,
    dbt,
    job,
    fdb_attr,
    pending,
    result_str: str = 'miopen_find_compile_result',
    check_str: str = 'find_compiled') -> list:
  """update find db + kernels from json results"""
  status = []
  if dbt.config_type != ConfigType.convolution:
    return [{
        'solver': 'all',
        'success': False,
        'result': 'Find Compile: BN FDB key parsing not implemented'
    }]

  solver_id_map = get_solver_ids()
  if result_str in fin_json.keys():
    for fdb_obj in fin_json.get(result_str):
      slv_stat = get_fin_slv_status(fdb_obj, check_str)
      status.append(slv_stat)

      if fdb_obj[check_str]:
        #returned entry is added to the table
        fdb_entry = __compose_fdb_entry(session, fin_json, fdb_obj, session_id,
                                        dbt, config, job, fdb_attr,
                                        solver_id_map, pending)
        __check_layout_mismatch(fdb_entry, slv_stat, config, dbt)
        if not pending:
          query = gen_update_query(fdb_entry, fdb_attr,
                                   dbt.find_db_table.__tablename__)
          session.execute(query)
        else:
          assert len(pending) == 1
          pending.pop()
          query = gen_insert_query(fdb_entry, fdb_attr,
                                   dbt.find_db_table.__tablename__)
          session.execute(query)

          fdb_entry = __update_fdb_entry(session,
                                         solver_id_map[fdb_obj['solver_name']],
                                         session_id, dbt, config, job, fdb_attr,
                                         pending)
          fdb_entry.kernel_group = fdb_entry.id
          query = gen_update_query(fdb_entry, ['kernel_group'],
                                   dbt.find_db_table.__tablename__)
          session.execute(query)

        if fdb_obj['reason'] == 'Success':
          __compose_kernel_entry(session, fdb_obj, fdb_entry, dbt)
          LOGGER.info('Updating find Db(Build) for job_id=%s', job.id)
        else:
          # JD: add info about reason to the logs table
          fdb_entry.valid = False
      else:
        LOGGER.warning("Failed find_db compile, cfg_id: %s, obj: %s",
                       fin_json['config_tuna_id'], fdb_obj)
  else:
    status = [{
        'solver': 'all',
        'success': False,
        'result': 'Find Compile: No results'
    }]

  session.commit()

  return status


def process_pdb_compile(session, fin_json, job, dbt, solver_id_map):
  """retrieve perf db compile json results"""
  status = []
  if dbt.config_type != ConfigType.convolution:
    LOGGER.warning('BN PDB key parsing not implemented')
    return [{
        'solver': 'all',
        'success': False,
        'result': 'Perf Compile: BN PDB key parsing not implemented'
    }]
  if fin_json['miopen_perf_compile_result']:

    def actuator(func, pdb_obj, dbt, job, solver_id_map):
      return func(session, pdb_obj, dbt, job, solver_id_map)

    for pdb_obj in fin_json['miopen_perf_compile_result']:
      slv_stat = get_fin_slv_status(pdb_obj, 'perf_compiled')
      status.append(slv_stat)
      if pdb_obj['perf_compiled']:
        session_retry(
            session, compose_job_cache_entrys,
            functools.partial(actuator,
                              pdb_obj=pdb_obj,
                              dbt=dbt,
                              job=job,
                              solver_id_map=solver_id_map), LOGGER)
        LOGGER.info('Updating pdb job_cache for job_id=%s', job.id)
  else:
    status = [{
        'solver': 'all',
        'success': False,
        'result': 'Perf Compile: No results'
    }]

  return status


def compose_job_cache_entrys(session, pdb_obj, dbt, job, solver_id_map):
  """Compose new pdb kernel cache entry from fin input"""
  for kern_obj in pdb_obj['kernel_objects']:
    kernel_obj = dbt.fin_cache_table()
    populate_kernels(kern_obj, kernel_obj)
    kernel_obj.solver_id = solver_id_map[pdb_obj['solver_name']]
    kernel_obj.job_id = job.id

    session.add(kernel_obj)
  session.commit()

  return True


def populate_kernels(kern_obj, kernel_obj):
  """populate kernel object"""
  kernel_obj.kernel_name = kern_obj['kernel_file']
  kernel_obj.kernel_args = kern_obj['comp_options']
  kernel_obj.kernel_blob = bytes(kern_obj['blob'], 'utf-8')
  kernel_obj.kernel_hash = kern_obj['md5_sum']
  kernel_obj.uncompressed_size = kern_obj['uncompressed_size']
  return kernel_obj


def __check_layout_mismatch(fdb_entry: SimpleDict, status: dict, config,
                            dbt) -> bool:
  """Check that the fdb key returned by fin matches the config being tuned,
  states to error if not"""
  fdb_key = fdb_entry.fdb_key
  try:
    fds, vals, _, _ = parse_pdb_key(fdb_key, dbt.config_type)
  except CustomError as err:
    print(err)
    return False

  if dbt.config_type == ConfigType.convolution:
    __check_layout_mismatch_conv(fds, vals, fdb_entry, status, config, dbt)
  elif dbt.config_type == ConfigType.batch_norm:
    __check_layout_mismatch_bn(fds, vals, fdb_entry, status, config, dbt)
  else:
    raise CustomError(
        'layout missmatch check for config type: %s not implemented',
        dbt.config_type)


def __check_layout_mismatch_conv(fdb_entry: SimpleDict, status: dict,
                                 config: dict, dbt) -> bool:
  """Check layout missmatch for conv"""

  key_layout = vals[fds.index('out_layout')]
  cfg_layout = config.out_layout

  if cfg_layout != key_layout:
    status['success'] = False
    status['result'] = f"fdb_key layout mismatch with config"\
                       f" {key_layout} != {cfg_layout}"
    fdb_entry.valid = False
    return False

  return True


def __check_layout_mismatch_bn(fdb_entry: SimpleDict, status: dict, config,
                               dbt) -> bool:
  """Check layout missmatch for BN"""
  raise NotImplementedError("Not implemented")


def __compose_kernel_entry(session, fdb_obj, fdb_entry, dbt):
  """Compose a new Kernel Cache entry from fin input"""
  # Now we have the ID, lets add the binary cache objects
  for kern_obj in fdb_obj['kernel_objects']:
    kernel_obj = dbt.kernel_cache()
    populate_kernels(kern_obj, kernel_obj)
    kernel_obj.kernel_group = fdb_entry.kernel_group
    session.add(kernel_obj)
  return True


def __update_fdb_entry(session, solver, session_id, dbt, config, job, fdb_attr,
                       pending):
  """ Add a new entry to fdb if there isnt one already """
  obj, fdb_entry = get_fdb_entry(session, solver, session_id, dbt, config,
                                 fdb_attr)
  if obj:  # existing entry in db
    # This can be removed if we implement the delete orphan cascade
    fdb_entry = obj
    if not fdb_entry.kernel_group is None:
      LOGGER.info('Invalidate kernel_group %s', fdb_entry.kernel_group)
      session.query(dbt.kernel_cache)\
          .filter(dbt.kernel_cache.valid == 1)\
          .filter(dbt.kernel_cache.kernel_group ==
                                        fdb_entry.kernel_group)\
          .update({'valid': 0})
  else:
    # Bundle Insert for later
    pending.append((job, fdb_entry))
  return fdb_entry


def get_fdb_entry(session, solver, session_id, dbt, config, fdb_attr):
  """ Get FindDb entry from db """
  obj = None
  fdb_entry = None

  conds = [
      f"session={session_id}", f"config={config.id}", f"solver={solver}",
      "opencl=0"
  ]
  cond_str = f"where {' AND '.join(conds)}"
  entries = gen_select_objs(session, fdb_attr, dbt.find_db_table.__tablename__,
                            cond_str)

  if entries:
    assert len(entries) == 1
    obj = entries[0]
  else:
    fdb_entry = SimpleDict()
    for attr in fdb_attr:
      setattr(fdb_entry, attr, None)
    setattr(fdb_entry, 'session', session_id)
    setattr(fdb_entry, 'config', config.id)
    setattr(fdb_entry, 'solver', solver)
    setattr(fdb_entry, 'opencl', False)
    setattr(fdb_entry, 'logger', LOGGER)

  return obj, fdb_entry


def __compose_fdb_entry(  #pylint: disable=too-many-arguments
    session, fin_json, fdb_obj, session_id, dbt, config, job, fdb_attr,
    solver_id_map, pending):
  """Compose a FindDB table entry from fin_output"""
  solver = solver_id_map[fdb_obj['solver_name']]
  fdb_entry = __update_fdb_entry(session, solver, session_id, dbt, config, job,
                                 fdb_attr, pending)
  fdb_entry.fdb_key = fin_json['db_key']
  fdb_entry.alg_lib = fdb_obj['algorithm']
  fdb_entry.params = fdb_obj['params']
  fdb_entry.workspace_sz = fdb_obj['workspace']
  fdb_entry.valid = True

  fdb_entry.kernel_time = -1
  if 'time' in fdb_obj:
    fdb_entry.kernel_time = fdb_obj['time']

  fdb_entry.kernel_group = fdb_entry.id

  return fdb_entry


def process_fdb_w_kernels(session,
                          fin_json,
                          context,
                          dbt,
                          fdb_attr,
                          pending,
                          result_str='miopen_find_compile_result',
                          check_str='find_compiled'):
  """initiate find db update"""
  job = SimpleDict(**context['job'])
  #get_db_obj_by_id(context['job']['id'], dbt.job_table)
  config = SimpleDict(**context['config'])
  #get_db_obj_by_id(context['config']['id'], dbt.config_table)

  callback = __update_fdb_w_kernels
  status = session_retry(
      session, callback,
      lambda x: x(session, fin_json, config, context['kwargs']['session_id'],
                  dbt, job, fdb_attr, pending, result_str, check_str), LOGGER)

  if not status:
    LOGGER.warning('Fin: Unable to update Database')
    status = [{
        'solver': 'all',
        'success': False,
        'result': 'Fin: Unable to update Database'
    }]

  return status


def clean_cache_table(dbt, job):
  """Remove the fin cache kernel entries for this job"""
  with DbSession() as session:
    try:
      LOGGER.info('Delete kernel cache entries job(%s)', job.id)
      job_cache = session.query(dbt.fin_cache_table)\
          .filter(dbt.fin_cache_table.job_id == job.id)
      job_cache.delete()
      invalid_fdb_cache = session.query(dbt.kernel_cache)\
          .filter(dbt.kernel_cache.valid == 0)
      invalid_fdb_cache.delete()
      session.commit()
    except OperationalError as err:
      session.rollback()
      LOGGER.warning('FinEval: Unable to clean %s / %s: %s',
                     dbt.fin_cache_table.__tablename__,
                     dbt.kernel_cache.__tablename__, err)
