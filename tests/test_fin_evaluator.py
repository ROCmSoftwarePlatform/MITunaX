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
import json
import os
import sys
import copy
from sqlalchemy.inspection import inspect

from utils import CfgImportArgs, LdJobArgs, GoFishArgs
from utils import get_worker_args, add_test_session
from tuna.dbBase.sql_alchemy import DbSession
from tuna.miopen.db.tables import MIOpenDBTables
from tuna.miopen.miopen_lib import MIOpen
from tuna.miopen.subcmd.import_configs import import_cfgs
from tuna.miopen.subcmd.load_job import add_jobs
from tuna.miopen.utils.config_type import ConfigType
from tuna.miopen.utils.metadata import ALG_SLV_MAP
from tuna.miopen.worker.fin_class import FinClass
from tuna.miopen.db.solver import get_solver_ids
from tuna.utils.db_utility import connect_db
from tuna.utils.logger import setup_logger
from tuna.utils.machine_utility import load_machines
from tuna.miopen.celery_tuning.celery_tasks import prep_kwargs
from tuna.miopen.utils.lib_helper import get_worker
from tuna.utils.utility import serialize_job_config_row
from tuna.libraries import Operation
from tuna.miopen.celery_tuning.celery_tasks import prep_worker

sys.path.append("../tuna")
sys.path.append("tuna")

this_path = os.path.dirname(__file__)

solver_id_map = get_solver_ids()


def add_cfgs():
  #import configs
  args = CfgImportArgs()
  args.tag = 'tuna_pytest_fin_eval'
  args.mark_recurrent = True
  args.file_name = f"{this_path}/../utils/configs/conv_configs.txt"

  dbt = MIOpenDBTables(config_type=args.config_type)
  import_cfgs(args, dbt, setup_logger('test_fin_eval'))
  return dbt


def add_fin_find_eval_job(session_id, dbt):
  #load jobs
  args = LdJobArgs
  args.label = 'tuna_pytest_fin_eval'
  args.tag = 'tuna_pytest_fin_eval'
  args.fin_steps = ['miopen_find_eval']
  args.session_id = session_id
  logger = setup_logger('test_add_fin_find_eval_job')

  #limit job scope
  args.algo = "miopenConvolutionAlgoDirect"
  solver_arr = ALG_SLV_MAP[args.algo]
  if solver_arr:
    solver_ids = []
    for solver in solver_arr:
      sid = solver_id_map.get(solver, None)
      solver_ids.append((solver, sid))
    args.solvers = solver_ids

  connect_db()
  return add_jobs(args, dbt, logger)


def add_fake_fdb_entries(job_query, dbt, kernel_group):

  with DbSession() as session:
    kernel_obj = dbt.kernel_cache()

    kernel_obj.kernel_group = kernel_group
    kernel_obj.kernel_name = 'placeholder'
    kernel_obj.kernel_args = 'no-args'
    kernel_obj.kernel_blob = bytes('nothing_here', 'utf-8')
    kernel_obj.kernel_hash = '0'
    kernel_obj.uncompressed_size = '0'

    session.add(kernel_obj)
    session.commit()

    job_entries = job_query.all()
    for entry in job_entries:
      fdb_entry = dbt.find_db_table()
      fdb_entry.config = entry.config
      fdb_entry.solver = solver_id_map.get(entry.solver)
      fdb_entry.opencl = False
      fdb_entry.session = dbt.session.id
      fdb_entry.fdb_key = 'nil'
      fdb_entry.params = 'nil'
      fdb_entry.kernel_time = -1
      fdb_entry.workspace_sz = 0
      fdb_entry.kernel_group = kernel_group
      session.add(fdb_entry)

    session.commit()


def test_fin_evaluator():
  miopen = MIOpen()
  miopen.args = GoFishArgs()
  machine_lst = load_machines(miopen.args)
  machine = machine_lst[0]
  miopen.args.label = 'tuna_pytest_fin_eval'
  miopen.args.session_id = add_test_session(label='tuna_pytest_fin_eval')

  #update solvers
  kwargs = get_worker_args(miopen.args, machine, miopen)
  fin_worker = FinClass(**kwargs)
  assert fin_worker.get_solvers()

  add_cfgs()
  dbt = MIOpenDBTables(config_type=ConfigType.convolution,
                       session_id=miopen.args.session_id)

  args = GoFishArgs()
  machine_lst = load_machines(args)
  miopen.args.update_applicability = True

  worker_lst = miopen.compose_worker_list(machine_lst)
  for worker in worker_lst:
    worker.join()

  #load jobs
  args = LdJobArgs
  args.label = 'tuna_pytest_fin_eval'
  args.tag = 'tuna_pytest_fin_eval'
  args.fin_steps = ['miopen_find_eval']
  args.session_id = miopen.args.session_id

  logger = setup_logger('test_fin_evaluator')
  num_jobs = add_jobs(args, dbt, logger)
  assert num_jobs > 0

  miopen.args.fin_steps = ["miopen_find_eval"]
  miopen.args.label = 'tuna_pytest_fin_eval'
  miopen.fetch_state.add('new')
  miopen.operation = Operation.EVAL
  miopen.set_state = 'eval_start'
  miopen.dbt = MIOpenDBTables(session_id=miopen.args.session_id,
                              config_type=ConfigType.convolution)
  with DbSession() as session:
    jobs = miopen.get_jobs(session, miopen.fetch_state, miopen.set_state,
                           miopen.args.session_id)
  entries = list(jobs)
  job_config_rows = miopen.compose_work_objs_fin(session, entries, miopen.dbt)
  assert job_config_rows

  f_vals = miopen.get_f_vals(machine, range(0))
  kwargs = miopen.get_kwargs(0, f_vals, tuning=True)

  kwargs['avail_gpus'] = 1
  fdb_attr = [column.name for column in inspect(miopen.dbt.find_db_table).c]
  fdb_attr.remove("insert_ts")
  fdb_attr.remove("update_ts")

  res_set = []
  for elem in job_config_rows:
    job_dict, config_dict = serialize_job_config_row(elem)
    context = {
        'job': job_dict,
        'config': config_dict,
        'operation': miopen.operation,
        'arch': miopen.dbt.session.arch,
        'num_cu': miopen.dbt.session.num_cu,
        'kwargs': kwargs,
        'fdb_attr': fdb_attr
    }

    worker = prep_worker(copy.deepcopy(context))
    worker.dbt = miopen.dbt
    worker.fin_steps = miopen.args.fin_steps
    fin_json = worker.run()
    res_set.append((fin_json, context))

  with DbSession() as session:
    for fin_json, context in res_set:
      #testing process_fin_evaluator results
      miopen.process_fin_evaluator_results(session, fin_json, context)

  with DbSession() as session:
    valid_fin_err = session.query(dbt.job_table).filter(dbt.job_table.session==miopen.args.session_id)\
                                         .filter(dbt.job_table.state=='errored')\
                                         .filter(dbt.job_table.result.contains('%Find Compile: No results%'))\
                                         .count()
    #ommiting valid Fin/MIOpen errors
    num_jobs = num_jobs - valid_fin_err
    count = session.query(dbt.job_table).filter(dbt.job_table.session==miopen.args.session_id)\
                                         .filter(dbt.job_table.state=='evaluated').count()
    assert count == num_jobs

  assert kwargs['fin_steps'] == ['miopen_find_eval']

  job_config = job_config_rows[0]
  job_dict, config_dict = serialize_job_config_row(job_config)
  #testing prep_kwargs
  worker_kwargs = prep_kwargs(
      context['kwargs'],
      [context['job'], context['config'], context['operation']])
  assert worker_kwargs['config']
  assert worker_kwargs['job']
  assert worker_kwargs['fin_steps'] == ['miopen_find_eval']
  fin_eval = get_worker(worker_kwargs, miopen.operation)

  #testing check_gpu
  fin_eval.check_gpu()

  # test get_fin_input
  file_name = fin_eval.get_fin_input()
  assert file_name

  find_eval_file = f"{this_path}/../utils/test_files/fin_output_find_eval.json"
  fin_json = json.loads(machine.read_file(find_eval_file))[1:]
  assert len(fin_json) == 1
