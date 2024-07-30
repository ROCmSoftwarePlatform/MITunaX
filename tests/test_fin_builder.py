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

<<<<<<< HEAD
import copy
from sqlalchemy.inspection import inspect
=======
import os
import sys
import copy

sys.path.append("../tuna")
sys.path.append("tuna")

this_path = os.path.dirname(__file__)
>>>>>>> develop

from sqlalchemy.inspection import inspect

from tuna.dbBase.sql_alchemy import DbSession
<<<<<<< HEAD
=======
from tuna.utils.machine_utility import load_machines
>>>>>>> develop
from tuna.miopen.db.tables import MIOpenDBTables
from tuna.miopen.miopen_lib import MIOpen
<<<<<<< HEAD
from tuna.miopen.utils.config_type import ConfigType
from tuna.utils.utility import serialize_job_config_row
from tuna.libraries import Operation
from tuna.miopen.celery_tuning.celery_tasks import prep_worker
from tuna.machine import Machine
from utils import GoFishArgs
from utils import add_test_session, add_test_jobs
=======
from tuna.miopen.utils.metadata import ALG_SLV_MAP
from tuna.miopen.db.solver import get_solver_ids
from tuna.utils.logger import setup_logger
from tuna.miopen.utils.config_type import ConfigType
from tuna.utils.utility import serialize_job_config_row
from tuna.miopen.celery_tuning.celery_tasks import prep_kwargs
from tuna.machine import Machine
from tuna.miopen.utils.lib_helper import get_worker
from tuna.libraries import Operation
from tuna.miopen.celery_tuning.celery_tasks import prep_worker


def add_cfgs():
  #import configs
  args = CfgImportArgs()
  args.tag = 'tuna_pytest_fin_builder'
  args.mark_recurrent = True
  args.file_name = f"{this_path}/../utils/configs/conv_configs_NCHW.txt"

  dbt = MIOpenDBTables(config_type=args.config_type)
  counts = import_cfgs(args, dbt, setup_logger('test_fin_builder'))
  return dbt


def add_fin_find_compile_job(session_id, dbt):
  #load jobs
  args = LdJobArgs
  args.label = 'tuna_pytest_fin_builder'
  args.tag = 'tuna_pytest_fin_builder'
  args.fin_steps = ['miopen_find_compile', 'miopen_find_eval']
  args.session_id = session_id
  logger = setup_logger('test_add_fin_find_compile_job')

  #limit job scope
  args.algo = "miopenConvolutionAlgoGEMM"
  solver_arr = ALG_SLV_MAP[args.algo]
  solver_id_map = get_solver_ids()
  if solver_arr:
    solver_ids = []
    for solver in solver_arr:
      sid = solver_id_map.get(solver, None)
      solver_ids.append((solver, sid))
    args.solvers = solver_ids
  args.only_applicable = True

  connect_db()
  return add_jobs(args, dbt, logger)
>>>>>>> develop


def test_fin_builder():
  miopen = MIOpen()
  miopen.args = GoFishArgs()
  miopen.args.label = 'tuna_pytest_fin_builder'
  miopen.args.session_id = add_test_session(label=miopen.args.label)

  #load jobs
<<<<<<< HEAD
  dbt = MIOpenDBTables(config_type=ConfigType.convolution)
  num_jobs = add_test_jobs(miopen, miopen.args.session_id, dbt,
                           miopen.args.label, miopen.args.label,
                           ['miopen_find_compile', 'miopen_find_eval'],
                           'test_add_fin_find_compile_job',
                           'miopenConvolutionAlgoGEMM')
  assert (num_jobs)

  #testing process_fdb_compile in process_fin_builder_results
  miopen.args.update_applicability = False
  miopen.args.fin_steps = ["miopen_find_compile"]
=======
  miopen.args.label = 'tuna_pytest_fin_builder'
  num_jobs = add_fin_find_compile_job(miopen.args.session_id, dbt)
  assert (num_jobs)

  #compile
  miopen.args.update_applicability = False
  miopen.args.fin_steps = ["miopen_find_compile"]
  miopen.args.label = 'tuna_pytest_fin_builder'
>>>>>>> develop
  miopen.fetch_state.add('new')
  miopen.operation = Operation.COMPILE
  miopen.set_state = 'compile_start'
  miopen.dbt = MIOpenDBTables(session_id=miopen.args.session_id,
                              config_type=ConfigType.convolution)
  jobs = None
  with DbSession() as session:
    jobs = miopen.get_jobs(session, miopen.fetch_state, miopen.set_state,
                           miopen.args.session_id)
<<<<<<< HEAD
  entries = list(jobs)
  job_config_rows = miopen.compose_work_objs_fin(session, entries, miopen.dbt)
  assert job_config_rows
=======
  entries = [job for job in jobs]
  job_config_rows = miopen.compose_work_objs_fin(session, entries, miopen.dbt)
  assert (job_config_rows)
>>>>>>> develop

  f_vals = miopen.get_f_vals(Machine(local_machine=True), range(0))
  kwargs = miopen.get_kwargs(0, f_vals, tuning=True)
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
      miopen.process_fin_builder_results(session, fin_json, context)

  with DbSession() as session:
    valid_fin_err = session.query(dbt.job_table).filter(dbt.job_table.session==miopen.args.session_id)\
                                         .filter(dbt.job_table.state=='errored')\
                                         .filter(dbt.job_table.result.contains('%Find Compile: No results%'))\
                                         .count()
    #ommiting valid Fin/MIOpen errors
    num_jobs = (num_jobs - valid_fin_err)
    count = session.query(dbt.job_table).filter(dbt.job_table.session==miopen.args.session_id)\
                                         .filter(dbt.job_table.state=='compiled').count()
    assert count == num_jobs
