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
"""MIOpen class that holds MIOpen specifig  tuning functionality"""

import sys
import copy
import json
from typing import List, Tuple, Any
from functools import lru_cache
from collections.abc import Iterable

from kombu.utils.uuid import uuid
from sqlalchemy.inspection import inspect
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.exc import OperationalError, DataError, IntegrityError
from tuna.mituna_interface import MITunaInterface
from tuna.miopen.utils.helper import print_solvers
from tuna.parse_args import TunaArgs, setup_arg_parser, args_check

from tuna.dbBase.sql_alchemy import DbSession
from tuna.tables_interface import DBTablesInterface
from tuna.utils.utility import SimpleDict, serialize_chunk
from tuna.utils.machine_utility import load_machines
from tuna.utils.db_utility import gen_select_objs, has_attr_set, get_class_by_tablename
from tuna.utils.db_utility import gen_update_query, session_retry, attach_tensor_info
from tuna.miopen.db.get_db_tables import get_miopen_tables
from tuna.miopen.db.mixin_tables import FinStep
from tuna.miopen.utils.metadata import MIOPEN_ALG_LIST
from tuna.miopen.metadata import MIOPEN_CELERY_STEPS
from tuna.miopen.worker.fin_class import FinClass
from tuna.miopen.db.session import Session
from tuna.miopen.subcmd.import_configs import run_import_configs
from tuna.miopen.subcmd.load_job import run_load_job
from tuna.miopen.subcmd.export_db import run_export_db
from tuna.miopen.subcmd.update_golden import run_update_golden
from tuna.miopen.parse_miopen_args import get_import_cfg_parser, get_load_job_parser
from tuna.miopen.parse_miopen_args import get_export_db_parser, get_update_golden_parser
from tuna.miopen.db.build_schema import create_tables, recreate_triggers
from tuna.miopen.db.triggers import drop_miopen_triggers, get_miopen_triggers
from tuna.miopen.utils.config_type import ConfigType
from tuna.miopen.db.tables import MIOpenDBTables
from tuna.machine import Machine
#from tuna.miopen.celery_tuning.celery_tasks import celery_enqueue
from tuna.miopen.utils.json_to_sql import process_fdb_w_kernels, process_pdb_compile
from tuna.miopen.utils.json_to_sql import clean_cache_table
from tuna.miopen.utils.helper import set_job_state
from tuna.miopen.worker.fin_utils import get_fin_result
from tuna.miopen.db.solver import get_solver_ids
from tuna.libraries import Library, Operation
from tuna.custom_errors import CustomError

MAX_ERRORED_JOB_RETRIES = 3
Q_NAME = None


class MIOpen(MITunaInterface):
  """Class to support MIOpen specific tuning functionality"""

  # pylint: disable=too-many-public-methods

  def __init__(self):
    super().__init__(library=Library.MIOPEN)
    self.args = None
    self.set_state = None

  def parse_args(self):
    # pylint: disable=too-many-statements
    """Function to parse arguments"""
    parser = setup_arg_parser(
        'Run Performance Tuning on a certain architecture', [
            TunaArgs.ARCH, TunaArgs.NUM_CU, TunaArgs.VERSION,
            TunaArgs.CONFIG_TYPE, TunaArgs.SESSION_ID, TunaArgs.MACHINES,
            TunaArgs.REMOTE_MACHINE, TunaArgs.LABEL, TunaArgs.RESTART_MACHINE,
            TunaArgs.DOCKER_NAME, TunaArgs.SHUTDOWN_WORKERS
        ])
    parser.add_argument(
        '--find_mode',
        dest='find_mode',
        type=int,
        default=1,
        help='Set the MIOPEN_FIND_MODE environment variable for MIOpen',
        choices=['1', '3'])
    parser.add_argument('--ticket',
                        dest='ticket',
                        type=str,
                        default=None,
                        help='Specify tuning ticket number')
    parser.add_argument(
        '--solver_id',
        type=int,
        dest='solver_id',
        default=None,
        help='Specify solver_id. Use --list_solvers to see options')
    parser.add_argument('--dynamic_solvers_only',
                        dest='dynamic_solvers_only',
                        action='store_true',
                        default=False,
                        help='Only tune dynamic solvers.')
    parser.add_argument(
        '-B',
        '--blacklist',
        dest='blacklist',
        type=str,
        default=None,
        help='MIOpen blacklist algorithm, if multiple then comma separate')
    parser.add_argument('-i',
                        '--reset_interval',
                        type=int,
                        dest='reset_interval',
                        required=False,
                        help='Restart interval for job in hours.')
    parser.add_argument(
        '--gpu_lim',
        dest='gpu_lim',
        type=int,
        default=None,
        help='Limit the number of gpu workers created by Tuna, index from 0')

    parser.add_argument('--enqueue_only',
                        action='store_true',
                        dest='enqueue_only',
                        help='Enqueue jobs to celery queue')

    subcommands = parser.add_subcommands(required=False)
    subcommands.add_subcommand('import_configs',
                               get_import_cfg_parser(),
                               required=False)

    subcommands.add_subcommand('load_job',
                               get_load_job_parser(),
                               required=False)

    subcommands.add_subcommand('export_db',
                               get_export_db_parser(),
                               required=False)

    subcommands.add_subcommand('update_golden',
                               get_update_golden_parser(),
                               required=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--add_tables',
                       dest='add_tables',
                       action='store_true',
                       help='Add MIOpen library specific tables')

    group.add_argument('--init_session',
                       action='store_true',
                       dest='init_session',
                       help='Set up a new tuning session.')
    group.add_argument(
        '--fin_steps',
        type=str,
        dest='fin_steps',
        help='Specify fin steps. Multiple steps should be comma separated.')
    group.add_argument('--list_solvers',
                       action='store_true',
                       dest='list_solvers',
                       help='List of solvers from the solver table')

    # JD: implement the following two using fin_steps
    group.add_argument('--update_solvers',
                       dest='update_solvers',
                       action='store_true',
                       help='Update the list of solvers in the database')
    group.add_argument('--update_applicability',
                       dest='update_applicability',
                       action='store_true',
                       help='Update the applicability table in the database')
    group.add_argument('-s',
                       '--status',
                       dest='check_status',
                       action='store_true',
                       default=False,
                       help='Check the status of machines')

    group.add_argument('-e',
                       '--exec',
                       dest='execute_cmd',
                       type=str,
                       default=None,
                       help='execute on each machine')

    self.args = parser.parse_args()

    if self.args.config_type is None:
      self.args.config_type = ConfigType.convolution

    #overwritte common lib args with subcommand args value
    if self.args.subcommand is not None:
      self.overwrite_common_args()

    if len(sys.argv) == 1:
      parser.print_help()
      sys.exit(-1)

    if self.args.list_solvers:
      print_solvers()
      raise CustomError('Printing solvers...')

    if self.args.fin_steps and self.args.subcommand != 'load_job':
      self.check_fin_args(parser)
      self.set_prefix()

    if self.args.find_mode is None and not (self.args.check_status or
                                            self.args.restart_machine or
                                            self.args.execute_cmd):
      parser.error('find_mode must be specified for a tuning run')

    if self.args.blacklist:
      self.check_blacklist(parser)

    args_check(self.args, parser)

    fin_session_steps = [
        'miopen_find_compile', 'miopen_find_eval', 'miopen_perf_compile',
        'miopen_perf_eval', 'get_applicability', 'find_compile', 'find_eval'
    ]
    has_fin = False
    if self.args.fin_steps:
      has_fin = all(x in fin_session_steps for x in self.args.fin_steps)

    if (self.args.update_applicability or has_fin) and not self.args.session_id:
      parser.error("session_id must be specified with this operation")

    self.dbt = MIOpenDBTables(session_id=self.args.session_id,
                              config_type=self.args.config_type)
    self.update_operation()

  def set_prefix(self):
    """Set redis key prefix"""
    if isinstance(self.args.fin_steps, Iterable):
      steps_str = ('-').join(x for x in self.args.fin_steps)
      self.prefix = f"d_{self.db_name}_sess_{self.args.session_id}_"\
                    f"{steps_str}"
    else:
      steps_str = self.args.fin_steps[0]
      self.prefix = f"d_{self.db_name}_sess_{self.args.session_id}_{steps_str}"

    self.logger.info('redis prefix: %s', self.prefix)

  def overwrite_common_args(self):
    """Overwrite common MIOpen_lib args with subcommand args"""
    if self.args.subcommand is not None:
      subc_dict = vars(self.args.get(self.args.subcommand))
      for sub_key in subc_dict:
        if sub_key in vars(self.args):
          self.args[sub_key] = subc_dict.get(sub_key)

  def check_fin_args(self, parser):
    """! Helper function for fin args
       @param parser The command line argument parser
        """
    valid_fin_steps = list(k for k in FinStep.__members__)
    if ',' in self.args.fin_steps:
      parser.error('Multiple fin_steps currently not supported')
    f_steps = self.args.fin_steps.split(',')
    self.args.fin_steps = f_steps
    for step in self.args.fin_steps:
      if step not in valid_fin_steps:
        parser.error(f"Supported fin steps are: {valid_fin_steps}")
    assert len(self.args.fin_steps) == 1

  def check_blacklist(self, parser):
    """! Helper function
       @param parser The command line argument parser
    """
    self.args.blacklist = self.args.blacklist.split(',')
    for sol in self.args.blacklist:
      if sol not in MIOPEN_ALG_LIST:
        parser.error("Incorrect blacklist value")

  def do_fin_work(self, gpu, f_vals):
    """! Helper function to execute job independendent fin work
      @param gpu Unique ID of the GPU
      @param f_vals Dict containing runtime information
    """
    kwargs = self.get_kwargs(gpu, f_vals)
    fin_worker = FinClass(**kwargs)

    if self.args.update_solvers:
      if not fin_worker.get_solvers():
        self.logger.error('No solvers returned from Fin class')

    return True

  def launch_worker(self, gpu_idx, f_vals, worker_lst):
    """! Function to launch worker
      @param gpu_idx Unique ID of the GPU
      @param f_vals Dict containing runtime information
      @param worker_lst List containing worker instances
      @retturn ret Boolean value
    """
    # pylint: disable=too-many-branches
    worker = None
    kwargs = self.get_kwargs(gpu_idx, f_vals)
    if self.args.update_applicability:
      kwargs['fin_steps'] = ['applicability']
      worker = FinClass(**kwargs)
      worker.start()
      worker_lst.append(worker)
      return True

    worker = FinClass(**kwargs)
    ret = False
    if self.args.check_status:
      if not super().check_status(worker, f_vals["b_first"], gpu_idx,
                                  f_vals["machine"], self.args.docker_name):
        ret = True
    elif self.args.init_session:
      Session().add_new_session(self.args, worker)
    elif self.args.execute_cmd:
      # JD: Move the worker.exec_command to machine
      self.logger.info(self.args.execute_cmd)
      _, _, _ = worker.exec_command(self.args.execute_cmd + " 2>&1 ")

    return ret

  def compose_worker_list(self, machines):
    # pylint: disable=too-many-branches
    """! Helper function to compose worker_list
      @param res DB query return item containg available machines
      @param args The command line arguments
    """
    worker_lst = []
    fin_work_done = False
    for machine in machines:
      if self.args.restart_machine:
        machine.restart_server(wait=False)
        continue

      #fin_steps should only contain one step
      worker_ids = None
      if self.args.fin_steps and 'eval' in self.args.fin_steps[0]:
        worker_ids = machine.get_avail_gpus()
        if self.args.gpu_lim and self.args.gpu_lim < len(worker_ids):
          worker_ids = range(self.args.gpu_lim)
      else:
        worker_ids = super().get_num_procs(machine)

      if self.args.update_applicability:
        f_vals = super().get_f_vals(machine, [1])
        kwargs = self.get_kwargs(0, f_vals)
        kwargs['fin_steps'] = ['applicability']
        worker = FinClass(**kwargs)
        query = worker.query_cfgs(self.args.label)
        cfg_rows = query.all()
        len_rows = len(cfg_rows)
        proc_lim = (len_rows + 99) / 100
        while len(worker_ids) > proc_lim:
          worker_ids.pop()

      if len(worker_ids) == 0:
        return None

      f_vals = super().get_f_vals(machine, worker_ids)

      if (self.args.update_solvers) and not fin_work_done:
        self.do_fin_work(0, f_vals)
        fin_work_done = True
        break

      for gpu_idx in worker_ids:
        self.logger.info('launch mid %u, proc %u', machine.id, gpu_idx)
        if not self.launch_worker(gpu_idx, f_vals, worker_lst):
          break

    return worker_lst

  def add_tables(self):
    ret_t = create_tables(get_miopen_tables())
    self.logger.info('DB creation successful: %s', ret_t)
    recreate_triggers(drop_miopen_triggers(), get_miopen_triggers())
    return True

  def run(self):
    # pylint: disable=duplicate-code
    """Main function to launch library"""
    res = None
    if self.args is None:
      self.parse_args()

    if self.args.add_tables:
      self.add_tables()
      return None

    if self.args.subcommand is not None and self.args.subcommand == 'import_configs':
      run_import_configs(self.args.import_configs, self.logger)
      return None

    if self.args.subcommand is not None and self.args.subcommand == 'load_job':
      run_load_job(self.args.load_job, self.logger)
      return None

    if self.args.subcommand is not None and self.args.subcommand == 'export_db':
      run_export_db(self.args.export_db, self.logger)
      return None

    if self.args.subcommand is not None and self.args.subcommand == 'update_golden':
      run_update_golden(self.args.update_golden, self.logger)
      return None

    machines = load_machines(self.args)
    res = self.compose_worker_list(machines)
    return res

  def get_envmt(self):
    """! Function to construct environment var
    """
    envmt = ["MIOPEN_LOG_LEVEL=4"]

    envmt.append("MIOPEN_SQLITE_KERN_CACHE=ON")
    envmt.append("MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS=1")

    if self.args.find_mode:
      envmt.append(f"MIOPEN_FIND_MODE={self.args.find_mode}")

    if self.args.blacklist:
      bk_str = ", ".join([f"{arg}=0" for arg in self.args.blacklist])
      for bk_var in bk_str.split(','):
        envmt.append(bk_var)

    return envmt

  def get_kwargs(self, gpu_idx, f_vals, tuning=False):
    """! Helper function to set up kwargs for worker instances
      @param gpu_idx Unique ID of the GPU
      @param f_vals Dict containing runtime information
      @param args The command line arguments
    """
    kwargs = super().get_kwargs(gpu_idx, f_vals, tuning)
    kwargs['fin_steps'] = self.args.fin_steps
    kwargs['dynamic_solvers_only'] = self.args.dynamic_solvers_only
    kwargs['config_type'] = self.args.config_type
    kwargs['reset_interval'] = self.args.reset_interval

    return kwargs

  def get_job_attr(self):
    """Get job attr for row selection"""
    job_attr: List[str]
    try:
      job_attr = [column.name for column in inspect(self.dbt.job_table).c]
      job_attr.remove("insert_ts")
      job_attr.remove("update_ts")
    except NoInspectionAvailable as error:
      self.logger.warning("Ignoring error for init_session: %s", error)
    return job_attr

  def get_jobs(self,
               session: DbSession,
               find_state: List[str],
               set_state: str,
               session_id: int,
               claim_num: int = None,
               no_update: bool = False):
    """Interface function to get jobs based on session and find_state"""
    #job_rows: List[SimpleDict]
    ids: list
    row: SimpleDict
    job_attr: List[str] = self.get_job_attr()

    self.logger.info('Fetching DB rows...')
    job_list = self.get_job_objs(session, find_state, self.args.label, self.dbt,
                                 job_attr, claim_num, self.args.fin_steps)

    if not self.check_jobs_found(job_list, find_state, session_id):
      return []

    if no_update:
      return job_list

    ids = [row.id for row in job_list]
    self.logger.info("%s jobs %s", find_state, ids)
    self.logger.info('Updating job state to %s', set_state)
    for job in job_list:
      job.state = set_state
      query: str = gen_update_query(job, ['state'],
                                    self.dbt.job_table.__tablename__)
      session.execute(query)

    session.commit()

    return job_list

  def get_job_objs(self,
                   session: DbSession,
                   find_state: list,
                   label: str,
                   dbt: DBTablesInterface,
                   job_attr: List[str],
                   claim_num: int = None,
                   fin_steps: List[str] = None) -> List[SimpleDict]:
    """Get list of job objects"""
    entries: List[Tuple[SimpleDict, ...]]
    conds: List[str] = [f"session={dbt.session.id}", "valid=1"]

    if label:
      conds.append(f"reason='{label}'")

    conds.append(f"retries<{self.max_job_retries}")
    conds.append("state in (" + str(find_state).strip('{').strip('}') + ")")

    entries = self.compose_work_objs(session, conds, dbt, job_attr, claim_num,
                                     fin_steps)
    return entries

  def compose_work_objs(self,
                        session: DbSession,
                        conds: List[str],
                        dbt: DBTablesInterface,
                        job_attr: List[str],
                        claim_num: int = None,
                        fin_steps: List[str] = None) -> List[SimpleDict]:
    """Query a job list for update"""
    ret = []
    if fin_steps:
      conds.append(f"fin_step like '%{fin_steps[0]}%'")
    else:
      conds.append("fin_step='not_fin'")

    cond_str = ' AND '.join(conds)
    if cond_str:
      cond_str = f"WHERE {cond_str}"
    if claim_num:
      cond_str += f" ORDER BY retries,config ASC LIMIT {claim_num} FOR UPDATE SKIP LOCKED"
    else:
      cond_str += " ORDER BY retries,config ASC FOR UPDATE SKIP LOCKED"

    #ret = get_job_rows(session, job_attr, dbt.job_table.__tablename__, cond_str)

    job_entries = gen_select_objs(session, job_attr,
                                  dbt.job_table.__tablename__, cond_str)

    ret = job_entries
    #if fin_steps:
    #  ret = self.compose_work_objs_fin(session, entries, dbt)
    #else:
    #  ret = entries

    #return ret
    return ret

  def compose_work_objs_fin(self, session, job_entries,
                            dbt) -> List[Tuple[SimpleDict, SimpleDict]]:
    """Return jobs for fin work"""
    ret = []

    cfg_rel = {
        key: {
            'key': list(val.local_columns)[0].name,
            'ftble': str(list(val.remote_side)[0]).split('.', maxsplit=1)[0],
            'fkey': str(list(val.remote_side)[0]).split('.')[1]
        } for key, val in inspect(dbt.config_table).relationships.items()
    }

    if job_entries:
      id_str = ','.join({str(job.config) for job in job_entries})
      cfg_cond_str = f"where valid=1 and id in ({id_str})"
      cfg_attr = [column.name for column in inspect(dbt.config_table).c]
      cfg_entries = gen_select_objs(session, cfg_attr,
                                    dbt.config_table.__tablename__,
                                    cfg_cond_str)

      cfg_entries = self.attach_tensors(session, cfg_rel, cfg_entries)

      cfg_map = {cfg.id: cfg for cfg in cfg_entries}

      for job in job_entries:
        ret.append((job, cfg_map[job.config]))

    return ret

  def attach_tensors(self, session, cfg_rel, cfg_entries):
    """attach tensor relationship information to config entries"""
    for _, val in cfg_rel.items():
      rel_attr = [
          column.name
          for column in inspect(get_class_by_tablename(val['ftble'])).c
      ]
      val['fattr'] = rel_attr

    attach_tensor_info(cfg_entries, session, cfg_rel)
    #for cfg in cfg_entries:
    #  for key, val in cfg_rel.items():
    #    rel_val = getattr(cfg, val['key'])
    #    rel_cond_str = f"where {val['fkey']}={rel_val}"
    #    setattr(
    #        cfg, key,
    #        gen_select_objs(session, val['fattr'], val['ftble'],
    #                       rel_cond_str)[0])
    return cfg_entries

  def check_jobs_found(self, job_rows: List[SimpleDict], find_state: List[Any],
                       session_id: int) -> bool:
    """check for end of jobs"""
    if not job_rows:
      # we are done
      self.logger.warning('No %s jobs found, session %s', find_state,
                          session_id)
      return False
    return True

  def get_job_tables(self, job_rows: List[Tuple[SimpleDict, ...]],
                     job_attr: List[str]) -> List[SimpleDict]:
    """find job tables in query results"""
    if has_attr_set(job_rows[0], job_attr):
      job_tables: List[SimpleDict] = job_rows
    else:
      job_i: int = 0
      tble: SimpleDict
      for i, tble in enumerate(job_rows[0]):
        if has_attr_set(tble, job_attr):
          job_i = i
          break
      job_tables = [row[job_i] for row in job_rows]

    return job_tables

  def update_operation(self):
    """Update the workers type that this library needs"""
    if self.args.fin_steps:
      if 'miopen_find_compile' in self.args.fin_steps \
      or 'miopen_perf_compile' in self.args.fin_steps:
        self.fetch_state.add('new')
        self.set_state = 'compile_start'
        self.operation = Operation.COMPILE
      elif 'miopen_find_eval' in self.args.fin_steps or 'miopen_perf_eval' in self.args.fin_steps:
        self.fetch_state.add('new')
        self.fetch_state.add('compiled')
        self.set_state = 'eval_start'
        self.operation = Operation.EVAL

    if self.args.update_applicability:
      self.fetch_state.add("new")

  def has_tunable_operation(self):
    """Check if its a tuning loop operation"""
    if self.args is None:
      self.parse_args()
    if self.args.subcommand and "load_job" in self.args.subcommand:
      return False
    if self.args.shutdown_workers:
      return True

    return self.args.fin_steps and any(
        s in self.args.fin_steps for s in MIOPEN_CELERY_STEPS)

  @lru_cache(1)
  def get_context_items(self):
    """Helper function to get items for celery job context"""
    f_vals = self.get_f_vals(Machine(local_machine=True), range(0), tuning=True)
    kwargs = self.get_kwargs(0, f_vals, tuning=True)
    fdb_attr = [column.name for column in inspect(self.dbt.find_db_table).c]
    fdb_attr.remove("insert_ts")
    fdb_attr.remove("update_ts")
    return kwargs, fdb_attr

  def get_context_list(self, session, batch_jobs):
    """Return list of jobs (context) for celery queue"""

    kwargs, fdb_attr = self.get_context_items()
    context_list = []
    entries = self.compose_work_objs_fin(session, batch_jobs, self.dbt)
    serialized_jobs = serialize_chunk(entries)
    #build context for each celery task
    for job, config in serialized_jobs:
      context = {
          'job': job,
          'config': config,
          'operation': self.operation,
          'arch': self.dbt.session.arch,
          'num_cu': self.dbt.session.num_cu,
          'kwargs': kwargs,
          'fdb_attr': fdb_attr
      }
      context_list.append(context)

    return context_list

  def celery_enqueue_call(self, context, q_name, task_id=False):
    """Enqueue job (context) for queue:q_name"""

    #hacky way to get the Q_NAME to the task decorator for interpreter to decorate the
    #function with correct q_name arg
    #if import is moved to top it will result in circular imports
    Q_NAME = q_name  #pylint: disable=import-outside-toplevel,unused-variable,invalid-name,redefined-outer-name
    from tuna.miopen.celery_tuning.celery_tasks import celery_enqueue  #pylint: disable=import-outside-toplevel

    return celery_enqueue.apply_async((context,),
                                      task_id=('-').join([self.prefix,
                                                          uuid()]),
                                      queue=q_name,
                                      reply_to=q_name)

  async def parse_result(self, data):
    """Function callback for celery async jobs to store results"""
    data = json.loads(data)
    #self.logger.info(data)

    with DbSession() as session:
      try:
        fin_json = data['result']['ret']
        context = data['result']['context']
      except KeyError as kerr:
        self.logger.error(kerr)
        return False

      #self.logger.info('Parsing: %s', fin_json)
      #self.logger.info('Parsing context: %s', context)
      if self.operation == Operation.COMPILE:
        self.process_fin_builder_results(session, fin_json, context)
      elif self.operation == Operation.EVAL:
        self.process_fin_evaluator_results(session, fin_json, context)
      else:
        raise CustomError('Unsupported tuning operation')

      return True

  def process_fin_builder_results(self, session, fin_json, context):
    """Process result from fin_build worker"""
    job = SimpleDict(**context['job'])
    pending = []
    solver_id_map = get_solver_ids()

    failed_job = False
    result_str = ''
    status = None
    try:
      if fin_json:
        if 'miopen_find_compile_result' in fin_json:
          status = process_fdb_w_kernels(session, fin_json,
                                         copy.deepcopy(context), self.dbt,
                                         context['fdb_attr'], pending)

        elif 'miopen_perf_compile_result' in fin_json:
          status = process_pdb_compile(session, fin_json, job, self.dbt,
                                       solver_id_map)

        success, result_str = get_fin_result(status)
        failed_job = not success

    except (OperationalError, IntegrityError) as err:
      self.logger.warning('FinBuild: Unable to update Database %s', err)
      session.rollback()
      failed_job = True
    except DataError as err:
      self.logger.warning(
          'FinBuild: Invalid data, likely large workspace. DB Error: %s', err)
      session.rollback()
      failed_job = True

    if failed_job:
      set_job_state(session, job, self.dbt, 'errored', False, result=result_str)
    else:
      set_job_state(session,
                    job,
                    self.dbt,
                    'compiled',
                    False,
                    result=result_str)

    return True

  def process_fin_evaluator_results(self, session, fin_json, context):
    """Process fin_json result"""
    job = SimpleDict(**context['job'])
    failed_job = True
    result_str = ''
    pending = []
    orig_state = 'compiled'

    try:
      if fin_json:
        if 'miopen_find_eval_result' in fin_json:
          status = process_fdb_w_kernels(session,
                                         fin_json,
                                         copy.deepcopy(context),
                                         self.dbt,
                                         context['fdb_attr'],
                                         pending,
                                         result_str='miopen_find_eval_result',
                                         check_str='evaluated')
        elif 'miopen_perf_eval_result' in fin_json:
          status = process_fdb_w_kernels(session,
                                         fin_json,
                                         copy.deepcopy(context),
                                         self.dbt,
                                         context['fdb_attr'],
                                         pending,
                                         result_str='miopen_perf_eval_result',
                                         check_str='evaluated')

        success, result_str = get_fin_result(status)
        failed_job = not success

      if failed_job:
        if job.retries >= (MAX_ERRORED_JOB_RETRIES - 1):  #pylint: disable=no-member
          self.logger.warning('max job retries exhausted, setting to errored')
          set_job_state(session, job, self.dbt, 'errored', result=result_str)
        else:
          self.logger.warning('resetting job state to %s, incrementing retries',
                              orig_state)
          set_job_state(session,
                        job,
                        self.dbt,
                        orig_state,
                        increment_retries=True,
                        result=result_str)
      else:
        self.logger.info("\n\n Setting job state to evaluated")
        set_job_state(session, job, self.dbt, 'evaluated', result=result_str)
        clean_cache_table(self.dbt, job)
    except (OperationalError, IntegrityError) as err:
      self.logger.warning('FinBuild: Unable to update Database %s', err)
      session.rollback()
      set_job_state(session, job, self.dbt, 'errored', result=result_str)

    return True

  def reset_job_state_on_ctrl_c(self):
    """Resetting job state for jobs in flight"""
    temp_obj = SimpleDict()
    temp_obj.session_id = self.args.session_id  #pylint: disable=invalid-name
    attribs = ['state']
    temp_obj.state = 1

    self.logger.info('Resetting job state in DB for in flight jobs')

    if self.operation == Operation.COMPILE:
      state = 16
    elif self.operation == Operation.EVAL:
      state = 12

    query = gen_update_query(temp_obj, attribs,
                             self.dbt.job_table.__tablename__,
                             [('session', self.args.session_id),
                              ('state', state)])
    with DbSession() as session:

      #pylint: disable=duplicate-code
      def callback() -> bool:
        session.execute(query)
        session.commit()
        return True

      #pylint: enable=duplicate-code

      assert session_retry(session, callback, lambda x: x(), self.logger)
      self.logger.info('Sucessfully reset job state')
      return True

    return False
