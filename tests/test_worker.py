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

import os
import sys
#import pytest
from multiprocessing import Value, Lock, Queue
from subprocess import Popen, PIPE

sys.path.append("../tuna")
sys.path.append("tuna")

this_path = os.path.dirname(__file__)

from tuna.utils.machine_utility import load_machines
from tuna.miopen.worker.fin_class import FinClass
from tuna.machine import Machine
from tuna.sql import DbCursor
from tuna.miopen.utils.config_type import ConfigType
from utils import get_worker_args, add_test_session
from utils import CfgImportArgs, LdJobArgs, GoFishArgs
from tuna.miopen.db.tables import MIOpenDBTables
from tuna.utils.db_utility import connect_db
from tuna.utils.logger import setup_logger
from tuna.miopen.subcmd.import_configs import import_cfgs
from tuna.miopen.subcmd.load_job import test_tag_name as tag_name_test, add_jobs
from miopen.miopen_lib import MIOpen


def add_job(w):
  #import configs
  args = CfgImportArgs()
  args.tag = 'tuna_pytest_worker'
  args.mark_recurrent = True
  args.file_name = f"{this_path}/../utils/configs/conv_configs.txt"
  logger = setup_logger('test_add_job')

  dbt = MIOpenDBTables(session_id=w.session_id, config_type=args.config_type)
  _ = import_cfgs(args, dbt, setup_logger('test_worker'))

  args = GoFishArgs()
  machine_lst = load_machines(args)
  machine = machine_lst[0]

  #update solvers
  miopen = MIOpen()
  miopen.args = args
  kwargs = get_worker_args(args, machine, miopen)
  fin_worker = FinClass(**kwargs)
  assert (fin_worker.get_solvers())

  #get applicability
  args.update_applicability = True
  args.label = 'tuna_pytest_worker'
  args.session_id = w.session_id
  worker_lst = miopen.compose_worker_list(machine_lst)
  for worker in worker_lst:
    worker.join()

  #load jobs
  args = LdJobArgs
  args.label = 'tuna_pytest_worker'
  args.tag = 'tuna_pytest_worker'
  args.fin_steps = ['not_fin']
  args.session_id = w.session_id

  connect_db()
  if args.tag:
    try:
      tag_name_test(args.tag, dbt)
    except ValueError as terr:
      print(terr)

  num_jobs = add_jobs(args, dbt, logger)
  assert num_jobs > 0


def get_job(w):
  with DbCursor() as cur:
    cur.execute("UPDATE conv_job SET valid=0 WHERE reason='tuna_pytest_worker'")
  # to force commit
  with DbCursor() as cur:
    cur.execute(
        f"SELECT id FROM conv_job WHERE reason='tuna_pytest_worker' and session={w.session_id} LIMIT 1"
    )
    res = cur.fetchall()
    assert (len(res) == 1)
    job_id = res[0][0]
    assert (job_id)
    cur.execute(
        f"UPDATE conv_job SET state='new', valid=1, retries=0 WHERE id={job_id}"
    )

  #test get_job()
  job = w.get_job('new', 'compile_start', True)
  assert job == True
  with DbCursor() as cur:
    cur.execute(f"SELECT state FROM conv_job WHERE id={job_id}")
    res = cur.fetchall()
    assert (res[0][0] == 'compile_start')
    job = w.get_job('new', 'compile_start', True)
    assert job == False
    cur.execute(f"UPDATE conv_job SET valid=0 WHERE id={job_id}")


def multi_queue_test(w):
  # a separate block was necessary to force commit
  with DbCursor() as cur:
    cur.execute(
        "UPDATE conv_job SET state='new', valid=1 WHERE reason='tuna_pytest_worker' and session={} LIMIT {}"
        .format(w.session_id, w.claim_num + 1))
  job = w.get_job('new', 'compile_start', True)
  assert job == True
  res = None
  with DbCursor() as cur:
    cur.execute(
        f"SELECT state FROM conv_job WHERE reason='tuna_pytest_worker' and session={w.session_id} AND state='compile_start' AND valid=1"
    )
    res = cur.fetchall()
  assert (len(res) == w.claim_num)

  with DbCursor() as cur:
    cur.execute(
        f"UPDATE conv_job SET state='compiling' WHERE reason='tuna_pytest_worker' and session={w.session_id} AND state='compile_start' AND valid=1"
    )
  for i in range(w.claim_num - 1):
    job = w.get_job('new', 'compile_start', True)
    with DbCursor() as cur:
      assert job == True
      cur.execute(
          f"SELECT state FROM conv_job WHERE reason='tuna_pytest_worker' and session={w.session_id} AND state='compile_start' AND valid=1"
      )
      res = cur.fetchall()
      assert (len(res) == 0)

  job = w.get_job('new', 'compile_start', True)
  assert job == True
  with DbCursor() as cur:
    cur.execute(
        f"SELECT state FROM conv_job WHERE reason='tuna_pytest_worker' and session={w.session_id} AND state='compile_start' AND valid=1"
    )
    res = cur.fetchall()
    assert (len(res) == 1)
    cur.execute(
        f"UPDATE conv_job SET valid=0 WHERE reason='tuna_pytest_worker' and session={w.session_id}"
    )


def test_worker():

  cmd = 'hostname'
  subp = Popen(cmd, stdout=PIPE, shell=True, universal_newlines=True)
  hostname = subp.stdout.readline().strip()
  machine = Machine(hostname=hostname, local_machine=True)

  keys = {}
  num_gpus = Value('i', 2)
  v = Value('i', 0)
  e = Value('i', 0)

  session_id = add_test_session()

  keys = {
      'machine': machine,
      'gpu_id': 0,
      'num_procs': num_gpus,
      'bar_lock': Lock(),
      'envmt': ["MIOPEN_LOG_LEVEL=7"],
      'reset_interval': False,
      'app_test': False,
      'label': 'tuna_pytest_worker',
      'use_tuner': False,
      'job_queue': Queue(),
      'queue_lock': Lock(),
      'end_jobs': e,
      'fin_steps': ['not_fin'],
      'config_type': ConfigType.convolution,
      'session_id': session_id,
      'find_mode': 1,
      'blacklist': None
  }

  #worker is an interface and has no sql tables, using fin
  w = FinClass(**keys)

  add_job(w)
  get_job(w)
  w.queue_end_reset()
  multi_queue_test(w)
  w.check_env()
