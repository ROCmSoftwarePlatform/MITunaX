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
"""Module to handle Fin work"""

import json
import os
import tempfile
import functools
from typing import List, Dict, Tuple
import paramiko
try:
  import queue
except ImportError:
  import Queue as queue  #type: ignore

from sqlalchemy import func as sqlalchemy_func
from sqlalchemy.exc import IntegrityError, InvalidRequestError  #pylint: disable=wrong-import-order
from sqlalchemy.inspection import inspect

from tuna.worker_interface import WorkerInterface
from tuna.dbBase.sql_alchemy import DbSession
from tuna.miopen.utils.metadata import FIN_CACHE, INVERS_DIR_MAP
from tuna.miopen.worker.fin_utils import compose_config_obj
from tuna.miopen.utils.config_type import ConfigType
from tuna.utils.db_utility import session_retry
from tuna.miopen.db.solver import get_solver_ids, get_id_solvers
from tuna.utils.db_utility import gen_select_objs, get_class_by_tablename, attach_tensor_info
from tuna.utils.utility import split_packets, SimpleDict


class FinClass(WorkerInterface):
  """Class to provide Tuna support for Fin"""

  # pylint: disable=too-many-instance-attributes
  # pylint: disable=too-many-public-methods
  # pylint: disable=no-member

  def __init__(self, **kwargs):
    """Constructor"""
    allowed_keys = set([
        'fin_steps', 'local_file', 'fin_infile', 'fin_outfile', 'config_type',
        'dynamic_solvers_only'
    ])
    self.__dict__.update((key, None) for key in allowed_keys)

    self.supported_fin_steps = ["get_solvers", "applicability"]
    self.fin_steps = []
    _, self.local_file = tempfile.mkstemp()
    self.fin_infile = self.local_file.split("/tmp/", 1)[1] + ".json"
    _, self.local_output = tempfile.mkstemp()
    self.fin_outfile = self.local_output.split("/tmp/", 1)[1] + ".json"

    _, self.id_solver_map = get_id_solvers(
    )  #hyphenated names used by miopen::solver.ToString()
    self.all_configs = []
    self.fin_list = []
    self.multiproc = False
    self.first_pass = True
    self.dynamic_solvers_only = False
    self.solver_id_map = get_solver_ids()

    self.__dict__.update(
        (key, value) for key, value in kwargs.items() if key in allowed_keys)

    self.config_type = ConfigType.convolution if self.config_type is None else ConfigType(
        self.config_type)

    super().__init__(**kwargs)

    #add cache directories
    self.envmt.append(
        f"MIOPEN_USER_DB_PATH=/tmp/miopenpdb/thread-{self.gpu_id}/config/miopen"
    )
    self.envmt.append(
        f"MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopenpdb/thread-{self.gpu_id}/cache")

    self.cfg_attr = [column.name for column in inspect(self.dbt.config_table).c]

    # dict of relationship_column : dict{local_key, foreign_table_name, foreign_key, [remote_attr]}
    self.cfg_rel = {
        key: {
            'key': list(val.local_columns)[0].name,
            'ftble': str(list(val.remote_side)[0]).split('.', maxsplit=1)[0],
            'fkey': str(list(val.remote_side)[0]).split('.')[1]
        } for key, val in inspect(self.dbt.config_table).relationships.items()
    }
    for key, val in self.cfg_rel.items():
      rel_attr = [
          column.name
          for column in inspect(get_class_by_tablename(val['ftble'])).c
      ]
      val['fattr'] = rel_attr

    self.fdb_attr = [
        column.name for column in inspect(self.dbt.find_db_table).c
    ]
    self.fdb_attr.remove("insert_ts")
    self.fdb_attr.remove("update_ts")

  def get_miopen_v(self) -> str:
    """Interface function to get new branch hash"""
    commit_hash: str
    _, commit_hash, _ = self.exec_docker_cmd(
        "cat /opt/rocm/include/miopen/version.h "
        "| grep MIOPEN_VERSION_TWEAK | cut -d ' ' -f 3")
    if "No such file" in commit_hash:
      _, commit_hash, _ = self.exec_docker_cmd(
          "cat /opt/rocm/miopen/include/miopen/version.h "
          "| grep MIOPEN_VERSION_TWEAK | cut -d ' ' -f 3")
    self.logger.info('Got branch commit hash: %s', commit_hash)

    return commit_hash

  def check_env(self) -> bool:
    """Interface function to check the miopen env version vs presumed miopen version"""
    if super().check_env():
      env_miopen_v: str = self.get_miopen_v()
      if self.dbt.session.miopen_v != env_miopen_v:
        raise ValueError(
            f'session miopen_v {self.dbt.session.miopen_v} does not match env miopen_v\
            {env_miopen_v}')
    else:
      return False

    return True

  def chk_abort_file(self):
    """Checking presence of abort file to terminate processes immediately"""
    abort_reason = []
    if os.path.exists(f'/tmp/miopen_abort_{self.machine.arch}'):
      abort_reason.append(self.machine.arch)

    if os.path.exists(f'/tmp/miopen_abort_mid_{self.machine.id}'):
      abort_reason.append('mid_' + str(self.machine.id))
    if abort_reason:
      for reason in abort_reason:
        self.logger.warning('/tmp/mipen_abort_%s file found, returning', reason)
      return True

    return False

  def set_db_tables(self):
    """Initialize tables"""
    # pylint: disable=import-outside-toplevel
    from tuna.miopen.db.tables import MIOpenDBTables
    self.dbt = MIOpenDBTables(session_id=self.session_id,
                              config_type=self.config_type)

  def compose_work_objs(
      self, session: DbSession,
      conds: List[str]) -> List[Tuple[SimpleDict, SimpleDict]]:
    """query for job and config tuple"""
    ret = []
    if self.fin_steps:
      conds.append(f"fin_step like '%{self.fin_steps[0]}%'")
    else:
      conds.append("fin_step='not_fin'")

    job_entries = super().compose_work_objs(session, conds)

    if job_entries:
      id_str = ','.join([str(job[0].config) for job in job_entries])
      cfg_cond_str = f"where valid=1 and id in ({id_str})"
      cfg_entries = gen_select_objs(session, self.cfg_attr,
                                    self.dbt.config_table.__tablename__,
                                    cfg_cond_str)

      #attach tensor relationship information to config entries
      attach_tensor_info(cfg_entries, session, self.cfg_rel)
      #for cfg in cfg_entries:
      #  for key, val in self.cfg_rel.items():
      #    rel_val = getattr(cfg, val['key'])
      #    rel_cond_str = f"where {val['fkey']}={rel_val}"
      #    setattr(
      #        cfg, key,
      #        gen_select_objs(session, val['fattr'], val['ftble'],
      #                        rel_cond_str)[0])

      cfg_map = {cfg.id: cfg for cfg in cfg_entries}

      for job in job_entries:
        ret.append((job[0], cfg_map[job[0].config]))

    return ret

  def __compose_fincmd(self):
    """Helper function to compose fin docker cmd"""
    if self.machine.local_machine:
      # Skip the copy and use the /tmp/* version of the files
      fin_ifile = self.local_file
      fin_ofile = self.local_output
    else:
      fin_ifile = FIN_CACHE + "/" + self.fin_infile
      #Currently not used, but will be used in the future
      #machine_cfg = get_qts_machine_data(self.machine.id, self.machine.hostname,
      #                                   self.logger)
      try:
        self.logger.info("Fin: copying local fin input_file: %s to remote %s",
                         self.local_file, fin_ifile)
        # TODO: remove redundant file copies  # pylint: disable=fixme
        self.cnx.ssh.open_sftp().put(self.local_file, fin_ifile)
        self.logger.info("Fin: Successfully copied to remote")
      except paramiko.ssh_exception.SSHException:
        self.logger.warning('unable to connect to remote %s', fin_ifile)
      except IOError:
        self.logger.warning('unable to receive file: %s skipping ... ',
                            fin_ifile)

      fin_ofile = FIN_CACHE + "/" + self.fin_outfile
    bash_cmd = f"/opt/rocm/bin/fin -i {fin_ifile} -o {fin_ofile}"
    self.logger.info('Executing fin cmd: %s', bash_cmd)
    return bash_cmd

  def get_solvers(self):
    """Getting solvers from MIOpen to update Tuna DB"""
    self.fin_steps = ['get_solvers']
    solvers = self.__get_fin_results()
    if solvers is None:
      return False
    if 'all_solvers' not in solvers[1]:
      self.logger.error('all_solvers key not found in fin output')
    self.__parse_solvers(solvers[1]['all_solvers'])

    return True

  def __get_fin_results(self):
    """Helper function to launch fin docker cmd, cat output of the cmd and parse the json"""
    # pylint: disable=broad-except
    result = None

    if self.__prep_fin_input(self.local_file, to_file=True):
      fin_cmd = self.__compose_fincmd()
      ret_code, out, err = self.exec_docker_cmd(fin_cmd)
      if ret_code > 0:
        self.logger.warning('Err executing cmd: %s', fin_cmd)
        self.logger.warning(out)
        raise ValueError(
            f'Failed to execute fin cmd: {fin_cmd} err: {err.read()}')

      result = self.__parse_out()

    return result

  def __parse_out(self):
    """Parse fin output helper function"""
    # pylint: disable=broad-except
    result = None
    if not self.machine.local_machine:
      fin_outfile = FIN_CACHE + "/" + self.fin_outfile
      # TODO: This should be copied back out using cat is bad # pylint: disable=fixme
      _, ssh_stdout, _ = self.exec_command(f"cat {fin_outfile}")
      result_json = []

      for line in ssh_stdout:
        result_json.append(line)
      try:
        result = json.loads('\n'.join(result_json))
      except Exception as err:
        self.logger.warning('Err loading fin json: %s', err)
        return None
    else:
      with open(self.local_output) as out_file:  # pylint: disable=unspecified-encoding
        try:
          result = json.load(out_file)
        except Exception as err:
          self.logger.error('Unable to load fin json file %s', err)
          for line in out_file:
            self.logger.error(line)
          return None

    return result

  def applicability(self):
    """Getting applicability from MIOpen to update Tuna DB"""
    self.fin_steps = ['applicability']
    applic_res = self.__get_fin_results()
    if applic_res is None:
      return False

    self.__parse_applicability(applic_res)

    return True

  def query_cfgs(self, label=None):
    """query all configs from table, optionally limit by label"""
    with DbSession() as session:
      query = session.query(self.dbt.config_table)\
                        .filter(self.dbt.config_table.valid == 1)

      if label:
        query = query.filter(self.dbt.config_table.id == self.dbt.config_tags_table.config)\
            .filter(self.dbt.config_tags_table.tag == label)

      #order by id for splitting configs into blocks
      query = query.order_by(self.dbt.config_table.id)
      return query

  def __set_all_configs(self, idx: int = 0, num_blk: int = 1) -> bool:
    """Gathering all configs from Tuna DB to set up fin input file"""
    if idx == 0:
      query = self.query_cfgs(self.label)
      rows = query.all()

      len_rows = len(rows)
      master_cfg_list = []
      for row in rows:
        r_dict = compose_config_obj(row, self.config_type)
        if self.config_type == ConfigType.batch_norm:
          r_dict['direction'] = row.get_direction()
        master_cfg_list.append(r_dict)

      block_size = len_rows // num_blk  #size of the config block
      extra = len_rows % num_blk  #leftover configs, don't divide evenly
      self.logger.info("cfg workdiv: num_blocks: %s, block_size: %s, extra: %s",
                       num_blk, block_size, extra)
      for i in range(num_blk):
        start = i * block_size  #start of a process block
        end = (i + 1) * block_size
        #distributing leftover configs to processes
        if i < extra:
          start += i
          end += 1 + i
        else:
          start += extra
          end += extra

        if start >= len_rows:
          self.job_queue.put([])
        else:
          self.logger.info("cfg workdiv: start %s, end %s", start, end)
          self.job_queue.put(master_cfg_list[start:end])
    try:
      self.all_configs = self.job_queue.get(True, 180)
    except queue.Empty:
      self.logger.warning('No jobs found for process %s...', idx)
      self.all_configs = []

    if not self.all_configs:
      return False

    return True

  def __create_dumplist(self):
    """Creating json dump to be used as fin input file"""
    self.fin_list = []
    if len(self.fin_steps) == 1 and self.fin_steps == ["get_solvers"]:
      self.fin_list = [{"steps": self.fin_steps}]
      self.logger.info("Creating dumplist for: %s", self.fin_steps[0])
      return True

    if "applicability" in self.fin_steps:
      self.logger.info("Creating dumplist for: %s", self.fin_steps[0])
      idx = 0
      num_blk = 1
      if self.multiproc:
        idx = self.gpu_id
        num_blk = self.num_procs.value

      if not self.__set_all_configs(idx, num_blk):
        return False
      return self.__compose_fin_list()

    self.logger.error("Fin steps not recognized: %s", self.fin_steps)
    self.logger.info("Fin steps recognized are: %s", self.supported_fin_steps)
    return False

  def __compose_fin_list(self):
    """Helper function to set fin_list for dump"""

    arch = self.dbt.session.arch
    ncu = self.dbt.session.num_cu
    for cfg in self.all_configs:
      self.fin_list.append({
          "steps": self.fin_steps,
          "arch": arch,
          "num_cu": ncu,
          "config_tuna_id": cfg["id"],
          "config": cfg,
          "direction": int(INVERS_DIR_MAP[cfg["direction"]])
      })
    return True

  def __dump_json(self, outfile, to_file=True):
    """Dumping json to outfile"""
    if to_file is True:
      if not os.path.exists(outfile):
        os.mknod(outfile)
      with open(outfile, 'w') as fout:  # pylint: disable=unspecified-encoding
        fout.write("[\n")
        i = 0
        while i < len(self.fin_list):
          json_out = json.dumps(self.fin_list[i])
          fout.write(json_out)
          if i != len(self.fin_list) - 1:
            fout.write(',\n')
          i += 1

        fout.write("\n]")
      self.logger.info('Fin input file written to %s', outfile)
    else:
      jdump = json.dumps(self.fin_list)
      return jdump

    return True

  def __prep_fin_input(self, outfile=None, to_file=True):
    """Main function in Fin that produces Fin input file"""

    self.cnx = self.machine.connect(self.chk_abort_file())
    ret = False
    if outfile is None:
      outfile = "fin_input.json"
    if self.__create_dumplist():
      ret = self.__dump_json(outfile, to_file)
    else:
      self.logger.warning("Could not create dumplist for Fin input file")

    return ret

  def __insert_applicability(self, session: DbSession,
                             json_in: List[Dict]) -> bool:
    """write applicability to sql"""
    inserts = []
    app_values = []
    app_cfgs = []
    for elem in json_in:
      if "applicable_solvers" in elem.keys():
        cfg_id = elem["input"]["config_tuna_id"]

        if not elem["applicable_solvers"]:
          self.logger.warning("No applicable solvers for %s", cfg_id)

        app_cfgs.append(f"{cfg_id}")

        for solver in elem["applicable_solvers"]:
          try:
            solver_id = self.solver_id_map[solver]
            vals = f"({self.session_id}, {cfg_id}, {solver_id}, 1)"
            app_values.append(vals)
          except KeyError:
            self.logger.warning('Solver %s not found in solver table', solver)
            self.logger.info("Please run 'go_fish.py --update_solver' first")

    cleanup = f"delete from {self.dbt.solver_app.__tablename__} where session={self.session_id}"\
               " and config in (" + ", ".join(app_cfgs) + ");"
    ins_str = f"insert ignore into {self.dbt.solver_app.__tablename__}"\
               " (session, config, solver, applicable)"\
               " values " + ", ".join(app_values) + ";"
    inserts.append(cleanup)
    inserts.append(ins_str)

    with self.job_queue_lock:
      self.logger.info('Commit bulk configs (%s), entries (%s), please wait',
                       len(app_cfgs), len(app_values))
      for sql_str in inserts:
        session.execute(sql_str)
      session.commit()
      self.logger.info('End bulk inserts')

    return True

  def __parse_applicability(self, json_in):
    """Function to parse fin outputfile and populate DB with results"""
    self.logger.info('Parsing fin solver applicability output...')
    if json_in is None:
      self.logger.error("JSON file returned from Fin is empty")
      return False

    all_packs = split_packets(json_in, 10000)

    with DbSession() as session:

      def actuator(func, pack):
        return func(session, pack)

      for pack in all_packs:
        session_retry(session, self.__insert_applicability,
                      functools.partial(actuator, pack=pack), self.logger)

      query = session.query(sqlalchemy_func.count(self.dbt.solver_app.id),
                            self.dbt.solver_app.applicable)
      query = query.filter(self.dbt.solver_app.session == self.session_id)  # pylint: disable=W0143
      if self.label:
        query = query.filter(self.dbt.solver_app.config == self.dbt.config_tags_table.config)\
            .filter(self.dbt.config_tags_table.tag == self.label)
      query = query.group_by(self.dbt.solver_app.applicable)
      sapp_count = query.all()
      self.logger.warning(
          "Finished parsing solver applicability, label(%s): %s", self.label,
          sapp_count)
    return True

  def __invalidate_solvers(self, sids, max_id):
    """Helper function to invalidate solver in DB that are not present in Fin outputfile"""
    solver_ids_invalid = []
    with DbSession() as session:
      i = 1
      try:
        while i <= max_id:
          #if solver has been removed
          if i not in sids:
            solver_ids_invalid.append(i)
            session.query(self.dbt.solver_table).filter(
                self.dbt.solver_table.id == i).update(
                    {self.dbt.solver_table.valid: 0})
            session.commit()
          i += 1
      except IntegrityError as err:
        self.logger.warning("DB err occurred %s", err)

    return solver_ids_invalid

  def __add_new_solvers(self, solvers):
    """Add new solvers to db and return the key for latest solver"""

    max_id = 1
    sids = []
    with DbSession() as session:
      for slv_map in solvers:
        idx = int(slv_map['id'])
        solver = slv_map['name']
        tunable = int(slv_map['tunable'])
        config_type = slv_map['type']
        valid = slv_map['valid']
        try:
          sids.append(idx)
          if idx > max_id:
            max_id = idx

          new_s = self.dbt.solver_table(id=idx,
                                        solver=solver,
                                        valid=valid,
                                        tunable=tunable,
                                        config_type=config_type,
                                        is_dynamic=slv_map['dynamic'])
          session.add(new_s)
          session.commit()
        except IntegrityError:
          self.logger.info(
              "Duplicate entry, updating solver %s: valid=1, tunable=%s",
              solver, tunable)
          session.rollback()
          session.query(self.dbt.solver_table).filter(
              self.dbt.solver_table.id == idx).update({
                  self.dbt.solver_table.valid: valid,
                  self.dbt.solver_table.solver: solver,
                  self.dbt.solver_table.tunable: tunable
              })
          session.commit()
        except InvalidRequestError as err2:
          self.logger.info("DB err occurred: %s", err2)

    return max_id, sids

  def __parse_solvers(self, solvers):
    """Function to parse solvers from fin output file"""
    # TODO: Refactor such that we query all the solvers # pylint: disable=fixme
    # from the db once then only insert/invalidate the new/invalid one
    max_id, sids = self.__add_new_solvers(solvers)

    solver_ids_invalid = []
    if len(sids) != max_id:
      solver_ids_invalid = self.__invalidate_solvers(sids, max_id)
      self.logger.info("invalid solvers: %s", solver_ids_invalid)

    s_count = 0
    with DbSession() as session:
      query = session.query(sqlalchemy_func.count(self.dbt.solver_table.id))
      s_count = query.one()[0]

    if max_id != s_count:
      #Note: we canot update invalid solvers missing from DB bc MIOpen does not report these
      self.logger.info(
          "Solver table missing some invalid solvers, please check MIOpens solver.cpp \
          file for solvers that have been invalidated and are missing from your DB"
      )
      self.logger.info("Current invalid solvers: %s", solver_ids_invalid)

    return True

  @staticmethod
  def populate_kernels(kern_obj, kernel_obj):
    """populate kernel object"""
    kernel_obj.kernel_name = kern_obj['kernel_file']
    kernel_obj.kernel_args = kern_obj['comp_options']
    kernel_obj.kernel_blob = bytes(kern_obj['blob'], 'utf-8')
    kernel_obj.kernel_hash = kern_obj['md5_sum']
    kernel_obj.uncompressed_size = kern_obj['uncompressed_size']
    return kernel_obj

  def init_check_env(self):
    """check environment on the first run"""
    if self.first_pass:
      self.first_pass = False
      try:
        self.check_env()
      except ValueError as verr:
        self.logger.error(verr)
        return False
    return True

  def run_fin_cmd(self):
    """Run a fin command after generating the JSON"""
    fin_output = self.machine.make_temp_file()
    cmd = []

    env_str = " ".join(self.envmt)
    cmd.append(env_str)
    cmd.extend(
        ['/opt/rocm/bin/fin', '-i',
         self.get_fin_input(), '-o', fin_output])  # pylint: disable=no-member

    ret_code, _ = super().run_command(cmd)
    if ret_code != 0:
      return None

    # load the output json file and strip the env
    fin_json = json.loads(self.machine.read_file(fin_output))[1:]
    assert len(fin_json) == 1
    fin_json = fin_json[0]
    return fin_json

  def step(self):
    """Inner loop for Process run defined in worker_interface"""
    _, self.id_solver_map = get_id_solvers(
    )  #hyphenated names used by miopen::solver.ToString()
    self.multiproc = True
    if "applicability" in self.fin_steps:
      self.applicability()

    self.multiproc = False
    return False
