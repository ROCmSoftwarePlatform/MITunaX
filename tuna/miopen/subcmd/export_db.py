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
"""Module to export find_db to txt file"""
import sqlite3
import os
import tempfile
import json
from collections import OrderedDict
from typing import Dict, Any, Optional, Union, List, Tuple
import base64
import argparse
import logging

from tuna.dbBase.sql_alchemy import DbSession
from tuna.miopen.db.find_db import FindDBMixin
from tuna.miopen.db.mixin_tables import GoldenMixin
from tuna.miopen.db.tables import MIOpenDBTables
from tuna.miopen.utils.metadata import SQLITE_PERF_DB_COLS
from tuna.utils.db_utility import DB_Type
from tuna.miopen.db.solver import get_id_solvers
from tuna.utils.logger import setup_logger
from tuna.miopen.utils.analyze_parse_db import get_config_sqlite, insert_solver_sqlite
from tuna.miopen.utils.analyze_parse_db import get_sqlite_cfg_dict
from tuna.miopen.utils.metadata import INVERS_DIR_MAP
from tuna.miopen.parse_miopen_args import get_export_db_parser
from tuna.miopen.worker.fin_utils import compose_config_obj

DIR_NAME: dict = {'F': 'Fwd', 'B': 'BwdData', 'W': 'BwdWeights'}

ID_SOLVER_MAP: dict = {}


def require_id_solvers():
  """Initialise ID_SOLVER_MAP from database"""
  global ID_SOLVER_MAP  # pylint: disable=global-statement
  if not ID_SOLVER_MAP:
    _, ID_SOLVER_MAP = get_id_solvers()


def arg_export_db(args: argparse.Namespace, logger: logging.Logger):
  """export db args for exportdb"""
  if args.golden_v and not args.arch:
    logger.error('arch must be set with golden_v')


def get_filename(arch: str,
                 num_cu: Optional[int] = None,
                 filename: Optional[str] = None,
                 ocl: bool = False,
                 db_type: DB_Type = DB_Type.FIND_DB) -> str:
  """Helper function to compose filename"""
  version = "1.0.0"
  tuna_dir = f'tuna_{version}'
  if not os.path.exists(tuna_dir):
    os.makedirs(tuna_dir)

  if filename:
    final_name = f'{tuna_dir}/{filename}'
  elif not num_cu:
    final_name = f'{tuna_dir}/{arch}'
  elif num_cu > 64:
    final_name = f'{tuna_dir}/{arch}{num_cu:x}'
  else:
    final_name = f"{tuna_dir}/{arch}_{num_cu}"

  if db_type == DB_Type.FIND_DB:
    # pylint: disable-next=consider-using-f-string ; more readable
    extension = '.{}.fdb.txt'.format('OpenCL' if ocl else 'HIP')
  elif db_type == DB_Type.KERN_DB:
    extension = '.kdb'
  else:
    extension = ".db.txt"

  final_name = f"{final_name}{extension}"

  return final_name


def fin_nc_job(cfg_lst):
  """Construct a fin network_config job from a config
  """
  #arch and num_cu are required by fin, but unused for this command
  job_list = []
  for config in cfg_lst:
    job = {
        "steps": ["network_config"],
        "arch": 'gfx908',
        "num_cu": 120,
        "config_tuna_id": config.id,
        "direction": int(INVERS_DIR_MAP[config.direction]),
        "config": compose_config_obj(config)
    }
    job_list.append(job)

  return job_list


def fin_network_config(config_lst, logger):
  """rerieve network_config from fin for config """
  _, fin_ifile = tempfile.mkstemp(suffix='.json')
  _, fin_ofile = tempfile.mkstemp(suffix='.json')

  with open(fin_ifile, 'w') as in_file:
    in_file.write(json.dumps(fin_nc_job(config_lst), indent=2))

  fin_cmd = f"/opt/rocm/bin/fin -i {fin_ifile} -o {fin_ofile}"
  logger.info('Executing fin cmd: %s', fin_cmd)

  os.system(fin_cmd)

  result = None
  with open(fin_ofile, 'r') as out_file:  # pylint: disable=unspecified-encoding
    try:
      result = json.load(out_file)
    except Exception as err:
      logger.error('Unable to load fin json file %s', err)
      for line in out_file:
        logger.error(line)

  network_config_dict = {}
  for elem in result:
    if "network_config" in elem.keys():
      network_config_dict[elem['config_tuna_id']] = elem['network_config']

  return network_config_dict


def get_base_query(dbt: MIOpenDBTables, args: argparse.Namespace,
                   logger: logging.Logger):
  """ general query for fdb/pdb results """
  with DbSession() as session:
    query = session.query(args.src_table, dbt.config_table)
    if args.golden_v is not None:
      query = query.filter(args.src_table.golden_miopen_v == args.golden_v)\
              .filter(args.src_table.arch == args.arch)
      if args.num_cu:
        query = query.filter(args.src_table.num_cu == args.num_cu)
      logger.info("golden_miopen_v: %s, arch: %s, num_cu: %s", args.golden_v,
                  args.arch, args.num_cu)
    else:
      query = query.filter(args.src_table.session == dbt.session.id)
      logger.info("rocm_v : %s", dbt.session.rocm_v)
      logger.info("miopen_v : %s", dbt.session.miopen_v)

    query = query.filter(args.src_table.valid == 1)\
        .filter(args.src_table.opencl == args.opencl)\
        .filter(args.src_table.config == dbt.config_table.id)

    if args.config_tag:
      logger.info("config_tag : %s", args.config_tag)
      query = query.filter(dbt.config_tags_table.tag == args.config_tag)\
          .filter(dbt.config_table.config == dbt.config_table.id)

  return query


def get_fdb_query(dbt: MIOpenDBTables, args: argparse.Namespace,
                  logger: logging.Logger):
  """ Helper function to create find db query
  """
  query = get_base_query(dbt, args, logger)
  query = query.filter(args.src_table.kernel_time != -1)\
      .filter(args.src_table.workspace_sz != -1)

  if args.src_table == dbt.golden_table:
    query = query.order_by(args.src_table.fdb_key,
                           args.src_table.update_ts.desc(),
                           args.src_table.num_cu.asc())
  else:
    query = query.order_by(args.src_table.fdb_key,
                           args.src_table.update_ts.desc())

  return query


def get_pdb_query(dbt: MIOpenDBTables, args: argparse.Namespace,
                  logger: logging.Logger):
  """Compose query to get perf_db rows based on filters from args"""
  query = get_fdb_query(dbt, args, logger)
  query = query.filter(args.src_table.params != '')\
      .filter(args.src_table.solver == dbt.solver_table.id)\
      .filter(dbt.solver_table.tunable == 1)

  return query


def add_entry_to_solvers(fdb_entry: Union[GoldenMixin, FindDBMixin],
                         solvers: Dict[str, Dict[str, Any]],
                         logger: logging.Logger) -> bool:
  """check if fdb_key + solver exists in solvers, add if not present
  return False if similar entry already exists
  return True if the fdb_entry is added successfully
  """
  fdb_key = fdb_entry.fdb_key
  if fdb_key not in solvers:
    solvers[fdb_key] = {}
  elif fdb_entry.solver in solvers[fdb_key].keys():
    _ = logger
    #logger.warning("Skipped duplicate solver: %s : %s with ts %s vs prev ts %s",
    #                fdb_key, fdb_entry.solver, fdb_entry.update_ts,
    #                solvers[fdb_key][fdb_entry.solver])
    return False

  solvers[fdb_key][fdb_entry.solver] = fdb_entry.update_ts
  return True


def build_miopen_fdb(query, logger: logging.Logger) -> OrderedDict:
  """return dict with key: fdb_key, val: list of fdb entries"""
  find_db: OrderedDict = OrderedDict()
  solvers: Dict[str, Dict[str, Any]] = {}
  db_entries = query.all()
  total_entries = len(db_entries)
  logger.info("fdb query returned: %s", total_entries)

  for fdb_entry, _ in db_entries:
    if add_entry_to_solvers(fdb_entry, solvers, logger):
      fdb_key = fdb_entry.fdb_key
      lst = find_db.get(fdb_key)
      if not lst:
        find_db[fdb_key] = [fdb_entry]
      else:
        lst.append(fdb_entry)

  require_id_solvers()
  for _, entries in find_db.items():
    entries.sort(key=lambda x: (float(x.kernel_time), ID_SOLVER_MAP[x.solver]))
    while len(entries) > 4:
      entries.pop()

  return find_db


def write_fdb(arch, num_cu, ocl, find_db, filename=None):
  """
  Serialize find_db map to plain text file in MIOpen format
  """
  file_name = get_filename(arch, num_cu, filename, ocl, DB_Type.FIND_DB)

  require_id_solvers()
  with open(file_name, 'w') as out:  # pylint: disable=unspecified-encoding
    for key, solvers in sorted(find_db.items(), key=lambda kv: kv[0]):
      solvers.sort(
          key=lambda x: (float(x.kernel_time), ID_SOLVER_MAP[x.solver]))
      lst = []
      # for alg_lib, solver_id, kernel_time, workspace_sz in solvers:
      for rec in solvers:
        # pylint: disable-next=consider-using-f-string ; more reable
        lst.append('{slv}:{},{},{alg}'.format(rec.kernel_time,
                                              rec.workspace_sz,
                                              slv=ID_SOLVER_MAP[rec.solver],
                                              alg=rec.alg_lib))
      out.write(f"{key}={';'.join(lst)}\n")
  return file_name


def export_fdb(dbt: MIOpenDBTables, args: argparse.Namespace,
               logger: logging.Logger):
  """Function to export find_db to txt file
  """
  query = get_fdb_query(dbt, args, logger)
  miopen_fdb = build_miopen_fdb(query, logger)

  logger.info("write fdb to file.")
  return write_fdb(args.arch, args.num_cu, args.opencl, miopen_fdb,
                   args.filename)


def build_miopen_kdb(dbt: MIOpenDBTables, find_db, logger: logging.Logger):
  """ create miopen kernel db object for export
  """
  num_fdb_entries = 0
  num_kdb_blobs = 0
  kern_db = []
  with DbSession() as session:
    total = len(find_db.items())
    last_pcnt = 0
    for _, entries in find_db.items():
      num_fdb_entries += 1
      entries.sort(key=lambda x: float(x.kernel_time))
      fastest_slv = entries[0]
      query = session.query(dbt.kernel_cache)\
          .filter(dbt.kernel_cache.kernel_group == fastest_slv.kernel_group)\
          .filter(dbt.kernel_cache.valid == 1)
      for kinder in query.all():
        num_kdb_blobs += 1
        kern_db.append(kinder)
      pcnt = int(num_fdb_entries * 100 / total)
      if pcnt > last_pcnt:
        logger.warning("Building db: %s%%, blobs: %s", pcnt, num_kdb_blobs)
        last_pcnt = pcnt

  logger.warning("Total FDB entries: %s, Total blobs: %s", num_fdb_entries,
                 num_kdb_blobs)
  return kern_db


def write_kdb(arch, num_cu, kern_db, logger: logging.Logger, filename=None):
  """
  Write blob map to sqlite
  """
  file_name = get_filename(arch, num_cu, filename, False, DB_Type.KERN_DB)
  if os.path.isfile(file_name):
    os.remove(file_name)

  conn = sqlite3.connect(file_name)
  cur = conn.cursor()
  cur.execute(
      "CREATE TABLE `kern_db` (`id` INTEGER PRIMARY KEY ASC,`kernel_name` TEXT NOT NULL,"
      "`kernel_args` TEXT NOT NULL,`kernel_blob` BLOB NOT NULL,`kernel_hash` TEXT NOT NULL,"
      "`uncompressed_size` INT NOT NULL);")
  cur.execute(
      "CREATE UNIQUE INDEX `idx_kern_db` ON kern_db(kernel_name, kernel_args);")

  ins_list: List[Tuple[str, str]] = []
  for kern in kern_db:
    name = kern.kernel_name
    args = kern.kernel_args
    #check if extensions should be added
    if not name.endswith('.o'):
      name += ".o"
    if not "-mcpu=" in args:
      if not name.endswith('.mlir.o'):
        args += f" -mcpu={arch}"

    ins_key = (name, args)
    if ins_key not in ins_list:
      ins_list.append(ins_key)
      cur.execute(
          "INSERT INTO kern_db (kernel_name, kernel_args, kernel_blob, kernel_hash, "
          "uncompressed_size) VALUES(?, ?, ?, ?, ?);",
          (name, args, base64.b64decode(
              kern.kernel_blob), kern.kernel_hash, kern.uncompressed_size))

  conn.commit()
  cur.close()
  conn.close()

  logger.warning("Inserted blobs: %s", len(ins_list))
  return file_name


def build_miopen_fdb_skews(args: argparse.Namespace, query,
                           logger: logging.Logger) -> OrderedDict:
  """return dict with key: fdb_key + num_cu, val: list of fdb entries"""
  miopen_fdb: OrderedDict = OrderedDict()
  with DbSession() as session:
    db_entries = query.all()
    fdb_ids = []
    for fdb_entry, _ in db_entries:
      fdb_ids.append(fdb_entry.id)

    skews = [int(x[0]) for x in session.query(args.src_table.num_cu).distinct()\
              .filter(args.src_table.id.in_(fdb_ids)).all()]
    logger.info("skews %s", skews)

  for num_cu in skews:
    cu_query = query.filter(args.src_table.num_cu == num_cu)
    miopen_fdb_skew = build_miopen_fdb(cu_query, logger)
    for key, value in miopen_fdb_skew.items():
      miopen_fdb[f"{key}_cu{num_cu}"] = value

  return miopen_fdb


def export_kdb(dbt: MIOpenDBTables,
               args: argparse.Namespace,
               logger: logging.Logger,
               skew_fdbs=True):
  """
  Function to export the kernel cache
  """
  query = get_fdb_query(dbt, args, logger)

  miopen_fdb: OrderedDict = OrderedDict()
  if skew_fdbs and not args.num_cu:
    miopen_fdb = build_miopen_fdb_skews(args, query, logger)
  else:
    miopen_fdb = build_miopen_fdb(query, logger)

  logger.info("Building kdb.")
  kern_db = build_miopen_kdb(dbt, miopen_fdb, logger)

  logger.info("write kdb to file.")
  return write_kdb(args.arch, args.num_cu, kern_db, logger, args.filename)


def create_sqlite_tables(arch, num_cu, filename=None):
  """create sqlite3 tables"""
  local_path = get_filename(arch, num_cu, filename, False, DB_Type.PERF_DB)

  cnx = sqlite3.connect(local_path)

  cur = cnx.cursor()
  cur.execute(
      "CREATE TABLE IF NOT EXISTS `config` (`id` INTEGER PRIMARY KEY ASC,`layout` TEXT NOT NULL,"
      "`data_type` TEXT NOT NULL,`direction` TEXT NOT NULL,`spatial_dim` INT NOT NULL,"
      "`in_channels` INT NOT NULL,`in_h` INT NOT NULL,`in_w` INT NOT NULL,`in_d` INT NOT NULL,"
      "`fil_h` INT NOT NULL,`fil_w` INT NOT NULL,`fil_d` INT NOT NULL,"
      "`out_channels` INT NOT NULL, `batchsize` INT NOT NULL,"
      "`pad_h` INT NOT NULL,`pad_w` INT NOT NULL,`pad_d` INT NOT NULL,"
      "`conv_stride_h` INT NOT NULL,`conv_stride_w` INT NOT NULL,`conv_stride_d` INT NOT NULL,"
      "`dilation_h` INT NOT NULL,`dilation_w` INT NOT NULL,`dilation_d` INT NOT NULL,"
      "`bias` INT NOT NULL,`group_count` INT NOT NULL)")
  cur.execute(
      "CREATE TABLE IF NOT EXISTS `perf_db` (`id` INTEGER PRIMARY KEY ASC,`solver` TEXT NOT NULL,"
      "`config` INTEGER NOT NULL, `params` TEXT NOT NULL)")

  cur.execute(
      "CREATE UNIQUE INDEX IF NOT EXISTS `idx_config` ON config( layout,data_type,direction,"
      "spatial_dim,in_channels,in_h,in_w,in_d,fil_h,fil_w,fil_d,out_channels,"
      "batchsize,pad_h,pad_w,pad_d,conv_stride_h,conv_stride_w,conv_stride_d,"
      "dilation_h,dilation_w,dilation_d,bias,group_count )")
  cur.execute(
      "CREATE UNIQUE INDEX IF NOT EXISTS `idx_perf_db` ON perf_db(solver, config)"
  )

  cur.close()
  cnx.commit()
  return cnx, local_path


def insert_perf_db_sqlite(cnx, perf_db_entry, ins_cfg_id):
  """insert perf_db entry into sqlite"""
  perf_db_dict = perf_db_entry.to_dict()
  perf_db_dict['config'] = ins_cfg_id
  perf_db_dict = {
      k: v for k, v in perf_db_dict.items() if k in SQLITE_PERF_DB_COLS
  }
  require_id_solvers()
  perf_db_dict['solver'] = ID_SOLVER_MAP[perf_db_dict['solver']]

  insert_solver_sqlite(cnx, perf_db_dict)

  return perf_db_dict


def populate_sqlite(cfg_map, num_perf, cnx, perf_db_entry, cfg_entry,
                    total_entries, logger: logging.Logger):
  """Analyze perf_dv entry"""
  if cfg_entry.id in cfg_map:
    ins_cfg_id = cfg_map[cfg_entry.id]
  else:
    cfg_dict = get_sqlite_cfg_dict(perf_db_entry.fdb_key)

    #filters cfg_dict by SQLITE_CONFIG_COLS, inserts cfg if missing
    ins_cfg_id = get_config_sqlite(cnx, cfg_dict)
    cfg_map[cfg_entry.id] = ins_cfg_id

  pdb_dict = insert_perf_db_sqlite(cnx, perf_db_entry, ins_cfg_id)
  num_perf += 1

  if num_perf % (total_entries // 10) == 0:
    cnx.commit()
    logger.info("PDB count: %s, mysql cfg: %s, pdb: %s", num_perf, cfg_entry.id,
                pdb_dict)


def export_pdb(dbt: MIOpenDBTables, args: argparse.Namespace,
               logger: logging.Logger):
  """ export perf db from mysql to sqlite """
  cnx, local_path = create_sqlite_tables(args.arch, args.num_cu, args.filename)
  num_perf = 0
  cfg_map: Dict[Any, Any] = {}
  solvers: Dict[str, Dict[str, Any]] = {}
  db_entries = get_pdb_query(dbt, args, logger).all()
  total_entries = len(db_entries)
  logger.info("pdb query returned: %s", total_entries)

  for perf_db_entry, cfg_entry in db_entries:
    if add_entry_to_solvers(perf_db_entry, solvers, logger):
      populate_sqlite(cfg_map, num_perf, cnx, perf_db_entry, cfg_entry,
                      total_entries, logger)

  cnx.commit()
  logger.warning("Total number of entries in Perf DB: %s", num_perf)

  return local_path


def build_miopen_pdb(query, logger: logging.Logger) -> OrderedDict:
  """return dict with key: fdb_key, val: list of fdb entries"""
  perf_db: OrderedDict = OrderedDict()
  solvers: Dict[str, Dict[str, Any]] = {}
  nc_key_map: Dict[str, str] = {}
  db_entries = query.all()
  total_entries = len(db_entries)
  logger.info("pdb query returned: %s", total_entries)

  cfg_lst = []
  for _, config in db_entries:
    if config not in cfg_lst:
      cfg_lst.append(config)

  ntwk_cfg_map = fin_network_config(cfg_lst, logger)

  for pdb_entry, config in db_entries:
    if add_entry_to_solvers(pdb_entry, solvers, logger):
      nc_key = ntwk_cfg_map[config.id]
      lst = perf_db.get(nc_key)
      if not lst:
        perf_db[nc_key] = [pdb_entry]
      else:
        lst.append(pdb_entry)

  return perf_db


def write_pdb(arch, num_cu, ocl, perf_db, filename=None):
  """
  Serialize perf_db map to plain text file in MIOpen format
  """
  file_name = get_filename(arch, num_cu, filename, ocl, DB_Type.PERF_DB)

  require_id_solvers()
  with open(file_name, 'w') as out:  # pylint: disable=unspecified-encoding
    for key, solvers in sorted(perf_db.items(), key=lambda kv: kv[0]):
      solvers.sort(
          #key=lambda x: (float(x.kernel_time), ID_SOLVER_MAP[x.solver]))
          key=lambda x: (ID_SOLVER_MAP[x.solver]))
      lst = []
      # for alg_lib, solver_id, kernel_time, workspace_sz in solvers:
      for rec in solvers:
        # pylint: disable-next=consider-using-f-string ; more reable
        lst.append('{slv}:{params}'.format(slv=ID_SOLVER_MAP[rec.solver],
                                           params=rec.params))
      out.write(f"{key}={';'.join(lst)}\n")
  return file_name


def export_pdb_txt(dbt: MIOpenDBTables, args: argparse.Namespace,
               logger: logging.Logger):
  """ export perf db from mysql to txt file """
  query = get_pdb_query(dbt, args, logger)
  miopen_pdb = build_miopen_pdb(query, logger)

  logger.info("write pdb to file.")
  return write_pdb(args.arch, args.num_cu, args.opencl, miopen_pdb,
                   args.filename)


def run_export_db(args: argparse.Namespace, logger: logging.Logger):
  """run export db script"""
  result_file = ''
  dbt = MIOpenDBTables(session_id=args.session_id)

  if args.session_id:
    try:
      args.arch = dbt.session.arch
      args.num_cu = dbt.session.num_cu
    except ValueError as terr:
      logger.error(str(terr))

  args.src_table = dbt.find_db_table
  if args.golden_v is not None:
    args.src_table = dbt.golden_table

  if args.find_db:
    result_file = export_fdb(dbt, args, logger)
  elif args.kern_db:
    result_file = export_kdb(dbt, args, logger)
  elif args.perf_db:
    result_file = export_pdb_txt(dbt, args, logger)

  print(result_file)


def main():
  """Main module function"""
  parser = get_export_db_parser()
  args = parser.parse_args()
  run_export_db(args, setup_logger('export_db'))


if __name__ == '__main__':
  main()
