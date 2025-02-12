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
""" Module for tagging and importing configs """
import os
import logging
import argparse
from typing import Any, Optional, Union, Tuple, List
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from tuna.dbBase.sql_alchemy import DbSession
from tuna.utils.db_utility import connect_db, ENGINE
from tuna.utils.logger import setup_logger
from tuna.miopen.parse_miopen_args import get_import_cfg_parser
from tuna.miopen.db.tables import ConfigType
from tuna.miopen.driver.convolution import DriverConvolution
from tuna.miopen.driver.base import DriverBase
from tuna.miopen.driver.batchnorm import DriverBatchNorm
from tuna.miopen.db.tables import MIOpenDBTables
from tuna.miopen.db.benchmark import Framework, Model


def create_query(tag: str, mark_recurrent: bool, config_id: int) -> dict:
  """Helper function to build query to add tag"""
  query_dict: dict
  if tag is None and mark_recurrent:
    query_dict = {"config": config_id, "recurrent": 1}
  elif tag is not None and not mark_recurrent:
    query_dict = {"config": config_id, "tag": tag}
  else:
    query_dict = {"config": config_id, "tag": tag, "recurrent": 1}
  return query_dict


def tag_config_v2(driver: DriverBase,
                  counts: dict,
                  dbt: MIOpenDBTables,
                  args: argparse.Namespace,
                  logger: logging.Logger,
                  new_cf: Union[DriverBase, None] = None) -> bool:
  """Adds tag for a config formatted from the fds structure.
        If mark_recurrent is ussed then it also marks it as such.
        Updates counter for tagged configs"""
  c_id = None
  if new_cf is None:
    c_id = driver.get_db_obj(keep_id=True).id
  else:
    c_id = new_cf.id

  with DbSession() as session:
    try:
      query_dict = create_query(args.tag, args.mark_recurrent, c_id)
      session.merge(dbt.config_tags_table(**query_dict))
      session.commit()
      counts['cnt_tagged_configs'].add(c_id)
    except IntegrityError as err:
      #if config/tag already exist we update recurrent=1 if required
      session.rollback()
      if "recurrent" in query_dict.keys():
        query_dict.pop("recurrent")
        #session doesnt support ON DUPLICATE KEY UPDATE, so we have to use the engine to execute
        with ENGINE.connect() as conn:
          conn.execute(dbt.config_tags_table.__table__.update().where(
              (dbt.config_tags_table.config == query_dict["config"]) &
              (dbt.config_tags_table.tag == query_dict["tag"])).values(
                  recurrent="1"))
      if "Duplicate" in str(err):
        logger.warning("Config/tag already present in %s\n",
                       dbt.config_tags_table.__tablename__)
      else:
        logger.error('Err occurred: %s', str(err))

  return True


def insert_config(driver: DriverBase, counts: dict, dbt: MIOpenDBTables,
                  args: argparse.Namespace,
                  logger: logging.Logger) -> Optional[Any]:
  """Inserts new config in the DB computed from the fds structure.
        Tags the newly inserted config. It config already exists,
        it will only tag it and log a warning for duplication."""
  new_cf = driver.get_db_obj(keep_id=True)

  with DbSession() as session:
    if new_cf.id is None:
      try:
        session.add(new_cf)
        session.commit()
        counts['cnt_configs'] += 1
        session.refresh(new_cf)
      except IntegrityError as err:
        logger.warning("Err occurred: %s", err)
        session.rollback()

    if args.mark_recurrent or args.tag:
      _ = tag_config_v2(driver, counts, dbt, args, logger, new_cf)

  return new_cf.id


def process_config_line_v2(driver: DriverBase, args: argparse.Namespace,
                           counts: dict, dbt: MIOpenDBTables,
                           logger: logging.Logger) -> bool:
  """Assumes config passed already exists and will skip the insert step
        if tag_only present. Otherwise it will first try and insert and
        then tag."""
  if args.tag_only:
    _ = tag_config_v2(driver, counts, dbt, args, logger, new_cf=None)
    return False

  _ = insert_config(driver, counts, dbt, args, logger)
  return True


def parse_line(args: argparse.Namespace, line: str, counts: dict,
               dbt: MIOpenDBTables, logger: logging.Logger) -> bool:
  """parse a driver line or fdb line from an input file and insert the config"""
  if args.config_type == ConfigType.batch_norm:
    driver = DriverBatchNorm(line, args.command)
  else:
    driver = DriverConvolution(line, args.command)

  if not args.batch_list:
    process_config_line_v2(driver, args, counts, dbt, logger)
  else:
    for bsz in args.batch_list:
      logger.info('Batchsize: %s', bsz)
      driver.batchsize = bsz
      process_config_line_v2(driver, args, counts, dbt, logger)

  return True


def import_cfgs(args: argparse.Namespace, dbt: MIOpenDBTables,
                logger: logging.Logger) -> dict:
  """import configs to mysql from file with driver invocations"""
  connect_db()

  counts: dict = {}
  counts['cnt_configs'] = 0
  counts['cnt_tagged_configs'] = set()
  unique_lines: List[str] = []
  with open(os.path.expanduser(args.file_name), "r") as infile:  # pylint: disable=unspecified-encoding
    line_cnt = 0
    for line in infile:
      line_cnt += 1
      line = line.strip()
      if not line in unique_lines:
        unique_lines.append(line)
        logger.info("parsed: %u, unique: %u", line_cnt, len(unique_lines))
    for line in unique_lines:
      try:
        parse_line(args, line, counts, dbt, logger)
      except ValueError as err:
        logger.warning(err)

  return counts


def set_import_cfg_batches(args: argparse.Namespace):
  """Setting batches for import_configs subcommands"""
  #import configs
  if args.batches is not None:
    args.batch_list = [int(x) for x in args.batches.split(',')]
  else:
    args.batch_list = []


def print_models(logger: logging.Logger) -> bool:
  """Display models from the db table"""
  with DbSession() as session:
    models = session.query(Model).all()
    for model in models:
      logger.info('model %s version %s ', model.model, model.version)
  return True


def add_model(args: argparse.Namespace, logger: logging.Logger) -> bool:
  """Add new model and version to the db table"""
  with DbSession() as session:
    new_model = Model(model=args.add_model, version=args.md_version)
    try:
      session.add(new_model)
      session.commit()
      logger.info('Added model %s with version %s ', args.add_model,
                  str(args.md_version))
    except IntegrityError as err:
      logger.error(err)
      return False

  return True


def add_frameworks(args: argparse.Namespace, logger: logging.Logger) -> bool:
  """Bring DB table up to speed with enums defined in FrameworkEnum"""
  with DbSession() as session:
    new_framework = Framework(framework=args.add_framework,
                              version=args.fw_version)
    try:
      session.add(new_framework)
      session.commit()
      logger.info('Added framework %s with version %s ', args.add_framework,
                  str(args.fw_version))
    except IntegrityError as err:
      logger.error(err)
      return False

  return True


def get_database_id(framework: Framework, fw_version: int, model: int,
                    md_version: float, dbt: MIOpenDBTables,
                    logger: logging.Logger) -> Tuple[int, int]:
  """Get DB id of item"""

  mid = -1
  fid = -1
  with DbSession() as session:
    try:
      res = session.query(
          dbt.framework.id).filter(dbt.framework.framework == framework)\
                           .filter(dbt.framework.version == fw_version).one()
      fid = res.id
    except NoResultFound as dberr:
      logger.error(dberr)
      logger.error(
          "Framework not present in the DB. Please run 'import_benchmark.py --add_framework' "\
     "to populate the DB table"
      )
    try:
      res = session.query(dbt.model.id).filter(dbt.model.model == model)\
                                       .filter(dbt.model.version == md_version).one()
      mid = res.id
    except NoResultFound as dberr:
      logger.error(
          "Model not present in the DB. Please run 'import_config.py --add_model' to "\
     "populate the DB table"
      )
      logger.error(dberr)
  return mid, fid


def add_benchmark(args: argparse.Namespace, dbt: MIOpenDBTables,
                  logger: logging.Logger) -> bool:
  """Add new benchmark"""
  mid, fid = get_database_id(args.framework, args.fw_version, args.model,
                             args.md_version, dbt, logger)
  if mid is None:
    logger.error('Could not find DB entry for model:%s, version:%s', args.model,
                 args.md_version)
    return False
  if fid is None:
    logger.error('Could not find DB entry for framework:%s, version:%s',
                 args.framework, args.fw_version)
    return False
  commands = []
  if args.driver:
    commands.append(args.driver)
  else:
    with open(os.path.expanduser(args.file_name), "r") as infile:  # pylint: disable=unspecified-encoding
      for line in infile:
        commands.append(line)

  count = 0

  with DbSession() as session:
    for cmd in commands:
      try:
        if args.config_type == ConfigType.convolution:
          driver = DriverConvolution(line=cmd)
        else:
          driver = DriverBatchNorm(line=cmd)
        db_obj = driver.get_db_obj(keep_id=True)
        if db_obj.id is None:
          logger.error('Config not present in the DB: %s', str(driver))
          logger.error('Please use import_configs.py to import configs')

        benchmark = dbt.benchmark()
        benchmark.framework = fid
        benchmark.model = mid
        benchmark.config = db_obj.id
        benchmark.gpu_number = args.gpu_count
        benchmark.driver_cmd = str(driver)
        benchmark.batchsize = driver.batchsize
        session.add(benchmark)
        session.commit()
        count += 1
      except (ValueError, IntegrityError) as verr:
        logger.warning(verr)
        session.rollback()
  logger.info('Benchmarked %s configs', count)
  return True


def check_import_benchmark_args(args: argparse.Namespace) -> None:
  """Checking args for import_benchmark subcommand"""
  if args.add_model and not args.md_version:
    raise ValueError('Version needs to be specified with model')
  if args.add_benchmark and not (args.model and args.framework and
                                 args.gpu_count and args.md_version and
                                 args.fw_version and
                                 (args.driver or args.file_name)):
    raise ValueError(
        """Model, md_version, framework, fw_version, driver(or filename), \n
         and gpus need to all be specified to add a new benchmark""")


# pylint: disable=too-many-return-statements
def run_import_configs(args: argparse.Namespace,
                       logger: logging.Logger) -> bool:
  """Main function"""
  dbt = MIOpenDBTables(session_id=None, config_type=args.config_type)

  if args.print_models or args.add_model or args.add_framework or args.add_benchmark:
    check_import_benchmark_args(args)

  if args.print_models:
    print_models(logger)
    return True
  if args.add_model:
    add_model(args, logger)
    return True
  if args.add_framework:
    add_frameworks(args, logger)
    return True
  if args.add_benchmark:
    add_benchmark(args, dbt, logger)
    return True
  #if not (args.tag and args.framework and args.fw_version and args.model and
  #        args.md_version):
  #  logger.error(
  #      """Tag, framework & version, model & version arguments is required to \
  #              import configurations""")
  #  return False

  #mid, fid = get_database_id(args.framework, args.fw_version, args.model,
  #                           args.md_version, dbt, logger)
  #if mid is None or fid is None:
  #  logger.error(
  #      'Please use --add_model and --add_framework to add new model and framework'
  #  )
  #  return False

  set_import_cfg_batches(args)
  counts = import_cfgs(args, dbt, logger)
  #tagging imported configs with benchmark
  #add_benchmark(args, dbt, logger)

  logger.info('New configs added: %u', counts['cnt_configs'])
  if args.tag or args.tag_only:
    logger.info('Tagged configs: %u', len(counts['cnt_tagged_configs']))

  return True


def main():
  """ main """
  parser = get_import_cfg_parser(with_yaml=False)
  args = parser.parse_args()
  run_import_configs(args, setup_logger('import_configs'))


if __name__ == '__main__':
  main()
