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
"""Utility module for DB helper functions"""

import os
import enum
import random
import logging
from time import sleep
from datetime import datetime
from typing import Callable, Any, List, Dict
import pymysql
from sqlalchemy.exc import OperationalError, IntegrityError, ProgrammingError
from sqlalchemy import create_engine

from tuna.dbBase.sql_alchemy import DbSession
from tuna.dbBase.base_class import BASE
from tuna.utils.metadata import NUM_SQL_RETRIES
from tuna.utils.logger import setup_logger
from tuna.utils.utility import get_env_vars
from tuna.utils.utility import SimpleDict

LOGGER = setup_logger('db_utility')

ENV_VARS = get_env_vars()

ENGINE = create_engine(f"mysql+pymysql://{ENV_VARS['user_name']}:{ENV_VARS['user_password']}" +\
                         f"@{ENV_VARS['db_hostname']}:3306/{ENV_VARS['db_name']}",
                       encoding="utf8")


def connect_db():
  """Create DB if it doesnt exist"""
  db_name = None
  if 'TUNA_DB_NAME' in os.environ:
    db_name = os.environ['TUNA_DB_NAME']
  else:
    raise ValueError('DB name must be specified in env variable: TUNA_DB_NAME')

  try:
    ENGINE.execute(f'Use {db_name}')
    return
  except OperationalError:  # as err:
    LOGGER.warning('Database %s does not exist, attempting to create database',
                   db_name)

  try:
    ENGINE.execute(f'Create database if not exists {db_name}')
  except OperationalError as err:
    LOGGER.error('Database creation failed %s for username: %s', err,
                 ENV_VARS['user_name'])
  ENGINE.execute(f'Use {db_name}')
  ENGINE.execute('SET GLOBAL max_allowed_packet=4294967296')


def create_tables(all_tables):
  """Function to create or sync DB tables/triggers"""
  #pylint: disable=too-many-locals
  connect_db()
  for table in all_tables:
    try:
      table.__table__.create(ENGINE)
      LOGGER.info("Created: %s", table.__tablename__)

    except (OperationalError, ProgrammingError) as err:
      LOGGER.warning('Err occurred %s \n For table: %s.', err, table)
      LOGGER.warning(
          'Schema migration not implemented, please update schema manually')
      continue

  return True


def create_indices(all_indices):
  """Create indices from index list"""
  with ENGINE.connect() as conn:
    for idx in all_indices:
      try:
        conn.execute(idx)
        LOGGER.info('Idx created successfully: %s', idx)
      except (OperationalError, ProgrammingError) as oerr:
        LOGGER.info('%s \n', oerr)
        continue


def session_retry(session: DbSession,
                  callback: Callable,
                  actuator: Callable,
                  logger: logging.Logger = LOGGER) -> Any:
  """retry handling for a callback function using an actuator (lamda function with params)"""
  for idx in range(NUM_SQL_RETRIES):
    try:
      return actuator(callback)
    except OperationalError as error:
      logger.warning('%s, DB contention sleeping (%s)...', error, idx)
      session.rollback()
      sleep(random.randint(1, 30))
    except pymysql.err.OperationalError as error:
      logger.warning('%s, DB contention sleeping (%s)...', error, idx)
      session.rollback()
      sleep(random.randint(1, 30))
    except IntegrityError as error:
      logger.error('Query failed: %s', error)
      session.rollback()
      return False

  logger.error('All retries have failed.')
  return False


def get_attr_vals(obj, attr_list):
  """create the dictionary of values for the attribute list """
  attr_vals = {}
  for attr in attr_list:
    val = getattr(obj, attr)
    if val is None:
      val = 'NULL'
    elif isinstance(val, (datetime, str)):
      val = f"'{val}'"
    elif isinstance(val, bytes):
      val = val.decode('utf-8')
      val = f"'{val}'"
    else:
      val = str(val)
    attr_vals[attr] = val
  return attr_vals


def gen_update_query(obj, attribs, tablename, where_clause_tuples_lst=None):
  """Create an update query string to table with tablename for an object (obj)
  for the attributes in attribs"""
  set_arr = []
  attr_vals = get_attr_vals(obj, attribs)
  for attr in attribs:
    set_arr.append(f"{attr}={attr_vals[attr]}")

  set_str = ','.join(set_arr)
  if where_clause_tuples_lst:
    where_clause = ' AND '.join(f"{x}={y}" for x, y in where_clause_tuples_lst)
    query = f"UPDATE {tablename} SET {set_str}"\
            f" WHERE {where_clause};"
  else:
    query = f"UPDATE {tablename} SET {set_str}"\
            f" WHERE id={obj.id};"
  LOGGER.info('Query Update: %s', query)
  return query


def gen_insert_query(obj, attribs, tablename):
  """create a select query and generate name space objects for the results"""
  attr_list = list(attribs)
  attr_list.remove('id')
  attr_str = ','.join(attr_list)

  attr_vals = get_attr_vals(obj, attr_list)
  val_list = [attr_vals[a] for a in attr_list]
  val_str = ','.join(val_list)

  query = f"INSERT INTO {tablename}({attr_str})"\
          f" SELECT {val_str};"
  LOGGER.info('Query Insert: %s', query)
  return query


def gen_select_objs(session, attribs, tablename, cond_str):
  """create a select query and generate name space objects for the results"""
  ret = get_job_rows(session, attribs, tablename, cond_str)
  entries = None

  if ret:
    entries = db_rows_to_obj(ret, attribs)

  return entries


def get_job_rows(session, attribs, tablename, cond_str):
  """Get db rows"""
  ret = None
  attr_str = ','.join(attribs)
  query = f"SELECT {attr_str} FROM {tablename}"\
          f" {cond_str};"
  LOGGER.info('Query Select: %s', query)
  try:
    ret = session.execute(query)
  except (Exception, KeyboardInterrupt) as ex:  #pylint: disable=broad-except
    LOGGER.warning(ex)
    ret = None
    session.rollback()

  return ret


def db_rows_to_obj(ret, attribs):
  """Compose SimpleDict list of db jobs"""
  entries = []
  for row in ret:
    #LOGGER.info('select_row: %s', row)
    entry = SimpleDict()
    for i, col in enumerate(attribs):
      setattr(entry, col, row[i])
    entries.append(entry)
  return entries


def has_attr_set(obj, attribs):
  """test if a namespace has the supplied attributes"""
  for attr in attribs:
    if not hasattr(obj, attr):
      return False
  return True


def get_class_by_tablename(tablename):
  """use tablename to find class"""
  # pylint: disable=protected-access
  for class_name in BASE._decl_class_registry.values():
    if hasattr(class_name,
               '__tablename__') and class_name.__tablename__ == tablename:
      return class_name
  return None


def build_dict_val_key(obj: SimpleDict, exclude: List[str] = ['id']):  # pylint: disable=W0102
  """take object with to_dict function and create a key using values from the object's \
  sorted keys"""
  obj_dict = obj.to_dict()
  for val in exclude:
    obj_dict.pop(val, False)
  obj_vals = [str(obj_dict[key]) for key in sorted(obj_dict.keys())]
  map_key: str = '-'.join(obj_vals)
  return map_key


def get_session_val_map(session: DbSession,
                        table: BASE,
                        attribs: List[str],
                        val: str = 'id'):
  """return a map of known object values to ids"""
  objs = gen_select_objs(session, attribs, table.__tablename__, "")
  val_map: Dict[str, int] = {}
  for obj in objs:
    map_key = build_dict_val_key(obj, exclude=[val])
    val_map[map_key] = obj.to_dict()[val]
  return val_map


class DB_Type(enum.Enum):  # pylint: disable=invalid-name ; @chris rename, maybe?
  """@alex defines the types of databases produced in tuning sessions?"""
  FIND_DB = 1
  KERN_DB = 2
  PERF_DB = 3


def attach_tensor_info(cfg_entries, session, cfg_rel):
  """Attach tensor relationship info"""
  for cfg in cfg_entries:
    for key, val in cfg_rel.items():
      rel_val = getattr(cfg, val['key'])
      rel_cond_str = f"where {val['fkey']}={rel_val}"
      setattr(
          cfg, key,
          gen_select_objs(session, val['fattr'], val['ftble'], rel_cond_str)[0])
