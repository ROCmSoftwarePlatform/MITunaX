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
"""Module that encapsulates the DB representation of a Driver cmd"""

from typing import List, Union, Dict, Any
from abc import abstractmethod
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm import Query
from sqlalchemy.inspection import inspect
from tuna.utils.logger import setup_logger
from tuna.dbBase.sql_alchemy import DbSession
from tuna.utils.db_utility import build_dict_val_key, get_session_val_map
from tuna.miopen.db.tensortable import TensorTable
from tuna.miopen.db.convolutionjob_tables import ConvolutionConfig
from tuna.miopen.utils.metadata import TENSOR_PRECISION
from tuna.miopen.utils.parsing import parse_line
from tuna.driver import DriverBase

LOGGER = setup_logger('MIOpenDriver_driver_base')


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
class MIOpenDriver(DriverBase):
  """Represents db tables based on ConfigType"""
  tensor_attr: List[str] = [column.name for column in inspect(TensorTable).c]
  tensor_id_map: Dict[str, int] = {}

  def __init__(self, line: str = str(), db_obj: ConvolutionConfig = None):
    super().__init__(line, db_obj)

  @abstractmethod
  def config_set_defaults(self) -> None:
    """Setting config DB defaults to avoid duplicates through SELECT"""
    raise NotImplementedError("Not implemented")

  @abstractmethod
  def set_cmd(self, data_type: str) -> None:
    """Set cmd based on tensor data type"""
    raise NotImplementedError("Not implemented")

  @abstractmethod
  def compose_weight_t(self) -> dict:
    """Build weight_tensor"""
    raise NotImplementedError("Not implemented")

  @abstractmethod
  def parse_row(self, db_obj: ConvolutionConfig):
    """Abstract/Inference for Overwritting base class function for batch_norm"""
    raise NotImplementedError("Not implemented")

  @abstractmethod
  def get_layouts(self):
    """Return operation layouts"""
    raise NotImplementedError("Not implemented")

  def parse_fdb_key(self, line: str) -> None:
    """Import config attributes from fdb key line"""
    raise NotImplementedError("Not implemented")

  @staticmethod
  def test_skip_arg(tok1: str):
    """Check if token is skipable"""
    raise NotImplementedError("Not implemented")

  @staticmethod
  def get_params(tok1: str):
    """Get full arg name"""
    raise NotImplementedError("Not implemented")

  @staticmethod
  def get_check_valid(tok1: str, tok2: Union[str, int]):
    """Check if valid conv arg"""
    raise NotImplementedError("Not implemented")

  @staticmethod
  def get_common_cols() -> List[str]:
    """Returns common MIOpenDriver command line args"""
    return ['wall', 'time', 'iter', 'verify']

  def construct_driver(self, line: str) -> bool:
    """Takes MIOpen line description of a configuration"""

    LOGGER.info('Processing line: %s', line)
    if line.find('=') != -1:
      self.parse_fdb_key(line)
    elif line.find('MIOpenDriver') != -1:
      self.parse_driver_line(line)
    else:
      LOGGER.warning('Skipping line: %s', line)
      return False

    self.config_set_defaults()
    return True

  def construct_driver_from_db(self, db_obj: Any) -> bool:
    """Takes a db row of a configuration and returns the string representation"""
    raise NotImplementedError("Not implemented")

  @staticmethod
  def get_tensor_id(session: Session, tensor_dict: dict) -> int:
    """Return tensor id based on dict"""

    query: Query
    ret_id: int = -1
    row: str
    query = Query(TensorTable.id).filter_by(**tensor_dict)
    try:
      res: list
      res = session.execute(query).fetchall()
      if len(res) > 1:
        LOGGER.error(tensor_dict)
        for row in res:
          LOGGER.error(row)
        raise ValueError('Tensor table duplication. Only one row should match')
      if not res:
        raise ValueError('Missing from Tensor table. One row should match')
      ret_id = res[0][0]
    except IntegrityError as err:
      session.rollback()
      LOGGER.error("Error occurred: %s \n", err)
      raise ValueError(
          'Something went wrong with getting input tensor id from tensor table'
      ) from err
    except IndexError as err:
      raise ValueError(f'Tensor not found in table: {tensor_dict}') from err

    return ret_id

  def __insert_tensor(self, tensor_dict: dict) -> int:
    """Insert new row into tensor table and return primary key"""
    ret_id: int = -1
    session: Session
    with DbSession() as session:
      try:
        tid = TensorTable(**tensor_dict)
        tid.valid = 1
        key = build_dict_val_key(tid)
        #cache the tensor table to avoid queries
        if not MIOpenDriver.tensor_id_map:
          MIOpenDriver.tensor_id_map = get_session_val_map(
              session, TensorTable, MIOpenDriver.tensor_attr)
        id_map = MIOpenDriver.tensor_id_map
        if key in id_map:
          ret_id = id_map[key]
          LOGGER.info("Get Tensor: %s", ret_id)
        else:
          session.add(tid)
          session.commit()
          ret_id = tid.id
          id_map[key] = ret_id
          LOGGER.info("Insert Tensor: %s", ret_id)
      except IntegrityError as err:
        LOGGER.warning(err)
        session.rollback()
        #update tensor table cache
        MIOpenDriver.tensor_id_map = get_session_val_map(
            session, TensorTable, MIOpenDriver.tensor_attr)
        ret_id = self.get_tensor_id(session, tensor_dict)
        LOGGER.info("Get Tensor: %s", ret_id)
    return ret_id

  def get_input_t_id(self) -> int:
    """Build 1 row in tensor table based on layout from fds param
       Details are mapped in metadata LAYOUT"""
    ret_id: int = -1
    i_dict: Dict[str, int]

    i_dict = self.__compose_input_t()
    ret_id = self.__insert_tensor(i_dict)

    return ret_id

  def __compose_input_t(self) -> Dict[str, int]:
    """Build input_tensor"""
    i_dict: Dict[str, int] = {}
    i_dict['data_type'] = TENSOR_PRECISION[self.cmd]
    i_dict['num_dims'] = self.num_dims
    i_dict['dim0'] = 1

    # conv: self.in_layout, BN: self.layout
    if (self.in_layout or self.layout) in ('NCHW', 'NCDHW'):
      i_dict['dim1'] = self.in_channels
      i_dict['dim2'] = self.in_d
      i_dict['dim3'] = self.in_h
      i_dict['dim4'] = self.in_w
      i_dict['layout'] = self.in_layout if self.in_layout else self.layout
    elif (self.in_layout or self.layout) in ('NHWC', 'NDHWC'):
      i_dict['dim1'] = self.in_d
      i_dict['dim2'] = self.in_h
      i_dict['dim3'] = self.in_w
      i_dict['dim4'] = self.in_channels
      i_dict['layout'] = self.in_layout if self.in_layout else self.layout

    return i_dict

  def decompose_input_t(self, db_obj: Any) -> bool:
    """Use input_tensor to assign local variables to build driver cmd """
    #pylint: disable=attribute-defined-outside-init

    self.set_cmd(db_obj.input_t.data_type)
    self.num_dims = db_obj.input_t.num_dims
    # conv: self.in_layout, BN: self.layout
    if self.in_layout:
      self.in_layout = db_obj.input_t.layout
    else:
      self.layout = db_obj.input_t.layout

    if (self.in_layout or self.layout) in ('NCHW', 'NCDHW'):
      self.in_channels = db_obj.input_t.dim1
      self.in_d = db_obj.input_t.dim2
      self.in_h = db_obj.input_t.dim3
      self.in_w = db_obj.input_t.dim4
    elif (self.in_layout or self.layout) in ('NHWC', 'NDHWC'):
      self.in_d = db_obj.input_t.dim1
      self.in_h = db_obj.input_t.dim2
      self.in_w = db_obj.input_t.dim3
      self.in_channels = db_obj.input_t.dim4

    return True

  def get_weight_t_id(self) -> int:
    """Build 1 row in tensor table based on layout from fds param
     Details are mapped in metadata LAYOUT"""
    ret_id: int = -1
    w_dict: dict = {}

    w_dict = self.compose_weight_t()
    ret_id = self.__insert_tensor(w_dict)
    return ret_id

  def parse_driver_line(self, line: str):
    """Parse line and set attributes"""

    tok: list
    tmp_line: str = parse_line(line)

    tok = tmp_line.split()
    #pylint: disable=attribute-defined-outside-init
    self.cmd = tok[1]
    assert tok[1] != ''

    self.compose_fds(tok, line)
    if "layout" in line:
      self.update_default_layouts(line)

  def compose_fds(self, tok: list, line: str) -> bool:
    """Compose fds from driver line"""
    tok1: str
    tok2: str
    f_digi_v: Union[int, str]
    for (tok1, tok2) in zip(tok[2::2], tok[3::2]):
      # the following would not work for a full name argument
      if tok1[0] == '-':
        flag_type: str = tok1[1:]
      if self.test_skip_arg(flag_type):
        continue
      flag_value: str = tok2.strip()
      if flag_value.isdigit():
        f_digi_v = int(flag_value)
      else:
        f_digi_v = flag_value

      flag_sh_value = self.get_params(flag_type)
      if self.get_check_valid(flag_sh_value[0], f_digi_v):
        setattr(self, flag_sh_value[0], f_digi_v)
      else:
        raise ValueError(
            f'Invalid command line arg for {self.cmd}: {flag_sh_value[0]} - {f_digi_v} line: {line}'
        )
    return True

  def update_default_layouts(self, line: str):
    """Overwrite default layouts by the specified layout(s)"""
    value_set: set = set()
    layouts: list = self.get_layouts()

    for layout in layouts:
      if layout in line:
        value_set.add(getattr(self, layout))

    if len(value_set) != 1:
      raise ValueError(f"Layouts do not match: [x for x in {layouts}]")

    driver_layout = value_set.pop()
    for layout in layouts:
      setattr(self, layout, driver_layout)

  def to_dict(self) -> Dict[str, Union[str, int]]:
    """Return class to dictionary"""
    copy_dict: Dict[str, Union[str, int]] = {}
    key: str
    value: Union[int, str]
    for key, value in vars(self).items():
      if key == "_cmd":
        copy_dict["cmd"] = value
      else:
        copy_dict[key] = value
    return copy_dict

  def __eq__(self, other: object) -> bool:
    """Defining equality functionality"""
    if not isinstance(other, DriverBase):
      return NotImplemented
    if self.__class__ != other.__class__:
      return False
    return vars(self) == vars(other)
