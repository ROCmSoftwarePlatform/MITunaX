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

""" The necessary tables for rocMLIR tuning integration.
    Copied and adapted from example and miopen support.
"""

import enum
from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy import Text, Enum
from sqlalchemy.ext.declarative import declared_attr

from tuna.dbBase.base_class import BASE
from tuna.machine import Machine
from tuna.session_mixin import SessionMixin

#pylint: disable=too-few-public-methods

class SessionRocMLIR(BASE, SessionMixin):
  """Session table to keep track of tuning sessions"""
  #pylint: disable=attribute-defined-outside-init

  mlir_v = Column(String(length=64), nullable=False)

  __tablename__ = "session_rocmlir"
  __table_args__ = (UniqueConstraint("arch",
                                     "num_cu",
                                     "rocm_v",
                                     "mlir_v"
                                     "reason",
                                     name="uq_idx"),)

  def get_query(self, sess, sess_obj, entry):
    """get session matching this object"""
    query = sess.query(sess_obj.id)\
        .filter(sess_obj.arch == entry.arch)\
        .filter(sess_obj.num_cu == entry.num_cu)\
        .filter(sess_obj.rocm_v == entry.rocm_v)\
        .filter(sess_obj.mlir_v == entry.mlir_v)\
        .filter(sess_obj.reason == entry.reason)\

    return query

  def add_new_session(self, args, worker):
    """Add new session entry"""
    super().add_new_session(args, worker)
    return self.insert_session()


class JobEnum(enum.Enum):
  """Job status.  Numbers chosen to match miopen, which has more states."""
  # pylint: disable=invalid-name ; names represent entries in job_enum column
  # pylint: disable=duplicate-code
  new = 1
  running = 3
  completed = 4
  error = 5

class JobMixin():
  """Essential attributes for all jobs in the job tables"""

  @declared_attr
  def session(self):
    """session key, as a function to connect at run time"""
    return Column(Integer, ForeignKey("session.id"), nullable=False)

  reason = Column(String(length=60), nullable=False, server_default="")
  state = Column(Enum(JobEnum), nullable=False, server_default="new")
  retries = Column(Integer, nullable=False, server_default="0")
  result = Column(Text, nullable=True)

  compile_start = Column(DateTime,
                         nullable=False,
                         server_default=sqla_func.now())
  compile_end = Column(DateTime, nullable=False, server_default=sqla_func.now())
  gpu_id = Column(Integer, nullable=False, server_default="-1")
  machine_name = Column(String(length=60), nullable=False, server_default="")


class ConvolutionJob(BASE, JobMixin):
  """Represents convolutions job table"""
  __tablename__ = "conv_job"
  __table_args__ = (UniqueConstraint('config', 'session', name="uq_idx"),)

  config = Column(Integer,
                  ForeignKey("conv_config.id"),
                  nullable=False,
                  index=True)


class ConvolutionConfig(BASE):
  """Represents convolution config table"""
  __tablename__ = "conv_config"

  data_type = Column(String(length=60), nullable=False, server_default="")
  fil_layout = Column(String(60), nullable=False, server_default="NCHW")
  in_layout = Column(String(60), nullable=False, server_default="NCHW")
  out_layout = Column(String(60), nullable=False, server_default="NCHW")
  direction = Column(String(length=8), nullable=False)
  in_channels = Column(Integer, nullable=False, server_default="0")
  in_h = Column(Integer, nullable=False, server_default="0")
  in_w = Column(Integer, nullable=False, server_default="0")
  fil_h = Column(Integer, nullable=False, server_default="0")
  fil_w = Column(Integer, nullable=False, server_default="0")
  out_channels = Column(Integer, nullable=False, server_default="0")
  batchsize = Column(Integer, nullable=False, server_default="0")
  pad_h = Column(Integer, nullable=False, server_default="0")
  pad_w = Column(Integer, nullable=False, server_default="0")
  conv_stride_h = Column(Integer, nullable=False, server_default="0")
  conv_stride_w = Column(Integer, nullable=False, server_default="0")
  dilation_h = Column(Integer, nullable=False, server_default="0")
  dilation_w = Column(Integer, nullable=False, server_default="0")
  group_size = Column(Integer, nullable=False, server_default="0")
  kernel_repeats = Column(Integer, nullable=False, server_default="0")


class ConvolutionResults():  # pylint: disable=too-many-instance-attributes
  """Collects the results of convolution tuning.
  """
  __tablename__ = "conv_find_db"
  __table_args__ = (UniqueConstraint("config",
                                     "session",
                                     name="uq_idx"),)

  @orm.reconstructor
  def __init__(self, **kwargs):
    self.logger = kwargs['logger'] if 'logger' in kwargs else setup_logger(
        'find_db')

  @declared_attr
  def session(self):
    """session column"""
    return Column(Integer, ForeignKey("session.id"), nullable=False)

  config = Column(Integer, ForeignKey("conv_config.id"), nullable=False)

  perf_config = Column(Text, nullable=False)
  kernel_time = Column(Float, nullable=False)

  def get_query(self, sess, result_obj, session_id):
    """Construct a Db query for the find object
    """
    query = sess.query(result_obj).filter(result_obj.session == session_id,
                                          result_obj.config == self.config)
    self.logger.info("result query %s-%s", session_id, self.config)
    return query

  # +++pf:  rewrite me for tuningRunner.py output!
  def parse(self, decoded_line):
    """parse logger output line for find db data """
    retval = False
    if '[SetValues]' in decoded_line:
      message = decoded_line.split('[SetValues]')[1]
      key = message.split(',')[0].strip()

      if key != '':
        fdb = {}
        direction = key.split('-')[-1][:1]
        lead_str = 'content inserted: '
        #each entry has 5 fields, 0 - alg:slv, 1 - kernel_time, 2 - workspace size,
        #3 - alg, 4 - kernel cache key
        idx_start = message.index(lead_str) + len(lead_str)
        slv_info = message[idx_start:]
        columns = slv_info.split(',')
        while len(columns) >= FDB_SLV_NUM_FIELDS:
          (_, slv) = columns[0].split(':')
          if slv not in self.fdb_slv_dir:
            self.fdb_slv_dir[slv] = {}
          if direction not in self.fdb_slv_dir[slv]:
            self.fdb_slv_dir[slv][direction] = {}
            if 'ktimes' not in self.fdb_slv_dir[slv][direction]:
              self.fdb_slv_dir[slv][direction]['ktimes'] = []

          fdb = self.fdb_slv_dir[slv][direction]

          fdb['fdb_key'] = key
          kernel_time = float(columns[1])
          fdb['workspace_size'] = int(columns[2])
          fdb['alg_lib'] = columns[3]
          fdb['kcache_key'] = columns[4]
          fdb['is_ocl'] = 0
          if 'MIOpen(OpenCL)' in decoded_line:
            fdb['is_ocl'] = 1

          fdb['ktimes'].append(kernel_time)

          self.fdb_slv_dir[slv][direction] = fdb

          retval = True

          for _ in range(FDB_SLV_NUM_FIELDS):
            columns.pop(0)

    return retval
