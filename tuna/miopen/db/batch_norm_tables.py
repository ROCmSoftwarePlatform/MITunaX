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
"""Represents Batch normalization table definitions """

from sqlalchemy import Column, Integer, String, UniqueConstraint, ForeignKey
from sqlalchemy.orm import relationship
from tuna.dbBase.base_class import BASE
from tuna.miopen.db.mixin_tables import BenchmarkMixin, CacheMixin
from tuna.miopen.db.mixin_tables import ConfigTagMixin, KernelCacheMixin
from tuna.miopen.db.mixin_tables import MIOpenJobMixin, SolverApplicabilityMixin
from tuna.miopen.utils.metadata import DIR_MAP

COMMON_UNIQ_FDS = ["config", "solver", "session"]

#pylint: disable=too-few-public-methods
#pylint: disable=duplicate-code


class BNJob(BASE, MIOpenJobMixin):
  """Represents batch norm job table"""
  __tablename__ = "bn_job"
  __table_args__ = (UniqueConstraint(*COMMON_UNIQ_FDS, name="uq_idx"),)

  config = Column(Integer,
                  ForeignKey("bn_config.id"),
                  nullable=False,
                  index=True)


class BNConfig(BASE):
  """Represents batch normalization table"""
  __tablename__ = "bn_config"
  __table_args__ = (UniqueConstraint("alpha",
                                     "beta",
                                     "forw",
                                     "verify",
                                     "back",
                                     "mode",
                                     "batchsize",
                                     "run",
                                     "input_tensor",
                                     name="uq_idx"),)

  alpha = Column(Integer, nullable=False, server_default="1.0")
  beta = Column(Integer, nullable=False, server_default="0.0")
  forw = Column(Integer, nullable=False, server_default="1")
  verify = Column(Integer, nullable=False, server_default="1")
  back = Column(Integer, nullable=False, server_default="0")
  mode = Column(Integer, nullable=False, server_default="0")
  batchsize = Column(Integer, nullable=False, server_default="32")
  run = Column(Integer, nullable=False, server_default="0")
  save = Column(Integer, nullable=False, server_default="0")
  input_tensor = Column(Integer, ForeignKey("tensor.id"), nullable=False)
  input_t = relationship("TensorTable",
                         backref="bn_input_tensor",
                         foreign_keys=[input_tensor],
                         lazy="joined")
  layout = Column(String(60), nullable=False, server_default="NCHW")
  driver = Column(String(length=512), nullable=False, server_default="")

  #pylint: disable=too-few-public-methods
  def get_direction(self):
    """synthesize direction"""
    return DIR_MAP[(self.forw + 4 * self.back)]


class BNConfigTags(BASE, ConfigTagMixin):
  """Represents config_tags tables"""
  __tablename__ = "bn_config_tags"
  __table_args__ = (UniqueConstraint("config", "tag", name="uq_idx"),)

  config = Column(Integer, ForeignKey("bn_config.id"), nullable=False)


class BNSolverApplicability(BASE, SolverApplicabilityMixin):
  """Represents bn_solver_applicability  table"""
  __tablename__ = "bn_solver_applicability"
  __table_args__ = (UniqueConstraint(*COMMON_UNIQ_FDS, name="uq_idx"),)

  config = Column(Integer,
                  ForeignKey("bn_config.id"),
                  nullable=False,
                  index=True)


class BNJobCache(BASE, CacheMixin):
  """Represents job_cache table for batch_norm"""
  __tablename__ = "bn_job_cache"
  __table_args__ = (UniqueConstraint("job_id", name="uq_cache_idx"),)

  job_id = Column(Integer, ForeignKey("bn_job.id"), nullable=False)


class BNFinJobCache(BASE, KernelCacheMixin):
  """Represents job_cache table for batch_norm"""
  __tablename__ = "bn_job_cache_fin"

  job_id = Column(Integer,
                  ForeignKey("bn_job.id",
                             onupdate="CASCADE",
                             ondelete="CASCADE"),
                  nullable=False)
  solver_id = Column(Integer,
                     ForeignKey("solver.id",
                                onupdate="CASCADE",
                                ondelete="CASCADE"),
                     nullable=False)


class BNKernelCache(BASE, KernelCacheMixin):
  """Represents kernel_cache table for batch_norm"""
  __tablename__ = "bn_kernel_cache"

  kernel_group = Column(Integer, nullable=True)


class BNBenchmark(BASE, BenchmarkMixin):
  """benchmark table for framework and model parameters"""
  __tablename__ = "bn_benchmark"
  __table_args__ = (UniqueConstraint("framework",
                                     "model",
                                     "batchsize",
                                     "gpu_number",
                                     "config",
                                     name="uq_idx"),)

  config = Column(Integer, ForeignKey("bn_config.id"), nullable=False)
