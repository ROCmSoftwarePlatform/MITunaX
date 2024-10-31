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
"""Module that encapsulates the DB representation based on configuration type"""

from tuna.miopen.db.batch_norm_tables import BNBenchmark, BNConfig
from tuna.miopen.db.batch_norm_tables import BNConfigTags, BNFinJobCache
from tuna.miopen.db.batch_norm_tables import BNJob, BNJobCache, BNKernelCache
from tuna.miopen.db.batch_norm_tables import BNSolverApplicability
from tuna.miopen.db.bn_golden_tables import BNGolden
from tuna.miopen.db.find_db import ConvolutionFindDB, BNFindDB
from tuna.miopen.db.convolutionjob_tables import ConvolutionJob
from tuna.miopen.db.convolutionjob_tables import ConvolutionConfig
from tuna.miopen.db.convolutionjob_tables import ConvolutionConfigTags
from tuna.miopen.db.convolutionjob_tables import ConvJobCache
from tuna.miopen.db.convolutionjob_tables import ConvSolverApplicability
from tuna.miopen.db.convolutionjob_tables import ConvolutionGolden, ConvolutionBenchmark
from tuna.miopen.db.convolutionjob_tables import ConvFinJobCache, ConvolutionKernelCache
from tuna.miopen.db.convolutionjob_tables import ConvSolverAnalyticsAggregated
from tuna.miopen.db.convolutionjob_tables import ConvSolverAnalyticsDetailed
from tuna.miopen.db.solver import Solver
from tuna.miopen.db.tensortable import TensorTable
from tuna.miopen.db.benchmark import Framework, Model
from tuna.miopen.utils.config_type import ConfigType
from tuna.tables_interface import DBTablesInterface
from tuna.miopen.db.session import Session


#pylint: disable=too-many-instance-attributes
#pylint: disable=too-few-public-methods
class MIOpenDBTables(DBTablesInterface):
  """Represents db tables based on ConfigType"""

  def __init__(self, **kwargs):
    """Constructor"""
    super().__init__()
    allowed_keys = set(['config_type', 'session_id'])
    self.__dict__.update((key, None) for key in allowed_keys)

    #for pylint
    self.config_table = None
    self.config_tags_table = None
    self.find_db_table = None
    self.solver_app = None
    self.cache_table = None
    self.fin_cache_table = None
    self.solver_table = None
    self.kernel_cache = None
    self.tensor_table = TensorTable
    self.golden_table = None
    self.config_type = None
    self.solver_analytics_aggregated = None
    self.solver_analytics_detailed = None
    self.session_table = Session

    self.__dict__.update(
        (key, value) for key, value in kwargs.items() if key in allowed_keys)
    self.set_tables()

  def set_tables(self, sess_class=Session):
    """Set appropriate tables based on config type"""

    self.solver_table = Solver
    self.model = Model
    self.framework = Framework

    super().set_tables(sess_class)
    if self.config_type == ConfigType.batch_norm:
      self.job_table = BNJob
      self.config_table = BNConfig
      self.config_tags_table = BNConfigTags
      self.find_db_table = BNFindDB
      self.solver_app = BNSolverApplicability
      self.cache_table = BNJobCache
      self.fin_cache_table = BNFinJobCache
      self.kernel_cache = BNKernelCache
      self.benchmark = BNBenchmark
      self.golden_table = BNGolden
    else:
      self.job_table = ConvolutionJob
      self.config_table = ConvolutionConfig
      self.config_tags_table = ConvolutionConfigTags
      self.find_db_table = ConvolutionFindDB
      self.solver_app = ConvSolverApplicability
      self.cache_table = ConvJobCache
      self.fin_cache_table = ConvFinJobCache
      self.kernel_cache = ConvolutionKernelCache
      self.golden_table = ConvolutionGolden
      self.benchmark = ConvolutionBenchmark
      self.solver_analytics_aggregated = ConvSolverAnalyticsAggregated
      self.solver_analytics_detailed = ConvSolverAnalyticsDetailed
