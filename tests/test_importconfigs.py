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

sys.path.append("../tuna")
sys.path.append("tuna")

this_path = os.path.dirname(__file__)

from tuna.dbBase.sql_alchemy import DbSession
from tuna.utils.logger import setup_logger
from tuna.miopen.subcmd.import_configs import import_cfgs, add_benchmark
from tuna.miopen.subcmd.import_configs import add_model, add_frameworks, print_models
from tuna.sql import DbCursor
from tuna.miopen.db.tables import MIOpenDBTables, ConfigType
from utils import CfgImportArgs
from tuna.miopen.db.benchmark import Framework, ModelEnum, FrameworkEnum
from tuna.miopen.db.convolutionjob_tables import ConvolutionBenchmark
from utils import DummyArgs


def test_importconfigs():
  test_import_benchmark()
  test_import_conv()
  test_import_batch_norm()


def test_import_conv():
  dbt = MIOpenDBTables(config_type=ConfigType.convolution)
  logger = setup_logger('test_importconfigs')
  res = None
  clean_tags = "TRUNCATE table conv_config_tags;"
  find_tags = "SELECT count(*) FROM conv_config_tags WHERE tag='conv_config_test';"
  find_configs = "SELECT count(*) FROM conv_config;"
  with DbCursor() as cur:
    cur.execute(clean_tags)

  before_cfg_num = 0
  with DbCursor() as cur:
    cur.execute(find_configs)
    res = cur.fetchall()
    before_cfg_num = res[0][0]

  cfg_file = "{0}/../utils/configs/conv_configs.txt".format(this_path)
  add_cfg_NHWC = "{0}/../tuna/go_fish.py miopen import_configs -f {0}/../utils/configs/conv_configs.txt -t conv_config_test -V 1.0.0 -C convolution --model Alexnet --md_version 1 --framework Pytorch --fw_version 1".format(
      this_path)
  args = CfgImportArgs
  args.file_name = cfg_file
  args.tag = "conv_config_test"
  args.version = '1.0.0'
  counts = import_cfgs(args, dbt, logger)
  os.system(add_cfg_NHWC)

  with DbCursor() as cur:
    cur.execute(find_tags)
    res = cur.fetchall()
    assert (res[0][0] == len(counts['cnt_tagged_configs']))

    cur.execute(find_configs)
    res = cur.fetchall()
    after_cfg_num = res[0][0]
    assert (after_cfg_num - before_cfg_num == counts['cnt_configs'])


def test_import_batch_norm():
  dbt = MIOpenDBTables(config_type=ConfigType.batch_norm)
  res = None
  logger = setup_logger('test_importconfigs')
  clean_tags = "TRUNCATE table bn_config_tags;"
  find_tags = "SELECT count(*) FROM bn_config_tags WHERE tag='bn_config_test';"
  find_configs = "SELECT count(*) FROM bn_config;"
  with DbCursor() as cur:
    cur.execute(clean_tags)

  before_cfg_num = 0
  with DbCursor() as cur:
    cur.execute(find_configs)
    res = cur.fetchall()
    before_cfg_num = res[0][0]

  cfg_file = "{0}/../utils/configs/batch_norm.txt".format(this_path)
  add_cfg_NHWC = "{0}/../tuna/go_fish.py miopen import_configs -f {0}/../utils/configs/batch_norm.txt -t bn_config_test -V 1.0.0 -C batch_norm --model Alexnet --md_version 1 --framework Pytorch --fw_version 1".format(
      this_path)
  args = CfgImportArgs
  args.file_name = cfg_file
  args.tag = "bn_config_test"
  args.version = '1.0.0'
  args.config_type = ConfigType.batch_norm
  counts = import_cfgs(args, dbt, logger)
  os.system(add_cfg_NHWC)

  with DbCursor() as cur:
    cur.execute(find_tags)
    res = cur.fetchall()
    assert (res[0][0] == len(counts['cnt_tagged_configs']))

    cur.execute(find_configs)
    res = cur.fetchall()
    after_cfg_num = res[0][0]
    assert (after_cfg_num - before_cfg_num == counts['cnt_configs'])


def test_import_benchmark():
  args = DummyArgs
  logger = setup_logger('utest_import_benchmark')
  models = {
      ModelEnum.ALEXNET: 1.0,
      ModelEnum.GOOGLENET: 2.0,
      ModelEnum.VGG19: 3.0
  }
  for key, value in models.items():
    args.add_model = key.value
    args.version = value
    args.md_version = 1
    add_model(args, logger)
  print_models(logger)

  frameworks = {
      FrameworkEnum.PYTORCH: 1.0,
      FrameworkEnum.TENSORFLOW: 1.0,
      FrameworkEnum.MIGRAPH: 1.0
  }
  for key, value in frameworks.items():
    args.add_framework = key.value
    args.version = value
    args.fw_version = 1
    add_frameworks(args, logger)
  with DbSession() as session:
    frmks = session.query(Framework).all()
    assert len(frmks) > 0

  args.config_type = ConfigType.convolution
  dbt = MIOpenDBTables(session_id=None, config_type=args.config_type)
  args.driver = None
  args.add_benchmark = True
  args.framework = FrameworkEnum.PYTORCH
  args.model = ModelEnum.ALEXNET
  args.gpu_count = 8
  args.batchsize = 512
  args.file_name = "{0}/../utils/configs/conv_configs.txt".format(this_path)
  add_benchmark(args, dbt, logger)
  with DbSession() as session:
    bk_entries = session.query(ConvolutionBenchmark).all()
    assert len(bk_entries) > 0
