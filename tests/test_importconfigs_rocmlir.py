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

from tuna.utils.logger import setup_logger
from tuna.rocmlir.import_configs import import_cfgs
from tuna.sql import DbCursor
from tuna.rocmlir.rocmlir_tables import RocMLIRDBTables
from utils import CfgImportArgs

SAMPLE_CONV_CONFIGS = """
# This section of the file comes from resnet50-miopen-configs
-n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
-n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1

# This section of the file is selected conv configs for unet (stable diffusion).
-F 1 -n 2 -c 1280 -H 32 -W 32 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
-F 1 -n 2 -c 1280 -H 32 -W 32 -k 640 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
"""


def test_importconfigs_rocmlir():
  with open("test-conv-configs", 'w') as f:
    f.write(SAMPLE_CONV_CONFIGS)
  test_import_conv()


def test_import_conv():
  dbt = RocMLIRDBTables(session_id=None)
  logger = setup_logger('test_importconfigs')
  find_configs = "SELECT count(*) FROM rocmlir_conv_config;"

  with DbCursor() as cur:
    cur.execute(find_configs)
    res = cur.fetchall()
    before_cfg_num = res[0][0]

  args = CfgImportArgs
  args.file_name = "test-conv-configs"
  counts = import_cfgs(args, dbt, logger)

  with DbCursor() as cur:
    cur.execute(find_configs)
    res = cur.fetchall()
    after_cfg_num = res[0][0]
    assert (after_cfg_num - before_cfg_num == counts)
