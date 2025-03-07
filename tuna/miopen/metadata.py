#!/usr/bin/env python3
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
""" This file contains mappings relevant to launching the MIOpen lib"""

MIOPEN_TUNING_STEPS = [
    'init_session', 'add_tables', 'update_applicability', 'update_solvers',
    'list_solvers', 'fin_steps', 'import_db', 'check_status', 'execute_cmd'
]

MIOPEN_CELERY_STEPS = [
    "miopen_find_compile", "miopen_find_eval", "miopen_perf_compile",
    "miopen_perf_eval"
]

MIOPEN_SUBCOMMANDS = [
    'import_configs', 'load_job', 'export_db', 'update_golden'
]

#tuning steps with 1 argument (possibly also --session_id)
MIOPEN_SINGLE_OP = [
    'init_session', 'add_tables', 'update_applicability', 'list_solvers'
]
