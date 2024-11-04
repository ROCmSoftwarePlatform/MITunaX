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
"""Utility functions to support fin commands
"""

from tuna.miopen.utils.metadata import NCHW_LAYOUT, NHWC_LAYOUT, NCDHW_LAYOUT, NDHWC_LAYOUT
from tuna.miopen.utils.metadata import PREC_TO_CMD, INVERS_DIR_MAP  #, DIR_MAP
from tuna.utils.logger import setup_logger
from tuna.utils.utility import arch2targetid
from tuna.miopen.utils.config_type import ConfigType
from tuna.miopen.driver.batchnorm import DriverBatchNorm

LOGGER = setup_logger('fin_utils')


def fin_job(steps, dynamic_only, job, config, dbt):
  """Construct a fin job dict from a config and a job
  """
  return_dict = {
      "steps": steps,
      "arch": arch2targetid(dbt.session.arch),
      "num_cu": dbt.session.num_cu,
      "config_tuna_id": config.id,
      "dynamic_only": dynamic_only,
  }

  if dbt.config_type == ConfigType.batch_norm:
    direction_t = int(config.forw) + 4 * int(config.back)
    return_dict['direction'] = direction_t
    return_dict["config"] = compose_config_obj(config, ConfigType.batch_norm)
    return_dict['config']['direction'] = direction_t
  else:
    return_dict["direction"] = int(INVERS_DIR_MAP[config.direction])
    return_dict["config"] = compose_config_obj(config, ConfigType.convolution)

  if job.solver:
    return_dict["solvers"] = [job.solver]

  print('fin_job: %s', return_dict)
  return return_dict


def get_fin_slv_status(json_obj, check_str):
  """Retrieve status information from fin json output, each represents a solver"""
  slv_stat = {}
  slv_stat['solver'] = json_obj['solver_name']
  slv_stat['success'] = json_obj[check_str]
  slv_stat['result'] = json_obj['reason']
  return slv_stat


def get_fin_result(status):
  """construct result string from status, and single success boolean"""
  #exception for Legacy Solver
  stat2 = []
  for suc, slv, res in [[x['success'], x['solver'], x['result']] for x in status
                       ]:
    entry = {'success': suc, 'solver': slv, 'result': res}
    if not suc and 'Legacy' in res:
      #legacy solvers won't return compiled kernels
      entry['success'] = True
    stat2.append(entry)

  result_str = ''
  success = False
  if True in [x['success'] for x in stat2]:
    success = True

  unanimous = False
  if len(stat2) > 0:
    res = stat2[0]['result']
    res_list = [res == val['result'] for val in stat2]
    unanimous = False not in res_list

  if unanimous:
    result_str = stat2[0]['result']
  else:
    for slv, res in [[x['solver'], x['result']] for x in stat2]:
      result_str += f' ({slv}: {res})'

  return success, result_str


def compose_config_obj(config, config_type=ConfigType.convolution):
  """Helper function to compose non-conv config obj"""
  return_config = {}
  input_t_dict = None
  in_layout = None
  weight_t_dict = None
  wei_layout = None
  cmd = None

  print(type(config))
  print(config)
  if config_type == ConfigType.convolution:
    return_config = compose_config_obj_conv(config, config_type)
  else:
    return_config = compose_config_obj_bn(config, config_type)
  #if 'input_t' in config.__dict__.keys():
  #  input_t_dict = {'input_t': config.input_t.to_dict()}
  #  cmd = PREC_TO_CMD[config_type][input_t_dict['input_t']['data_type']]
  #  in_layout = input_t_dict['input_t']['layout']
  #if 'weight_t' in config.__dict__.keys():
  #  weight_t_dict = {'weight_t': config.weight_t.to_dict()}
  #  cmd = PREC_TO_CMD[config_type][weight_t_dict['weight_t']['data_type']]
  #  wei_layout = weight_t_dict['weight_t']['layout']

  return return_config


def compose_config_obj_bn(config, config_type):
  driver = DriverBatchNorm(db_obj=config)
  print('driver: %s', driver.to_dict())
  return_config = config.to_dict()
  input_t_dict = update_input_t(return_config, config, config_type)

  return_config.update(get_tensor('in_layout', input_t_dict['input_t']))
  direction_t = int(config.forw) + 4 * int(config.back)
  #return_config['direction'] = DIR_MAP[direction_t]
  return_config['direction'] = direction_t
  return_config.pop('input_t')

  return return_config


def compose_config_obj_conv(config, config_type):
  return_config = config.to_dict()
  input_t_dict = update_input_t(return_config, config, config_type)
  weight_t_dict = update_weight_t(return_config, config, config_type)

  if return_config['in_layout'] != return_config['wei_layout'] != return_config[
      'out_layout']:
    LOGGER.error('Layouts must match. in_layout = %s, wei_layout=%s', in_layout,
                 wei_layout)
    return None

  return_config.update(get_tensor('in_layout', input_t_dict['input_t']))
  return_config.update(get_tensor('wei_layout', weight_t_dict['weight_t']))
  return_config.pop('input_t')
  return_config.pop('weight_t')

  return return_config


def update_input_t(return_config, config, config_type):
  input_t_dict = {}
  if 'input_t' in config.__dict__.keys():
    input_t_dict = {'input_t': config.input_t.to_dict()}
    cmd = PREC_TO_CMD[config_type][input_t_dict['input_t']['data_type']]
    in_layout = input_t_dict['input_t']['layout']
    return_config['cmd'] = cmd
  else:
    raise ValueError('input_t not present in config')

  return input_t_dict


def update_weight_t(return_config, config, config_type):
  weight_t_dict = {}
  if 'weight_t' in config.__dict__.keys():
    weight_t_dict = {'weight_t': config.weight_t.to_dict()}
    cmd = PREC_TO_CMD[config_type][weight_t_dict['weight_t']['data_type']]
    wei_layout = weight_t_dict['weight_t']['layout']
    return_config['cmd'] = cmd
  else:
    raise ValueError('input_t not present in config')

  return weight_t_dict


def get_tensor(tensor_type: str, tensor_dict: dict) -> dict:
  """Converts tensor dict to MIOpenDriver input"""
  ret_dict = {}
  layout = {}
  if tensor_dict['layout'] == 'NCHW':
    layout = NCHW_LAYOUT[tensor_type]
  elif tensor_dict['layout'] == 'NHWC':
    layout = NHWC_LAYOUT[tensor_type]
  elif tensor_dict['layout'] == 'NCDHW':
    layout = NCDHW_LAYOUT[tensor_type]
  elif tensor_dict['layout'] == 'NDHWC':
    layout = NDHWC_LAYOUT[tensor_type]
  else:
    LOGGER.error('unsupported layout: %s', tensor_dict['layout'])
  ret_dict[tensor_type] = tensor_dict['layout']
  for key, value in tensor_dict.items():
    if key in layout.keys():
      ret_dict[layout[key]] = value

  return ret_dict
