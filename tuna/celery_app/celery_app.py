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
"""Module to define celery app"""
import os
import subprocess
from celery import Celery
from celery.utils.log import get_task_logger
from tuna.custom_errors import CustomError

LOGGER = get_task_logger("celery_app")


def get_broker_env():
  """Set rabbitmq required env vars"""

  #defaults
  TUNA_CELERY_BROKER_HOST = 'localhost'
  TUNA_CELERY_BROKER_PORT = 5672

  if 'TUNA_CELERY_BROKER_USER' not in os.environ:
    raise CustomError('TUNA_CELERY_BROKER_USER must be specified in env')
  else:
    TUNA_CELERY_BROKER_USER = os.environ['TUNA_CELERY_BROKER_USER']
  if 'TUNA_CELERY_BROKER_PWD' not in os.environ:
    raise CustomError('TUNA_CELERY_BROKER_PWD must be specified in env')
  else:
    TUNA_CELERY_BROKER_PWD = os.environ['TUNA_CELERY_BROKER_PWD']

  if 'TUNA_CELERY_BROKER_HOST' in os.environ:
    TUNA_CELERY_BROKER_HOST = os.environ['TUNA_CELERY_BROKER_HOST']
  if 'TUNA_CELERY_BROKER_PORT' in os.environ:
    TUNA_CELERY_BROKER_PORT = os.environ['TUNA_CELERY_BROKER_PORT']
  if 'TUNA_CELERY_V_HOST' in os.environ:
    TUNA_CELERY_V_HOST = os.environ['TUNA_CELERY_V_HOST']

  return TUNA_CELERY_BROKER_HOST, TUNA_CELERY_BROKER_PORT, TUNA_CELERY_BROKER_USER, TUNA_CELERY_BROKER_PWD


def get_backend_env():
  """Get Redis env vars"""

  #defaults
  TUNA_CELERY_BACKEND_PORT = 6379
  TUNA_CELERY_BACKEND_HOST = 'localhost'

  if 'TUNA_CELERY_BACKEND_PORT' in os.environ:
    TUNA_CELERY_BACKEND_PORT = os.environ['TUNA_CELERY_BACKEND_PORT']
  if 'TUNA_CELERY_BACKEND_HOST' in os.environ:
    TUNA_CELERY_BACKEND_HOST = os.environ['TUNA_CELERY_BACKEND_HOST']

  return TUNA_CELERY_BACKEND_PORT, TUNA_CELERY_BACKEND_HOST


TUNA_CELERY_BROKER_HOST, TUNA_CELERY_BROKER_PORT, TUNA_CELERY_BROKER_USER, TUNA_CELERY_BROKER_PWD = get_broker_env(
)

TUNA_CELERY_BACKEND_PORT, TUNA_CELERY_BACKEND_HOST = get_backend_env()

#ampq borker & redis backend
app = Celery(
    'celery_app',
    broker_url=
    f"amqp://{TUNA_CELERY_BROKER_USER}:{TUNA_CELERY_BROKER_PWD}@{TUNA_CELERY_BROKER_HOST}:{TUNA_CELERY_BROKER_PORT}/",
    result_backend=
    f"redis://{TUNA_CELERY_BACKEND_HOST}:{TUNA_CELERY_BACKEND_PORT}/15",
    broker_transport_options={"heartbeat": 60},
    include=[
        'tuna.miopen.celery_tuning.celery_tasks',
        'tuna.example.celery_tuning.celery_tasks'
    ])


def stop_active_workers():
  """Shutdown active workers"""

  LOGGER.warning('Shutting down remote workers')
  try:
    if app.control.inspect().active() is not None:
      app.control.shutdown()
  except Exception as err:  #pylint: disable=broad-exception-caught
    LOGGER.warning('Exception occured while trying to shutdown workers: %s',
                   err)
    return False

  return True


def stop_named_worker(hostname):
  """Shutdown a specific worker"""
  LOGGER.warning('Shutting down remote worker: %s', hostname)
  try:
    app.control.shutdown(destination=[hostname])
  except Exception as exp:  #pylint: disable=broad-exception-caught
    LOGGER.warning('Exception occured while trying to shutdown workers: %s',
                   exp)
    return False

  return True


def purge_queue(q_names):
  """Purge jobs in queue"""
  for q_name in q_names:
    try:
      LOGGER.info('Purging Q %s', q_name)
      cmd = f"celery -A tuna.celery_app.celery_app purge -f -Q {q_name}".split(
          ' ')
      subp = subprocess.Popen(cmd)  #pylint: disable=consider-using-with
      #waiting for purge queue before continuing
      subp.wait()
    except Exception as exp:  #pylint: disable=broad-exception-caught
      LOGGER.info(exp)
      return False

  return True
