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
""" This file contains mappings relevant to launching MIOpenDriver and interacting with sql tables
"""

import os

from tuna.miopen.utils.config_type import ConfigType

DOCKER_CMD = "sudo docker run --device='/dev/kfd' --device='/dev/dri' -w /tmp/miopenpdb \
             -v /tmp/miopenpdb:/tmp/miopenpdb --user=root --group-add video --privileged=true \
             --rm {} bash  -c \"{}\""

MIOPEN_DB_VERSION = "1.0.0"
MIOPEN_USER_DB_PATH = "/tmp/miopenpdb/config/miopen"
MIOPEN_CACHE_DIR = "/tmp/miopenpdb/cache"
if 'MIOPEN_CACHE_DIR' in os.environ:
  MIOPEN_CACHE_DIR = os.environ['MIOPEN_CACHE_DIR']
KCACHE_DIR = f"{MIOPEN_CACHE_DIR}/tuna_kcache"
FIN_CACHE = "/tmp/miopenpdb/cache"
TUNA_DOCKER_NAME = 'miopentuna'
if 'TUNA_DOCKER_NAME' in os.environ:
  TUNA_DOCKER_NAME = os.environ['TUNA_DOCKER_NAME']
if 'FIN_CACHE' in os.environ:
  FIN_CACHE = os.environ['FIN_CACHE']

MYSQL_LOCK_WAIT_TIMEOUT = 1205

TABLE_COLS_CONV_MAP = {
    '-forw': ('direction', 0),
    'F': ('direction', 0),
    'H': ('in_h', 32),
    '-in_h': ('in_h', 32),
    'V': ('verify', 1),
    '-verify': ('verify', 1),
    'W': ('in_w', 32),
    '-in_w': ('in_w', 32),
    'c': ('in_channels', 3),
    '-in_channels': ('in_channels', 3),
    'g': ('group_count', 1),
    '-group_count': ('group_count', 1),
    'i': ('iter', 10),
    '-iter': ('iter', 10),
    'j': ('dilation_w', 1),
    '-dilation_w': ('dilation_w', 1),
    'k': ('out_channels', 32),
    '-out_channels': ('out_channels', 32),
    'l': ('dilation_h', 1),
    '-dilation_h': ('dilation_h', 1),
    'n': ('batchsize', 100),
    '-batchsize': ('batchsize', 100),
    'p': ('pad_h', 0),
    '-pad_h': ('pad_h', 0),
    'q': ('pad_w', 0),
    '-pad_w': ('pad_w', 0),
    'u': ('conv_stride_h', 1),
    '-conv_stride_h': ('conv_stride_h', 1),
    'v': ('conv_stride_w', 1),
    '-conv_stride_w': ('conv_stride_w', 1),
    'x': ('fil_w', 3),
    '-fil_w': ('fil_w', 3),
    'y': ('fil_h', 3),
    '-fil_h': ('fil_h', 3),
    'z': ('pad_mode', 'default'),
    '-pad_mode': ('pad_mode', 'default'),
    'm': ('mode', 'conv'),
    '-mode': ('mode', 'conv'),
    '!': ('in_d', '32'),
    '-in_d': ('in_d', '32'),
    '@': ('fil_d', '3'),
    '-fil_d': ('fil_d', '3'),
    '-dilation_d': ('dilation_d', '1'),
    '^': ('dilation_d', '1'),
    '-conv_stride_d': ('conv_stride_d', '1'),
    '#': ('conv_stride_d', '1'),
    '-pad_d': ('pad_d', '0'),
    '$': ('pad_d', '0'),
    '-spatial_dim': ('spatial_dim', '2'),
    '%': ('trans_output_pad_d', '0'),
    '-trans_output_pad_d': ('trans_output_pad_d', '0'),
    'X': ('trans_output_pad_w', '0'),
    '-trans_output_pad_w': ('trans_output_pad_w', '0'),
    'Y': ('trans_output_pad_h', '0'),
    '-trans_output_pad_h': ('trans_output_pad_h', '0'),
    '-in_layout': ('in_layout', 'NCHW'),
    'I': ('in_layout', 'NCHW'),
    '-out_layout': ('out_layout', 'NCHW'),
    'O': ('out_layout', 'NCHW'),
    '-fil_layout': ('fil_layout', 'NCHW'),
    'f': ('fil_layout', 'NCHW')
}

TABLE_COLS_BN_MAP = {
    '-alpha': ('alpha', 1.0),
    'A': ('alpha', 1.0),
    '-beta': ('beta', 0.0),
    'B': ('beta', 0.0),
    '!': ('in_d', 0),
    '-in_d': ('in_d', 0),
    'D': ('in_d', 0),
    '-forw': ('forw', 1),
    'F': ('forw', 1),
    'H': ('in_h', 32),
    '-in_h': ('in_h', 32),
    'V': ('verify', 1),
    '-verify': ('verify', 1),
    'W': ('in_w', 32),
    '-in_w': ('in_w', 32),
    '-back': ('back', 0),
    'b': ('back', 0),
    'c': ('in_channels', 3),
    '-in_channels': ('in_channels', 3),
    'i': ('iter', 1),
    '-iter': ('iter', 1),
    '-mode': ('mode', 0),
    'm': ('mode', 0),
    'n': ('batchsize', 32),
    '-batchsize': ('batchsize', 32),
    '-run': ('run', 0),
    'r': ('run', 0),
    '-save': ('save', 0),
    's': ('save', 0),
    '-layout': ('layout', 'NCHW'),
    'L': ('layout', 'NCHW')
}

SUPPORTED_LAYOUTS = ["NCHW", "NHWC", "NCDHW", "NDHWC"]

#NOTE: dim0 for input_tensor is 1
#3D layouts
NCDHW_LAYOUT = {
    'in_layout': {
        'dim1': 'in_channels',
        'dim2': 'in_d',
        'dim3': 'in_h',
        'dim4': 'in_w'
    },
    'layout': {
        'dim1': 'in_channels',
        'dim2': 'in_d',
        'dim3': 'in_h',
        'dim4': 'in_w'
    },
    'wei_layout': {
        'dim0': 'out_channels',
        'dim1': 'in_channels',
        'dim2': 'fil_d',
        'dim3': 'fil_h',
        'dim4': 'fil_w'
    }
}
NDHWC_LAYOUT = {
    'in_layout': {
        'dim1': 'in_d',
        'dim2': 'in_h',
        'dim3': 'in_w',
        'dim4': 'in_channels'
    },
    'layout': {
        'dim1': 'in_d',
        'dim2': 'in_h',
        'dim3': 'in_w',
        'dim4': 'in_channels'
    },
    'wei_layout': {
        'dim0': 'out_channels',
        'dim1': 'in_channels',
        'dim2': 'fil_d',
        'dim3': 'fil_h',
        'dim4': 'fil_w'
    }
}
#2D layouts
NCHW_LAYOUT = {
    'in_layout': {
        'dim1': 'in_channels',
        'dim2': 'in_d',
        'dim3': 'in_h',
        'dim4': 'in_w'
    },
    'layout': {
        'dim1': 'in_channels',
        'dim2': 'in_d',
        'dim3': 'in_h',
        'dim4': 'in_w'
    },
    'wei_layout': {
        'dim0': 'out_channels',
        'dim1': 'in_channels',
        'dim2': 'fil_d',
        'dim3': 'fil_h',
        'dim4': 'fil_w'
    }
}
NHWC_LAYOUT = {
    'in_layout': {
        'dim1': 'in_d',
        'dim2': 'in_h',
        'dim3': 'in_w',
        'dim4': 'in_channels'
    },
    'layout': {
        'dim1': 'in_d',
        'dim2': 'in_h',
        'dim3': 'in_w',
        'dim4': 'in_channels'
    },
    'wei_layout': {
        'dim0': 'out_channels',
        'dim1': 'in_channels',
        'dim2': 'fil_d',
        'dim3': 'fil_h',
        'dim4': 'fil_w'
    }
}

TENSOR_PRECISION = {
    'FP32': 'FP32',
    'conv': 'FP32',
    'FP16': 'FP16',
    'convfp16': 'FP16',
    'BF16': 'BF16',
    'convbfp16': 'BF16',
    'INT8': 'INT8',
    'convint8': 'INT8',
    'bnorm': 'FP32',
    'bnormfp16': 'FP16',
    'bnormbfp16': 'BFP16',
    'bnormfp16fp32': 'FP16FP32',
    'bnormbfp16fp32': 'BFP16FP32'
}

SUPPORTED_CONV_CMDS = ['conv', 'convfp16', 'convbfp16', 'convint8']
SUPPORTED_BN_CMDS = [
    'bnorm', 'bnormfp16', 'bnormbfp16', 'bnormfp16fp32', 'bnormbfp16fp32'
]

CONV_CONFIG_COLS = [
    'batchsize', 'spatial_dim', 'pad_h', 'pad_w', 'pad_d', 'conv_stride_h',
    'conv_stride_w', 'conv_stride_d', 'dilation_h', 'dilation_w', 'dilation_d',
    'group_count', 'mode', 'pad_mode', 'trans_output_pad_h',
    'trans_output_pad_w', 'trans_output_pad_d', 'out_layout', 'direction'
]

TENSOR_COLS = [
    'in_channels', 'out_channels', 'in_d', 'in_h', 'in_w', 'fil_d', 'fil_h',
    'fil_w'
]
IN_TENSOR_COLS = ['in_channels', 'in_d', 'in_h', 'in_w']

BN_CONFIG_COLS = [
    'batchsize', 'forw', 'mode', 'run', 'alpha', 'beta', 'back', 'run', 'save',
    'layout'
]

TABLE_COLS_FUSION_MAP = {
    'F': ('fusion_mode', 0),
    'H': ('in_h', 32),
    '-in_h': ('in_h', 32),
    'V': ('verify', 1),
    '-verify': ('verify', 1),
    'W': ('in_w', 32),
    '-in_w': ('in_w', 32),
    'c': ('in_channels', 3),
    '-in_channels': ('in_channels', 3),
    'i': ('iter', 1),
    '-iter': ('iter', 1),
    'j': ('dilation_w', 1),
    '-dilation_w': ('dilation_w', 1),
    'k': ('out_channels', 32),
    '-out_channels': ('out_channels', 32),
    'l': ('dilation_h', 1),
    '-dilation_h': ('dilation_h', 1),
    'm': ('activMode', 3),
    'n': ('batchsize', 32),
    '-batchsize': ('batchsize', 32),
    'p': ('pad_h', 0),
    '-pad_h': ('pad_h', 0),
    'q': ('pad_w', 0),
    '-pad_w': ('pad_w', 0),
    'u': ('conv_stride_h', 1),
    '-conv_stride_h': ('conv_stride_h', 1),
    'v': ('conv_stride_w', 1),
    '-conv_stride_w': ('conv_stride_w', 1),
    'x': ('fil_w', 3),
    '-fil_w': ('fil_w', 3),
    'y': ('fil_h', 3),
    '-fil_h': ('fil_h', 3)
}
FUSION_COLS = [v for _, v in TABLE_COLS_FUSION_MAP.items()]
CONV_COLS = [v for _, v in TABLE_COLS_CONV_MAP.items()]

CONV_SKIP_ARGS = ['i', 't', 'V', 's', 'b', 'w', 'S', 'Z']
BN_SKIP_ARGS = ['i', 't', 'V', 's', 'w', 'S']

DIR_MAP = {1: 'F', 2: 'B', 4: 'W'}
INVERS_DIR_MAP = {'F': 1, 'B': 2, 'W': 4}
DIRECTION = [1, 2, 4]

FIND_ONLY_EXCEPTION = {
    'gemm': 'MIOPEN_DEBUG_CONV_GEMM',
    'fft': 'MIOPEN_DEBUG_CONV_FFT'
}

ALG_TO_ENV = {
    'miopenConvolutionAlgoGEMM': 'MIOPEN_DEBUG_CONV_GEMM',
    'miopenConvolutionAlgoDirect': 'MIOPEN_DEBUG_CONV_DIRECT',
    'miopenConvolutionAlgoFFT': 'MIOPEN_DEBUG_CONV_FFT',
    'miopenConvolutionAlgoWinograd': 'MIOPEN_DEBUG_CONV_WINOGRAD',
    'miopenConvolutionAlgoImplicitGEMM': 'MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'
}

ENV_TO_ALG = {x: y for y, x in ALG_TO_ENV.items()}

ALG_SLV_MAP = {
    'miopenConvolutionAlgoGEMM': [
        'GemmFwd1x1_0_1', 'GemmFwd1x1_0_1_int8', 'GemmFwd1x1_0_2',
        'GemmFwdRest', 'GemmBwd1x1_stride2', 'GemmBwd1x1_stride1',
        'GemmBwdRest', 'GemmWrw1x1_stride1', 'GemmWrwUniversal'
    ],
    'miopenConvolutionAlgoDirect': [
        'ConvAsm3x3U', 'ConvAsm1x1U', 'ConvAsm1x1UV2', 'ConvBiasActivAsm1x1U',
        'ConvAsm5x10u2v2f1', 'ConvAsm5x10u2v2b1',
        'ConvAsm7x7c3h224w224k64u2v2p3q3f1', 'ConvOclDirectFwd11x11',
        'ConvOclDirectFwdGen', 'ConvOclDirectFwd3x3', 'ConvOclDirectFwd',
        'ConvOclDirectFwdFused', 'ConvOclDirectFwd1x1', 'ConvAsmBwdWrW3x3',
        'ConvAsmBwdWrW1x1', 'ConvOclBwdWrW2<1>', 'ConvOclBwdWrW2<2>',
        'ConvOclBwdWrW2<4>', 'ConvOclBwdWrW2<8>', 'ConvOclBwdWrW2<16>',
        'ConvOclBwdWrW2NonTunable', 'ConvOclBwdWrW53', 'ConvOclBwdWrW1x1'
    ],
    'miopenConvolutionAlgoFFT': ['fft'],
    'miopenConvolutionAlgoWinograd': [
        'ConvBinWinograd3x3U', 'ConvBinWinogradRxS',
        'ConvWinograd3x3MultipassWrW<3, 4>', 'ConvBinWinogradRxSf3x2',
        'ConvWinograd3x3MultipassWrW<3, 5>',
        'ConvWinograd3x3MultipassWrW<3, 6>',
        'ConvWinograd3x3MultipassWrW<3, 2>',
        'ConvWinograd3x3MultipassWrW<3, 3>',
        'ConvWinograd3x3MultipassWrW<7, 2>',
        'ConvWinograd3x3MultipassWrW<7, 3>',
        'ConvWinograd3x3MultipassWrW<7, 2, 1, 1>',
        'ConvWinograd3x3MultipassWrW<7, 3, 1, 1>',
        'ConvWinograd3x3MultipassWrW<1, 1, 7, 2>',
        'ConvWinograd3x3MultipassWrW<1, 1, 7, 3>',
        'ConvWinograd3x3MultipassWrW<5, 3>',
        'ConvWinograd3x3MultipassWrW<5, 4>', 'ConvBinWinogradRxSf2x3',
        'ConvMPBidirectWinograd<2, 3>', 'ConvMPBidirectWinograd<3, 3>',
        'ConvMPBidirectWinograd<4, 3>', 'ConvMPBidirectWinograd<5, 3>',
        'ConvMPBidirectWinograd<6, 3>', 'ConvMPBidirectWinograd_xdlops<2, 3>',
        'ConvMPBidirectWinograd_xdlops<3, 3>',
        'ConvMPBidirectWinograd_xdlops<4, 3>',
        'ConvMPBidirectWinograd_xdlops<5, 3>',
        'ConvMPBidirectWinograd_xdlops<6, 3>', 'ConvBinWinogradRxSf2x3g1'
    ],
    'miopenConvolutionAlgoImplicitGEMM': [
        'ConvHipImplicitGemmV4R1Fwd', 'ConvHipImplicitGemmV4R1WrW',
        'ConvHipImplicitGemmV4R4GenFwdXdlops',
        'ConvHipImplicitGemmV4R4GenWrWXdlops', 'ConvHipImplicitGemmV4R4Fwd',
        'ConvHipImplicitGemmBwdDataV1R1', 'ConvHipImplicitGemmBwdDataV4R1',
        'ConvHipImplicitGemmBwdDataV1R1Xdlops',
        'ConvHipImplicitGemmBwdDataV4R1Xdlops',
        'ConvHipImplicitGemmV4R4GenXdlopsFwdFp32',
        'ConvHipImplicitGemmV4R4GenXdlopsWrWFp32', 'ConvHipImplicitGemmV4R4WrW',
        'ConvAsmImplicitGemmV4R1DynamicFwd',
        'ConvAsmImplicitGemmV4R1DynamicFwd_1x1',
        'ConvHipImplicitGemmForwardV4R4Xdlops',
        'ConvAsmImplicitGemmV4R1DynamicBwd',
        'ConvAsmImplicitGemmV4R1DynamicWrw',
        'ConvAsmImplicitGemmGTCDynamicWrwXdlops',
        'ConvHipImplicitGemmWrwV4R4Xdlops',
        'ConvAsmImplicitGemmGTCDynamicFwdXdlops',
        'ConvHipImplicitGemmForwardV4R5Xdlops',
        'ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm',
        'ConvAsmImplicitGemmGTCDynamicBwdXdlops'
    ]
}

ENV_SLVGRP_MAP = {ALG_TO_ENV[x]: y for x, y in ALG_SLV_MAP.items()}

SLV_ALG_MAP = {}
for alg, slv_l in ALG_SLV_MAP.items():
  for slvr in slv_l:
    SLV_ALG_MAP[slvr] = alg
    SLV_ALG_MAP[slvr.replace(', ', '-')] = alg

SLV_ENV_MAP = {
    'ConvAsm3x3U':
        'MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U',
    'ConvAsm1x1U':
        'MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U',
    'ConvAsm1x1UV2':
        'MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2',
    'ConvAsm5x10u2v2f1':
        'MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2',
    'ConvAsm5x10u2v2b1':
        'MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2',
    'ConvAsm7x7c3h224w224k64u2v2p3q3f1':
        'MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224',
    'ConvAsmBwdWrW3x3':
        'MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3',
    'ConvOclDirectFwd11x11':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11',
    'ConvOclDirectFwdGen':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN',
    'ConvOclDirectFwd3x3':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD3X3',
    'ConvOclDirectFwd':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD',
    'ConvOclDirectFwd1x1':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1',
    'ConvOclBwdWrW2<1>':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2',
    'ConvOclBwdWrW2<2>':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2',
    'ConvOclBwdWrW2<4>':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2',
    'ConvOclBwdWrW2<8>':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2',
    'ConvOclBwdWrW2<16>':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2',
    'ConvOclBwdWrW2NonTunable':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2',
    'ConvOclBwdWrW53':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53',
    'ConvOclBwdWrW1x1':
        'MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1',
    'ConvBinWinograd3x3U':
        'MIOPEN_DEBUG_AMD_WINOGRAD_3X3',  #FP32 Winograd Fwd/Bwd, filter size fixed to 3x3.
    'ConvBinWinogradRxS':
        'MIOPEN_DEBUG_AMD_WINOGRAD_RXS',  #FP32 and FP16 Winograd Fwd/Bwd/WrW.
    #'ConvBinWinogradRxS':'MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW',
    #Subset of previous, controls only WrW (backward weights)
    #convolutions of the `ConvBinWinogradRxS` solver.
    'ConvBinWinogradRxSf3x2':
        'MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2',  #FP32 and FP16 Fwd/Bwd F(3,2) Winograd.
    'ConvWinograd3x3MultipassWrW<3, 2>':
        'MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2',  #WrW F(3,2) Multi-pass Winograd (stride 2 only).
    'ConvWinograd3x3MultipassWrW<3, 3>':
        'MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3',  #WrW F(3,3) Multi-pass Winograd (stride 2 only).
    'ConvWinograd3x3MultipassWrW<3, 4>':
        'MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4',  #WrW F(3,4) Multi-pass Winograd.
    'ConvWinograd3x3MultipassWrW<3, 5>':
        'MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5',  #WrW F(3,5) Multi-pass Winograd.
    'ConvWinograd3x3MultipassWrW<3, 6>':
        'MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6',  #WrW F(3,6) Multi-pass Winograd.
    'ConvWinograd3x3MultipassWrW<7, 2, 1, 1>':
        'MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2',  #WrW F(7x1,2x1) Multi-pass Winograd and
    #`ConvWinograd3x3MultipassWrW<1-1-7-2>`, WrW F(1x7,1x2) Multi-pass Winograd.
    'ConvWinograd3x3MultipassWrW<7, 3, 1, 1>':
        'MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3',  #WrW F(7x1,3x1) Multi-pass Winograd and
    #`ConvWinograd3x3MultipassWrW<1-1-7-3>`, WrW F(1x7,1x3) Multi-pass Winograd
    'Fused FP32 Winograd':
        'MIOPEN_DEBUG_AMD_FUSED_WINOGRAD'  #variable filter size.
}

#used in Parsing
FDS_3D_CONV = [
    'pad_d', 'pad_h', 'pad_w', 'out_channels', 'fil_d', 'fil_w', 'fil_h',
    'dilation_d', 'dilation_w', 'dilation_h', 'conv_stride_d', 'conv_stride_w',
    'conv_stride_h', 'in_channels', 'in_d', 'in_w', 'in_h', 'batchsize',
    'group_count', 'in_layout', 'out_layout', 'fil_layout'
]

#NOTE: Alex to check
FDS_3D_BN = ['in_channels', 'in_w', 'in_w', 'batchsize', 'layout']

FDS_2D_CONV = [
    'pad_h', 'pad_w', 'out_channels', 'fil_w', 'fil_h', 'dilation_w',
    'dilation_h', 'conv_stride_w', 'conv_stride_h', 'in_channels', 'in_w',
    'in_h', 'batchsize', 'group_count', 'in_layout', 'out_layout', 'fil_layout'
]

#NOTE: Alex to check
FDS_2D_BN = ['in_channels', 'in_w', 'in_w', 'batchsize', 'layout']

MIOPEN_ALG_LIST = [
    'MIOPEN_DEBUG_CONV_FFT', 'MIOPEN_DEBUG_CONV_DIRECT',
    'MIOPEN_DEBUG_CONV_GEMM', 'MIOPEN_DEBUG_CONV_WINOGRAD',
    'MIOPEN_DEBUG_CONV_IMPLICIT_GEMM', 'MIOPEN_DEBUG_CONV_SCGEMM'
]

FDB_FIELDS = [
    'id', 'fdb_key', 'config', 'solver', 'kernel_time', 'workspace_sz',
    'alg_lib', 'kcache_key', 'opencl', 'session'
]

SQLITE_CONFIG_COLS = [
    'layout', 'direction', 'data_type', 'spatial_dim', 'in_channels', 'in_h',
    'in_w', 'in_d', 'fil_h', 'fil_w', 'fil_d', 'out_channels', 'batchsize',
    'pad_h', 'pad_w', 'pad_d', 'conv_stride_h', 'conv_stride_w',
    'conv_stride_d', 'dilation_h', 'dilation_w', 'dilation_d', 'bias',
    'group_count'
]

CONV_2D_DEFAULTS = {
    'spatial_dim': 2,
    'cmd': 'conv',
    'mode': 'conv',
    'activMode': -1,
    'fusion_mode': -1,
    'in_channels': 3,
    'out_channels': 32,
    'in_h': 32,
    'in_w': 32,
    'in_d': 1,
    'fil_h': 3,
    'fil_w': 3,
    'fil_d': 1,
    'batchsize': 32,
    'pad_mode': 'default',
    'pad_h': 0,
    'pad_w': 0,
    'pad_d': 0,
    'conv_stride_h': 1,
    'conv_stride_w': 1,
    'conv_stride_d': 0,
    'dilation_h': 1,
    'dilation_w': 1,
    'dilation_d': 0,
    'trans_output_pad_h': 0,
    'trans_output_pad_w': 0,
    'trans_output_pad_d': 0,
    'in_layout': 'NCHW',
    'out_layout': 'NCHW',
    'fil_layout': 'NCHW',
    'group_count': 1,
    'num_dims': 2
}

CONV_3D_DEFAULTS = {
    'spatial_dim': 3,
    'cmd': 'conv',
    'mode': 'conv',
    'activMode': -1,
    'fusion_mode': -1,
    'in_channels': 3,
    'out_channels': 32,
    'in_h': 32,
    'in_w': 32,
    'in_d': 32,
    'fil_h': 3,
    'fil_w': 3,
    'fil_d': 3,
    'batchsize': 100,
    'pad_mode': 'default',
    'pad_h': 0,
    'pad_w': 0,
    'pad_d': 0,
    'conv_stride_h': 1,
    'conv_stride_w': 1,
    'conv_stride_d': 1,
    'dilation_d': 1,
    'dilation_h': 1,
    'dilation_w': 1,
    'trans_output_pad_h': 0,
    'trans_output_pad_w': 0,
    'trans_output_pad_d': 0,
    'in_layout': 'NCDHW',
    'out_layout': 'NCDHW',
    'fil_layout': 'NCDHW',
    'group_count': 1,
    'num_dims': 3
}

FUSION_DEFAULTS = {
    'cmd': 'CBAInfer',
    'pad_h': 0,
    'activMode': -1,
    'out_channels': 32,
    'fil_w': 3,
    'fusion_mode': -1,
    'dilation_w': 1,
    'fil_h': 3,
    'in_h': 32,
    'conv_stride_w': 1,
    'group_count': 1,
    'in_channels': 3,
    'in_w': 32,
    'dilation_h': 1,
    'conv_stride_h': 1,
    'pad_w': 0,
    'batchsize': 100,
    'pad_mode': 'default',
    'mode': 'conv',
    'fil_d': 3,
    'in_d': 32,
    'spatial_dim': 3,
    'conv_stride_d': 1,
    'dilation_d': 1,
    'pad_d': 0,
    'trans_output_pad_d': 0
}

ARCH_NUM_CU_LIST = [
    "gfx900-56", "gfx900-64", "gfx906-60", "gfx908-120", "gfx1030-36",
    "gfx90a-104", "gfx90a-110"
]

PREC_TO_CMD = {
    ConfigType.convolution: {
        'FP32': 'conv',
        'FP16': 'convfp16',
        'BF16': 'convbfp16',
        'INT8': 'convint8'
    },
    ConfigType.batch_norm: {
        'FP32': 'bnorm',
        'FP16': 'bnormfp16',
        'FP16FP32': 'bnormfp16fp32',
        'BFP16': 'bnormbfp16',
        'BFP16FP32': 'bnormbfp16fp32'
    }
}
CMD_TO_PREC = {
    'conv': 'FP32',
    'convfp16': 'FP16',
    'convbfp16': 'BF16',
    'convint8': 'INT8',
    'bnorm': 'FP32',
    'bnormfp16': 'FP16',
    'bnormbfp16': 'BFP16',
    'bnormfp16fp32': 'FP16FP32',
    'bnormBFP16FP32': 'BFP16FP32'
}

MYSQL_CONFIG_COLS = [
    'spatial_dim', 'in_channels', 'in_h', 'in_w', 'in_d', 'fil_h', 'fil_w',
    'fil_d', 'out_channels', 'batchsize', 'pad_h', 'pad_w', 'pad_d',
    'conv_stride_h', 'conv_stride_w', 'conv_stride_d', 'dilation_h',
    'dilation_w', 'dilation_d', 'group_count'
]

SQLITE_PERF_DB_COLS = ['config', 'solver', 'params']

MYSQL_PERF_CONFIG = ['layout', 'data_type', 'bias']
