# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import os
import logging
import numpy
import subprocess
import time
from datetime import timedelta


import torch
from torch.autograd import Variable


class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_output_dir(opt):
    filepath = os.path.join(opt.expName, 'main.log')

    if not os.path.exists(opt.expName):
        os.makedirs(opt.expName)

    # Safety check
    if os.path.exists(filepath) and opt.checkpoint == "":
        logging.warning("Experiment already exists!")

    # Create logger
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # quite down visdom
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info(opt)
    return logger


def wrap(data, **kwargs):
    if torch.is_tensor(data):
        var = Variable(data, **kwargs).cuda()
        return var
    else:
        return tuple([wrap(x, **kwargs) for x in data])


def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm(params, clip_th)
    return (not numpy.isfinite(befgad) or (befgad > ignore_th))


# Code taken from kastnerkyle gist:
# https://gist.github.com/kastnerkyle/cc0ac48d34860c5bb3f9112f4d9a0300

# Convenience function to reuse the defined env
def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=False):
    """
    Print and execute command on system
    """
    for line in execute(cmd, shell=shell):
        print(line, end="")


def array_to_binary_file(data, output_file_name):
    data = numpy.array(data, 'float32')
    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()


def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = numpy.fromfile(fid_lab, dtype=numpy.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    frame_number = features.size / dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return features, frame_number


def generate_merlin_wav(
        data, gen_dir, file_basename, norm_info_file,
        do_post_filtering=True, mgc_dim=60, fl=1024, sr=16000):
    # Made from Jose's code and Merlin
    gen_dir = os.path.abspath(gen_dir) + "/"
    if file_basename is None:
        base = "tmp_gen_wav"
    else:
        base = file_basename
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)

    file_name = os.path.join(gen_dir, base + ".cmp")
    fid = open(norm_info_file, 'rb')
    cmp_info = numpy.fromfile(fid, dtype=numpy.float32)
    fid.close()
    cmp_info = cmp_info.reshape((2, -1))
    cmp_mean = cmp_info[0, ]
    cmp_std = cmp_info[1, ]

    data = data * cmp_std + cmp_mean

    array_to_binary_file(data, file_name)
    # This code was adapted from Merlin. All licenses apply

    out_dimension_dict = {'bap': 1, 'lf0': 1, 'mgc': 60, 'vuv': 1}
    stream_start_index = {}
    file_extension_dict = {
        'mgc': '.mgc', 'bap': '.bap', 'lf0': '.lf0',
        'dur': '.dur', 'cmp': '.cmp'}
    gen_wav_features = ['mgc', 'lf0', 'bap']

    dimension_index = 0
    for feature_name in out_dimension_dict.keys():
        stream_start_index[feature_name] = dimension_index
        dimension_index += out_dimension_dict[feature_name]

    dir_name = os.path.dirname(file_name)
    file_id = os.path.splitext(os.path.basename(file_name))[0]
    features, frame_number = load_binary_file_frame(file_name, 63)

    for feature_name in gen_wav_features:

        current_features = features[
            :, stream_start_index[feature_name]:
            stream_start_index[feature_name] +
            out_dimension_dict[feature_name]]

        gen_features = current_features

        if feature_name in ['lf0', 'F0']:
            if 'vuv' in stream_start_index.keys():
                vuv_feature = features[
                    :, stream_start_index['vuv']:stream_start_index['vuv'] + 1]

                for i in range(frame_number):
                    if vuv_feature[i, 0] < 0.5:
                        gen_features[i, 0] = -1.0e+10  # self.inf_float

        new_file_name = os.path.join(
            dir_name, file_id + file_extension_dict[feature_name])

        array_to_binary_file(gen_features, new_file_name)

    pf_coef = 1.4
    fw_alpha = 0.58
    co_coef = 511

    sptkdir = os.path.abspath(os.path.dirname(__file__) + "/tools/SPTK-3.9/") + '/'
    sptk_path = {
        'SOPR': sptkdir + 'sopr',
        'FREQT': sptkdir + 'freqt',
        'VSTAT': sptkdir + 'vstat',
        'MGC2SP': sptkdir + 'mgc2sp',
        'MERGE': sptkdir + 'merge',
        'BCP': sptkdir + 'bcp',
        'MC2B': sptkdir + 'mc2b',
        'C2ACR': sptkdir + 'c2acr',
        'MLPG': sptkdir + 'mlpg',
        'VOPR': sptkdir + 'vopr',
        'B2MC': sptkdir + 'b2mc',
        'X2X': sptkdir + 'x2x',
        'VSUM': sptkdir + 'vsum'}

    worlddir = os.path.abspath(os.path.dirname(__file__) + "/tools/WORLD/") + '/'
    world_path = {
        'ANALYSIS': worlddir + 'analysis',
        'SYNTHESIS': worlddir + 'synth'}

    fw_coef = fw_alpha
    fl_coef = fl

    files = {'sp': base + '.sp',
             'mgc': base + '.mgc',
             'f0': base + '.f0',
             'lf0': base + '.lf0',
             'ap': base + '.ap',
             'bap': base + '.bap',
             'wav': base + '.wav'}

    mgc_file_name = files['mgc']
    cur_dir = os.getcwd()
    os.chdir(gen_dir)

    #  post-filtering
    if do_post_filtering:
        line = "echo 1 1 "
        for i in range(2, mgc_dim):
            line = line + str(pf_coef) + " "

        pe(
            '{line} | {x2x} +af > {weight}'
            .format(
                line=line, x2x=sptk_path['X2X'],
                weight=os.path.join(gen_dir, 'weight')), shell=True)

        pe(
            '{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | '
            '{c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
            .format(
                freqt=sptk_path['FREQT'], order=mgc_dim - 1,
                fw=fw_coef, co=co_coef, mgc=files['mgc'],
                c2acr=sptk_path['C2ACR'], fl=fl_coef,
                base_r0=files['mgc'] + '_r0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{freqt} -m {order} -a {fw} -M {co} -A 0 | '
            '{c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                freqt=sptk_path['FREQT'], fw=fw_coef, co=co_coef,
                c2acr=sptk_path['C2ACR'], fl=fl_coef,
                base_p_r0=files['mgc'] + '_p_r0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{mc2b} -m {order} -a {fw} | '
            '{bcp} -n {order} -s 0 -e 0 > {base_b0}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                mc2b=sptk_path['MC2B'], fw=fw_coef,
                bcp=sptk_path['BCP'], base_b0=files['mgc'] + '_b0'), shell=True)

        pe(
            '{vopr} -d < {base_r0} {base_p_r0} | '
            '{sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
            .format(
                vopr=sptk_path['VOPR'],
                base_r0=files['mgc'] + '_r0',
                base_p_r0=files['mgc'] + '_p_r0',
                sopr=sptk_path['SOPR'],
                base_b0=files['mgc'] + '_b0',
                base_p_b0=files['mgc'] + '_p_b0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{mc2b} -m {order} -a {fw} | '
            '{bcp} -n {order} -s 1 -e {order} | '
            '{merge} -n {order2} -s 0 -N 0 {base_p_b0} | '
            '{b2mc} -m {order} -a {fw} > {base_p_mgc}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                mc2b=sptk_path['MC2B'], fw=fw_coef,
                bcp=sptk_path['BCP'],
                merge=sptk_path['MERGE'], order2=mgc_dim - 2,
                base_p_b0=files['mgc'] + '_p_b0',
                b2mc=sptk_path['B2MC'],
                base_p_mgc=files['mgc'] + '_p_mgc'), shell=True)

        mgc_file_name = files['mgc'] + '_p_mgc'

    # Vocoder WORLD

    pe(
        '{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | '
        '{x2x} +fd > {f0}'
        .format(
            sopr=sptk_path['SOPR'], lf0=files['lf0'],
            x2x=sptk_path['X2X'], f0=files['f0']), shell=True)

    pe(
        '{sopr} -c 0 {bap} | {x2x} +fd > {ap}'.format(
            sopr=sptk_path['SOPR'], bap=files['bap'],
            x2x=sptk_path['X2X'], ap=files['ap']), shell=True)

    pe(
        '{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | '
        '{sopr} -d 32768.0 -P | {x2x} +fd > {sp}'.format(
            mgc2sp=sptk_path['MGC2SP'], alpha=fw_alpha,
            order=mgc_dim - 1, fl=fl, mgc=mgc_file_name,
            sopr=sptk_path['SOPR'], x2x=sptk_path['X2X'], sp=files['sp']),
        shell=True)

    pe(
        '{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'.format(
            synworld=world_path['SYNTHESIS'], fl=fl, sr=sr,
            f0=files['f0'], sp=files['sp'], ap=files['ap'],
            wav=files['wav']),
        shell=True)

    pe(
        'rm -f {ap} {sp} {f0} {bap} {lf0} {mgc} {mgc}_b0 {mgc}_p_b0 '
        '{mgc}_p_mgc {mgc}_p_r0 {mgc}_r0 {cmp} weight'.format(
            ap=files['ap'], sp=files['sp'], f0=files['f0'],
            bap=files['bap'], lf0=files['lf0'], mgc=files['mgc'],
            cmp=base + '.cmp'),
        shell=True)
    os.chdir(cur_dir)
