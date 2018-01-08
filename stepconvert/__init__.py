#!/usr/bin/env python3

from math import sqrt
from sys import argv
import sh
from os import listdir
import cv2

# Begin registers
CAL_ACOMMON_L = 0xD0
CAL_ACOMMON_H = 0xD1
CAL_ACP_L = 0xD3
CAL_ACP_H = 0xD4
CAL_BCP = 0xD5
CAL_alphaCP_L = 0xD6
CAL_alphaCP_H = 0xD7
CAL_TGC = 0xD8
CAL_AI_SCALE = 0xD9
CAL_BI_SCALE = 0xD9


VTH_L = 0xDA
VTH_H = 0xDB
KT1_L = 0xDC
KT1_H = 0xDD
KT2_L = 0xDE
KT2_H = 0xDF
KT_SCALE = 0xD2

# Common sensitivity coefficients
CAL_A0_L = 0xE0
CAL_A0_H = 0xE1
CAL_A0_SCALE = 0xE2
CAL_DELTA_A_SCALE = 0xE3
CAL_EMIS_L = 0xE4
CAL_EMIS_H = 0xE5
CAL_KSTA_L = 0xE6
CAL_KSTA_H = 0xE7


# Config register = 0xF5-F6
OSC_TRIM_VALUE = 0xF7

# Bits within configuration register 0x92
POR_TEST = 10

configLSB = 0b00111011
configMSB = 0b01000110
resolution = 0b11


def read_eeprom(fname):
    with open(fname) as FILE:
        text = FILE.read()
        text = text.strip()
        eeprom = [int(v) for v in text.split(',')]
    return eeprom


def read_ptat(fname):
    with open(fname) as FILE:
        text = FILE.read()
        text = text.strip()
        ptat = int(text.split(',')[-2])
    return ptat


def read_ir(fname):
    with open(fname) as FILE:
        text = FILE.read()
        text = text.strip()
        raw_ir = [int(v) for v in text.split(',')[:-2]]
    return raw_ir


def read_cpix(fname):
    with open(fname) as FILE:
        text = FILE.read()
        text = text.strip()
        cpix = int(text.split(',')[-1])
    return cpix


def read_config():
    return unsigned_16(configMSB, configLSB)


def twos_16(high_byte, low_byte):
    word = unsigned_16(high_byte, low_byte)
    if word > 32767:
        return word - 65536
    return word


def twos_8(byte):
    if byte > 127:
        return byte-256
    return byte


def unsigned_16(high_byte, low_byte):
    return (high_byte << 8) | low_byte


def calculate_ta(consts, ptat):
    ta = (consts['k_t1'] ** 2) - (4 * consts['k_t2'] * (consts['v_th'] - ptat))
    ta = -consts['k_t1'] + sqrt(ta)
    ta /= (2 * consts['k_t2'])
    ta += 25.0
    return ta


def precalculate_constants(eeprom):
    consts = dict()
    consts['resolution_comp'] = 2.0 ** (3 - resolution)
    consts['emissivity'] = unsigned_16(eeprom[CAL_EMIS_H], eeprom[CAL_EMIS_L]) / 32768.0
    consts['a_common'] = twos_16(eeprom[CAL_ACOMMON_H], eeprom[CAL_ACOMMON_L])
    consts['a_i_scale'] = (eeprom[CAL_AI_SCALE] & 0xF0) >> 4
    consts['b_i_scale'] = eeprom[CAL_BI_SCALE] & 0x0F
    consts['alpha_cp'] = unsigned_16(eeprom[CAL_alphaCP_H], eeprom[CAL_alphaCP_L])
    consts['alpha_cp'] /= (2.0 ** eeprom[CAL_A0_SCALE]) * consts['resolution_comp']
    consts['a_cp'] = twos_16(eeprom[CAL_ACP_H], eeprom[CAL_ACP_L]) / consts['resolution_comp']
    consts['b_cp'] = twos_8(eeprom[CAL_BCP]) / ((2.0 ** consts['b_i_scale']) * consts['resolution_comp'])
    consts['tgc'] = twos_8(eeprom[CAL_TGC]) / 32.0
    consts['k_t1_scale'] = (eeprom[KT_SCALE] & 0xF0) >> 4
    consts['k_t2_scale'] = (eeprom[KT_SCALE] & 0x0F) + 10
    consts['v_th'] = twos_16(eeprom[VTH_H], eeprom[VTH_L]) / consts['resolution_comp']
    consts['k_t1'] = twos_16(eeprom[KT1_H], eeprom[KT1_L])
    consts['k_t1'] /= ((2 ** consts['k_t1_scale']) * consts['resolution_comp'])
    consts['k_t2'] = twos_16(eeprom[KT2_H], eeprom[KT2_L])
    consts['k_t2'] /= ((2 ** consts['k_t2_scale']) * consts['resolution_comp'])
    return consts


def calculate_to(eeprom, ir_data, consts, ta, cpix):
    v_cp_off_comp = cpix - (consts['a_cp'] + consts['b_cp'] * (ta - 25.0))
    tak4 = (ta + 273.15) ** 4.0
    min_temp = 100000000
    max_temp = -100000000
    temps = []
    for i in range(64):
        a_ij = (consts['a_common'] + eeprom[i] * (2.0 ** consts['a_i_scale'])) / consts['resolution_comp']
        b_ij = twos_8(eeprom[0x40 + i]) / ((2.0 ** consts['b_i_scale']) * consts['resolution_comp'])
        v_ir_off_comp = ir_data[i] - (a_ij + b_ij * (ta - 25.0))
        v_ir_tgc_comp = v_ir_off_comp - consts['tgc'] * v_cp_off_comp
        alpha_ij = unsigned_16(eeprom[CAL_A0_H], eeprom[CAL_A0_L]) / (2.0 ** eeprom[CAL_A0_SCALE])
        alpha_ij += eeprom[0x80 + i] / (2.0 ** eeprom[CAL_DELTA_A_SCALE])
        alpha_ij /= consts['resolution_comp']
        alpha_comp = (alpha_ij - consts['tgc'] * consts['alpha_cp'])
        v_ir_comp = v_ir_tgc_comp / consts['emissivity']
        temperature = (((v_ir_comp / alpha_comp) + tak4) ** (1.0 / 4.0)) - 273.15
        temps.append(temperature)
        if temperature > max_temp:
            max_temp = temperature
        if temperature < min_temp:
            min_temp = temperature
    return temps, min_temp, max_temp

def fahrenheit(tc):
    return (tc*9/5)+32


def process_dir(batch_dir):
    if batch_dir[-1] != '/':
        batch_dir += '/'
    # Get directories
    eeprom_file = batch_dir + 'eeprom.csv'
    ir_cap_dir = batch_dir + 'ircapture/'
    cam_dir = batch_dir + 'cameraimages/'
    # Process the EEPROM data
    eeprom_data = read_eeprom(eeprom_file)
    constants = precalculate_constants(eeprom_data)
    # Count the images, making sure the IR and RGB match
    num_files = len(listdir(ir_cap_dir))
    if num_files != len(listdir(cam_dir)):
        raise FileNotFoundError
    ir_imgs = []
    imgs = []
    # Process the images
    for n in range(num_files):
        # Get the filenames
        fnum = str(n)
        if n < 100:
            fnum = '0' + fnum
            if n < 10:
                fnum = '0' + fnum
        ir_file = ir_cap_dir + 'pic' + fnum + '.txt'
        cam_name = cam_dir + 'pic' + fnum + '.png'#'.png'
        # Preliminary IR calculations
        ir_vals = read_ir(ir_file)
        ptat_val = read_ptat(ir_file)
        cpix_val = read_cpix(ir_file)
        TA = calculate_ta(constants, ptat_val)
        # Get the temperatures
        try:
            temperatures, max_t, min_t = calculate_to(eeprom_data, ir_vals, constants, TA, cpix_val)
            ir_imgs.append([str(fahrenheit(t)) for t in temperatures])
        except TypeError:
            print('FAIL ', fnum, '! Continuing ... ', sep='', end='')
        # Rotate the RGB images
        rot_name = '/tmp/rot.jpg'#'.png'
        sh.convert(cam_name, '-rotate', '270', rot_name)
        imgs.append(cv2.imread(rot_name))
    # Clean-up
    sh.rm(rot_name)
    return imgs, ir_imgs


def main(batch_dir):
#if __name__ == '__main__':
    print('Starting ...')
    #batch_dir = argv[1]
    if batch_dir[-1] != '/':
        batch_dir += '/'
    eeprom_file = batch_dir + 'eeprom.csv'
    eeprom_data = read_eeprom(eeprom_file)
    print('Read EEPROM data... ')
    constants = precalculate_constants(eeprom_data)
    print('Calculated constants ...')
    ir_cap_dir = batch_dir + 'ircapture/'
    temps_dir = batch_dir + 'temperatures/'
    rot_dir = batch_dir + 'rotated/'
    cam_dir = batch_dir + 'cameraimages/'
    try:
        sh.mkdir(temps_dir)
    except sh.ErrorReturnCode_1:
        pass
    num_files = len(listdir(ir_cap_dir))
    for n in range(num_files):
        fnum = str(n)
        if n < 100:
            fnum = '0' + fnum
            if n < 10:
                fnum = '0' + fnum
        print('Processing', fnum, '... ', end='')
        ir_file = ir_cap_dir + 'pic' + fnum + '.txt'
        temp_file = temps_dir + 'img' + fnum + '.csv'
        ir_vals = read_ir(ir_file)
        ptat_val = read_ptat(ir_file)
        cpix_val = read_cpix(ir_file)
        TA = calculate_ta(constants, ptat_val)
        try:
            temperatures, max_t, min_t = calculate_to(eeprom_data, ir_vals, constants, TA, cpix_val)
            with open(temp_file, 'w') as T_FILE:
                T_FILE.write(','.join([str(fahrenheit(t)) for t in temperatures]))
                T_FILE.write('\n')
        except TypeError:
            print('FAIL ', fnum, '! Continuing ... ', sep='', end='')
    try:
        sh.mkdir(rot_dir)
    except sh.ErrorReturnCode_1:
        pass
    num_files = len(listdir(cam_dir))
    for n in range(num_files):
        fnum = str(n)
        if n < 100:
            fnum = '0' + fnum
            if n < 10:
                fnum = '0' + fnum
        cam_name = cam_dir + 'pic' + fnum + '.png'#'.png'
        rot_name = rot_dir + 'img' + fnum + '.png'#'.png'
        sh.convert(cam_name, '-rotate', '270', rot_name)
    print('\nDone.')


if __name__=='__main__':
    main(argv[1])

