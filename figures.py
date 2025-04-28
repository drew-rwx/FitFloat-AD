#!/usr/bin/env python3

#
# This file is part of FitFloat, a floating-point array representation for GPUs that allows the user to choose the number of bits in the exponent and mantissa fields.
#
# BSD 3-Clause License
#
# Copyright (c) 2025, Andrew Rodriguez and Martin Burtscher
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# URL: The latest version of this code is available at https://github.com/burtscher/FitFloat.
#
# Sponsor: This code is based upon work supported by the U.S. Department of Energy, National Nuclear Security Administration, under Award Number DE-NA0003969.
#


#
# imports
#

import glob
from pathlib import Path
import sys
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import collections
import numpy as np

#
# functions
#


def read_file(filepath):
	file = Path(filepath)
	if not file.is_file():
		quit(f"ERR: '{filepath}' is not a file!")

	data = ""

	with open(filepath, "r") as fin:
		data = fin.read()

	return data


def read_file_by_delim(filepath, delim):
	data = read_file(filepath)
	data = data.split(delim)

	if not data[-1]: # python empty strings are false; if empty, throw it away
		data.pop()

	return data


def read_file_by_lines(filepath):
	return read_file_by_delim(filepath, "\n")


def list_files(fileglob):
	return glob.glob(fileglob)


def read_results_file(filepath):
	line_data = read_file_by_lines(filepath)

	expected_line_count_float = 105
	expected_line_count_double = 150
	if len(line_data) != expected_line_count_float and len(line_data) != expected_line_count_double: quit(f"ERR: {filepath} contains unexpected line count!")

	data = [] # list of named tuples
	for line in line_data:
		line = line.split(",")

		total_bits = int(line[0])
		expo_bits = int(line[1])
		mant_bits = int(line[2])

		kernel_1 = float(line[3])
		kernel_2 = float(line[4])
		kernel_3 = float(line[5])

		kernel_4 = -1.0 if len(line) < 8 else float(line[6])
		kernel_5 = -1.0 if len(line) < 8 else float(line[7])

		Runtimes = collections.namedtuple('Runtimes', ['total_bits', 'expo_bits', 'mant_bits', 'k1_rt', 'k2_rt', 'k3_rt', 'k4_rt', 'k5_rt'])
		r = Runtimes(total_bits, expo_bits, mant_bits, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5)

		data.append(r)

	data.sort(key=lambda entry: (entry.total_bits, entry.expo_bits, entry.mant_bits))

	return data


#
# tpb
#

def tpb():

	if len(sys.argv) < 5:
		quit(f"USAGE (Tpb): {sys.argv[0]} speedup FILE_IEEE_STEM FILE_FF_STEM FIG_TITLE")

	FILE_IEEE_STEM = sys.argv[2]
	FILE_FF_STEM = sys.argv[3]
	FIG_TITLE = sys.argv[4]

	tpbs = ['0128', '0256', '0384', '0512', '0640', '0768', '0896', '1024']
	speedup_list = []

	for tpb in tpbs:

		file_ieee = f"{FILE_IEEE_STEM}_{tpb}_TPB.results"
		file_ff = f"{FILE_FF_STEM}_{tpb}_TPB.results"

		ieee_data = read_results_file(file_ieee)
		ff_data = read_results_file(file_ff)

		if len(ieee_data) != len(ff_data): quit("ERR: wrong number of entries in each file!")

		speedup = []
		for ieee_e, ff_e in zip(ieee_data, ff_data):
			if ieee_e.total_bits != ff_e.total_bits: quit("ERR: IEEE entry does not match FF entry!")

			total_bits = ieee_e.total_bits

			k1_speedup = ieee_e.k1_rt / ff_e.k1_rt
			k2_speedup = ieee_e.k2_rt / ff_e.k2_rt
			k3_speedup = ieee_e.k3_rt / ff_e.k3_rt
			k4_speedup = ieee_e.k4_rt / ff_e.k4_rt
			k5_speedup = ieee_e.k5_rt / ff_e.k5_rt

			Speedups = collections.namedtuple('Speedups', ['total_bits', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5'])
			sp = Speedups(total_bits, k1_speedup, k2_speedup, k3_speedup, k4_speedup, k5_speedup)

			speedup.append(sp)

		speedup_list.append(speedup)

	#
	#
	#

	FIG_SIZE_WIDTH = 12
	FIG_SIZE_HEIGHT = 6

	MIN_BITS = 8 if "float" in sys.argv[3] else 31
	MAX_BITS = 32 if "float" in sys.argv[3] else 64
	MIN_SPEEDUP = 0.0
	MAX_SPEEDUP = 2.0

	NUM_KERNELS = 5

	for k in range(1, NUM_KERNELS + 1):

		fig, ax = plt.subplots(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))

		plt.title(FIG_TITLE)
		ax.set_xlabel("Number of Bits")
		ax.set_ylabel("Speedup")

		# get speedups
		bits = [e.total_bits for e in speedup_list[0]]

		for idx in range(0, len(tpbs)):

			kernel_speedup = [e[k] for e in speedup_list[idx]] # k == kernel number

			ax.plot(bits, kernel_speedup, label=f"{int(tpbs[idx])} TPB", marker=".")

		# grid lines
		ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
		ax.set_axisbelow(True)

		# x,y limits, ticks
		plt.xlim(MIN_BITS, MAX_BITS)
		plt.ylim(MIN_SPEEDUP, MAX_SPEEDUP)
		ax.set_xticks(np.linspace(MIN_BITS, MAX_BITS, num=(MAX_BITS - MIN_BITS + 1)))

		ax.legend()


		plt.tight_layout()
		plt.savefig(f"k{k}.{FIG_TITLE.replace(' ', '.')}.pdf")


#
# speedup
#

def speedup():

	if len(sys.argv) < 5:
		quit(f"USAGE (Speedup): {sys.argv[0]} speedup FILE_IEEE FILE_FF FIG_TITLE")


	# read in files

	FILE_IEEE = sys.argv[2]
	FILE_FF = sys.argv[3]
	FIG_TITLE = sys.argv[4]

	ieee_data = read_results_file(FILE_IEEE)
	ff_data = read_results_file(FILE_FF)

	if len(ieee_data) != len(ff_data): quit("ERR: wrong number of entries in each file!")

	# calculate speedup

	speedup = []
	for ieee_e, ff_e in zip(ieee_data, ff_data):
		if ieee_e.total_bits != ff_e.total_bits: quit("ERR: IEEE entry does not match FF entry!")

		total_bits = ieee_e.total_bits

		k1_speedup = ieee_e.k1_rt / ff_e.k1_rt
		k2_speedup = ieee_e.k2_rt / ff_e.k2_rt
		k3_speedup = ieee_e.k3_rt / ff_e.k3_rt
		k4_speedup = ieee_e.k4_rt / ff_e.k4_rt
		k5_speedup = ieee_e.k5_rt / ff_e.k5_rt

		Speedups = collections.namedtuple('Speedups', ['total_bits', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5'])
		sp = Speedups(total_bits, k1_speedup, k2_speedup, k3_speedup, k4_speedup, k5_speedup)

		speedup.append(sp)


	# extract info

	bits = [e.total_bits for e in speedup]

	k1_speedup = [e.sp1 for e in speedup]
	k2_speedup = [e.sp2 for e in speedup]
	k3_speedup = [e.sp3 for e in speedup]
	k4_speedup = [e.sp4 for e in speedup]
	k5_speedup = [e.sp5 for e in speedup]

	#
	#
	#

	FIG_SIZE_WIDTH = 12
	FIG_SIZE_HEIGHT = 6

	MIN_BITS = 8 if "float" in sys.argv[3] else 31
	MAX_BITS = 32 if "float" in sys.argv[3] else 64
	MIN_SPEEDUP = 0.0
	MAX_SPEEDUP = 2.0


	fig, ax = plt.subplots(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))

	plt.title(FIG_TITLE)
	ax.set_xlabel("Number of Bits")
	ax.set_ylabel("Speedup")

	ax.plot(bits, k1_speedup, label='k1', marker=".")
	ax.plot(bits, k2_speedup, label='k2', marker=".")
	ax.plot(bits, k3_speedup, label='k3', marker=".")
	ax.plot(bits, k4_speedup, label='k4', marker=".")
	ax.plot(bits, k5_speedup, label='k5', marker=".")

	# grid lines
	ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
	ax.set_axisbelow(True)

	# x,y limits, ticks
	plt.xlim(MIN_BITS, MAX_BITS)
	plt.ylim(MIN_SPEEDUP, MAX_SPEEDUP)
	ax.set_xticks(np.linspace(MIN_BITS, MAX_BITS, num=(MAX_BITS - MIN_BITS + 1)))

	ax.legend()

	plt.tight_layout()
	plt.savefig(f"{FIG_TITLE.replace(' ', '.')}.pdf")


#
# speedup, multiple machines
#

def speedup_m():

	if len(sys.argv) < 5:
		quit(f"USAGE (Speedup, Machines): {sys.argv[0]} speedup_m NUM_MACHINES FILE_IEEE_FILES FILE_FF_FILES")

	# read in files

	NUM_M = int(sys.argv[2])

	FILE_IEEE_LIST = []
	for idx in range(3, 3 + NUM_M):
		FILE_IEEE_LIST.append(sys.argv[idx])

	FILE_FF_S_LIST = []
	FILE_FF_LIST = []
	for idx in range(3 + NUM_M, len(sys.argv)):
		file_name = sys.argv[idx]
		s_file_name = file_name.split('_')
		s_file_name.insert(2, "s")
		s_file_name = '_'.join(s_file_name)

		FILE_FF_LIST.append(file_name)
		FILE_FF_S_LIST.append(s_file_name)

	FILE_FF_STEMS = [f.split(".")[0] for f in FILE_FF_LIST]

	ieee_data_list = []
	ff_data_list = []
	ff_s_data_list = []
	for idx in range(NUM_M):
		ieee_data = read_results_file(FILE_IEEE_LIST[idx])
		ff_data = read_results_file(FILE_FF_LIST[idx])
		ff_s_data = read_results_file(FILE_FF_S_LIST[idx])

		if len(ieee_data) != len(ff_data): quit("ERR: wrong number of entries in each file!")

		ieee_data_list.append(ieee_data)
		ff_data_list.append(ff_data)
		ff_s_data_list.append(ff_s_data)


	# calculate speedup for each file

	speedup_list = []

	for ieee_data, ff_data, ff_s_data in zip(ieee_data_list, ff_data_list, ff_s_data_list):

		speedup = []
		for ieee_e, ff_e, ff_s_e in zip(ieee_data, ff_data, ff_s_data):
			if ieee_e.total_bits != ff_e.total_bits: quit("ERR: IEEE entry does not match FF entry!")

			total_bits = ieee_e.total_bits

			k1_speedup = ieee_e.k1_rt / ff_e.k1_rt
			k2_speedup = ieee_e.k2_rt / ff_e.k2_rt
			k3_speedup = ieee_e.k3_rt / ff_e.k3_rt
			k4_speedup = ieee_e.k4_rt / ff_e.k4_rt
			k5_speedup = ieee_e.k5_rt / ff_e.k5_rt

			Speedups = collections.namedtuple('Speedups', ['total_bits', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5'])
			sp = Speedups(total_bits, k1_speedup, k2_speedup, k3_speedup, k4_speedup, k5_speedup)

			speedup.append(sp)

		speedup_list.append(speedup)

	#
	#
	#

	FIG_SIZE_WIDTH = 12
	FIG_SIZE_HEIGHT = 6

	MIN_BITS = 8 if "float" in sys.argv[3] else 31
	MAX_BITS = 32 if "float" in sys.argv[3] else 64
	MIN_SPEEDUP = 0.0
	

	NUM_KERNELS = 5

	MACHINES = ['Austin (NVIDIA RTX 3090)', 'Brooks (Radeon Instinct MI100)', 'Bulach']

	for k in range(1, NUM_KERNELS + 1):

		MAX_SPEEDUP = 2.0 if k == 5 else 2.0

		fig, ax = plt.subplots(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))

		FIG_TITLE = f"Speedup Across GPUs Kernel={k}"
		plt.title(FIG_TITLE)
		ax.set_xlabel("Number of Bits")
		ax.set_ylabel("Speedup")

		# get speedups
		bits = [e.total_bits for e in speedup_list[0]]

		for idx in range(0, NUM_M):

			kernel_speedup = [e[k] for e in speedup_list[idx]] # k == kernel number

			ax.plot(bits, kernel_speedup, label=f"{MACHINES[idx]}", marker=".")

		# grid lines
		ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
		ax.set_axisbelow(True)

		# x,y limits, ticks
		plt.xlim(MIN_BITS, MAX_BITS)
		plt.ylim(MIN_SPEEDUP, MAX_SPEEDUP)
		ax.set_xticks(np.linspace(MIN_BITS, MAX_BITS, num=(MAX_BITS - MIN_BITS + 1)))

		ax.legend()


		plt.tight_layout()
		plt.savefig(f"k{k}.{'.'.join(FILE_FF_STEMS)}.svg")


#
# speedup, benchmarks, one expo bit per line
#

def speedup_1e():

	if len(sys.argv) < 4:
		quit(f"USAGE (Speedup): {sys.argv[0]} spb SPEEDUPS_FILE FIG_TITLE")


	# read in files

	FILE = sys.argv[2]
	FIG_TITLE = sys.argv[3]

	data = read_file_by_lines(FILE)

	data_by_exponent = [
		[],
		[],
		[],
		[],
		[],
		[],
		[],
		[],
	]

	for l in data:
		l = l.split(',')

		expo = int(l[0])

		l.pop(0)

		mant = int(l[0])

		l.pop(0)

		speedups = []

		for sp in l:
			speedups.append(float(sp))

		data_by_exponent[expo - 4].append((1 + expo + mant, speedups))

	#
	#
	#

	FIG_SIZE_WIDTH = 5
	FIG_SIZE_HEIGHT = 5

	MIN_BITS = 17 if "float" in FILE else 33
	MAX_BITS = 32 if "float" in FILE else 64
	MIN_SPEEDUP = 0.0
	MAX_SPEEDUP = 2.0


	fig, ax = plt.subplots(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))

	plt.title(FIG_TITLE)
	ax.set_xlabel("Number of Bits")
	ax.set_ylabel("Speedup")

	# plot

	if 'float' in FILE:
		BM_NAMES = ["Accuracy", "Adam", "Aidw", "Attention", "Bilateral", "Bincount", "BScholes", "BSearch", "Car"]
	else:
		BM_NAMES = ["Adv", "Asta", "Burger"]

	BM_COLORS = ['#0000ff', '#00ff00', '#ff0000', '#ffff00', '#00ffff','#ff8900', '#d908ed', '#5100c4', '#ca8bf2']
	EXPO_SHAPES = ['o', 'v', 'h', '8', 's', 'p', 'P', '*']

	for curr_expo in range(len(data_by_exponent)):
		if len(data_by_exponent[curr_expo]) == 0:
			continue

		for curr_bm in range(len(data_by_exponent[curr_expo][0][1])):
			if "bulach" in FILE and BM_NAMES[curr_bm] == "BSearch":
				continue
			bm_x = [e[0] for e in data_by_exponent[curr_expo]]
			bm_y = [e[1][curr_bm] for e in data_by_exponent[curr_expo]]

			ax.plot(bm_x, bm_y, markeredgecolor='black', marker=EXPO_SHAPES[curr_expo], c=BM_COLORS[curr_bm], linestyle='dashed')

	# grid lines
	ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
	ax.set_axisbelow(True)

	# x,y limits, ticks
	plt.xlim(MIN_BITS, MAX_BITS)
	plt.ylim(MIN_SPEEDUP, MAX_SPEEDUP)

	if 'float' in FILE:
		ax.set_xticks(np.linspace(MIN_BITS, MAX_BITS, num=(MAX_BITS - MIN_BITS + 1)))
	else:
		ax.set_xticks([32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64])

	if 'float' in FILE:
		ax.legend(loc="upper right", ncol=2, handles=[
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[0], label=BM_NAMES[0]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[1], label=BM_NAMES[1]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[2], label=BM_NAMES[2]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[3], label=BM_NAMES[3]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[4], label=BM_NAMES[4]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[5], label=BM_NAMES[5]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[6], label=BM_NAMES[6]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[7], label=BM_NAMES[7]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[8], label=BM_NAMES[8]),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[0], label='4 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[1], label='5 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[2], label='6 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[3], label='7 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[4], label='8 Expo. Bits'),
			])
	else:
		ax.legend(loc="upper right", ncol=2, handles=[
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[0], label=BM_NAMES[0]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[1], label=BM_NAMES[1]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[2], label=BM_NAMES[2]),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[0], label=' 4 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[1], label=' 5 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[2], label=' 6 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[3], label=' 7 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[4], label=' 8 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[5], label=' 9 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[6], label='10 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[7], label='11 Expo. Bits'),
			])

	plt.tight_layout()
	plt.savefig(f"{FIG_TITLE.replace(' ', '.')}.pdf")

def speedup_1e_us():

	if len(sys.argv) < 4:
		quit(f"USAGE (Speedup): {sys.argv[0]} spb SPEEDUPS_FILE FIG_TITLE")


	# read in files

	FILE = sys.argv[2]
	FIG_TITLE = sys.argv[3]

	data = read_file_by_lines(FILE)

	data_by_exponent = [
		[],
		[],
		[],
		[],
		[],
		[],
		[],
		[],
	]

	for l in data:

		l = l.split(',')

		expo = int(l[0])

		l.pop(0)

		mant = int(l[0])

		l.pop(0)

		speedups = []

		for sp in l:
			speedups.append(float(sp))

		data_by_exponent[expo - 4].append((1 + expo + mant, speedups))

	#
	#
	#

	FIG_SIZE_WIDTH = 5
	FIG_SIZE_HEIGHT = 5

	MIN_BITS = 17 if "float" in FILE else 33
	MAX_BITS = 32 if "float" in FILE else 64
	MIN_SPEEDUP = 0.0
	MAX_SPEEDUP = 2.0


	fig, ax = plt.subplots(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))

	plt.title(FIG_TITLE)
	ax.set_xlabel("Number of Bits")
	ax.set_ylabel("Speedup")

	# plot

	BM_NAMES = ["Init. General", "Init. Special", "Add", "Count", "HtD Memcpy", "DtH Memcpy"]

	BM_COLORS = ['#0000ff', '#00ff00', '#ff0000', '#ffff00', '#00ffff','#ff8900', '#d908ed', '#5100c4', '#ca8bf2']
	EXPO_SHAPES = ['o', 'v', 'h', '8', 's', 'p', 'P', '*']

	for curr_expo in range(len(data_by_exponent)):
		if len(data_by_exponent[curr_expo]) == 0:
			continue

		for curr_bm in range(len(data_by_exponent[curr_expo][0][1])):
			if "Memcpy" in BM_NAMES[curr_bm]:
				continue
			bm_x = [e[0] for e in data_by_exponent[curr_expo]]
			bm_y = [e[1][curr_bm] for e in data_by_exponent[curr_expo]]

			ax.plot(bm_x, bm_y, markeredgecolor='black', marker=EXPO_SHAPES[curr_expo], c=BM_COLORS[curr_bm], linestyle='dashed')

	# grid lines
	ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
	ax.set_axisbelow(True)

	# x,y limits, ticks
	plt.xlim(MIN_BITS, MAX_BITS)
	plt.ylim(MIN_SPEEDUP, MAX_SPEEDUP)

	if 'float' in FILE:
		ax.set_xticks(np.linspace(MIN_BITS, MAX_BITS, num=(MAX_BITS - MIN_BITS + 1)))
	else:
		ax.set_xticks([32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64])

	if 'float' in FILE:
		ax.legend(loc="upper right", ncol=2, handles=[
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[0], label=BM_NAMES[0]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[1], label=BM_NAMES[1]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[2], label=BM_NAMES[2]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[3], label=BM_NAMES[3]),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[0], label='4 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[1], label='5 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[2], label='6 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[3], label='7 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[4], label='8 Expo. Bits'),
			])
	else:
		ax.legend(loc="upper right", ncol=2, handles=[
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[0], label=BM_NAMES[0]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[1], label=BM_NAMES[1]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[2], label=BM_NAMES[2]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[3], label=BM_NAMES[3]),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[0], label=' 4 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[1], label=' 5 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[2], label=' 6 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[3], label=' 7 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[4], label=' 8 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[5], label=' 9 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[6], label='10 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[7], label='11 Expo. Bits'),
			])

	plt.tight_layout()
	plt.savefig(f"{FIG_TITLE.replace(' ', '.')}.ours.pdf")


def speedup_1e_memcpy():

	if len(sys.argv) < 5:
		quit(f"USAGE (Speedup): {sys.argv[0]} spb SPEEDUPS_FILE_1 SPEEDUPS_FILE_2 FIG_TITLE")


	# read in files

	FILE = sys.argv[2]
	FIG_TITLE = sys.argv[4]

	data = read_file_by_lines(FILE)

	data_by_exponent = [
		[],
		[],
		[],
		[],
		[],
		[],
		[],
		[],
	]

	for l in data:

		l = l.split(',')

		expo = int(l[0])

		l.pop(0)

		mant = int(l[0])

		l.pop(0)

		speedups = []

		for sp in l:
			speedups.append(float(sp))

		data_by_exponent[expo - 4].append((1 + expo + mant, speedups))

	###

	FILE2 = sys.argv[3]

	data = read_file_by_lines(FILE2)

	data_by_exponent2 = [
		[],
		[],
		[],
		[],
		[],
		[],
		[],
		[],
	]

	for l in data:

		l = l.split(',')

		expo = int(l[0])

		l.pop(0)

		mant = int(l[0])

		l.pop(0)

		speedups = []

		for sp in l:
			speedups.append(float(sp))

		data_by_exponent2[expo - 4].append((1 + expo + mant, speedups))

	#
	#
	#

	FIG_SIZE_WIDTH = 5
	FIG_SIZE_HEIGHT = 5

	MIN_BITS = 17 if "float" in FILE else 33
	MAX_BITS = 32 if "float" in FILE else 64
	MIN_SPEEDUP = 0.0
	MAX_SPEEDUP = 2.0


	fig, ax = plt.subplots(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))

	plt.title(FIG_TITLE)
	ax.set_xlabel("Number of Bits")
	ax.set_ylabel("Speedup")

	# plot

	BM_NAMES = ["Init. General", "Init. Special", "H2D Memcpy - 4090", "D2H Memcpy - 4090", "H2D Memcpy - A100", "D2H Memcpy - A100"]

	BM_COLORS = ['#0000ff', '#00ff00', '#ff0000', '#ffff00', '#00ffff','#ff8900', '#d908ed', '#5100c4', '#ca8bf2']
	EXPO_SHAPES = ['o', 'v', 'h', '8', 's', 'p', 'P', '*']

	for curr_expo in range(len(data_by_exponent)):
		if len(data_by_exponent[curr_expo]) == 0:
			continue

		for curr_bm in [5, 4]:
			bm_x = [e[0] for e in data_by_exponent[curr_expo]]
			bm_y = [e[1][curr_bm] for e in data_by_exponent[curr_expo]]

			ax.plot(bm_x, bm_y, markeredgecolor='black', marker=EXPO_SHAPES[curr_expo], c=BM_COLORS[curr_bm - 2], linestyle='dashed')

			bm_x = [e[0] for e in data_by_exponent2[curr_expo]]
			bm_y = [e[1][curr_bm] for e in data_by_exponent2[curr_expo]]

			if FILE == FILE2:
				continue

			ax.plot(bm_x, bm_y, markeredgecolor='black', marker=EXPO_SHAPES[curr_expo], c=BM_COLORS[curr_bm], linestyle='dashed')

	# grid lines
	ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
	ax.set_axisbelow(True)

	# x,y limits, ticks
	plt.xlim(MIN_BITS, MAX_BITS)
	plt.ylim(MIN_SPEEDUP, MAX_SPEEDUP)

	if 'float' in FILE:
		ax.set_xticks(np.linspace(MIN_BITS, MAX_BITS, num=(MAX_BITS - MIN_BITS + 1)))
	else:
		ax.set_xticks([32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64])

	if 'float' in FILE:
		ax.legend(loc="upper right", ncol=2, handles=[
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[2], label=BM_NAMES[2]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[3], label=BM_NAMES[3]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[4], label=BM_NAMES[4]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[5], label=BM_NAMES[5]),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[0], label='4 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[1], label='5 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[2], label='6 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[3], label='7 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[4], label='8 Expo. Bits'),
			])
	else:
		ax.legend(loc="upper right", ncol=2, handles=[
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[2], label=BM_NAMES[2]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[3], label=BM_NAMES[3]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[4], label=BM_NAMES[4]),
				mpatches.Patch(edgecolor='black', facecolor=BM_COLORS[5], label=BM_NAMES[5]),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[0], label=' 4 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[1], label=' 5 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[2], label=' 6 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[3], label=' 7 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[4], label=' 8 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[5], label=' 9 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[6], label='10 Expo. Bits'),
				lines.Line2D([], [], color='black', markeredgecolor='black', linestyle='', marker=EXPO_SHAPES[7], label='11 Expo. Bits'),
			])

	plt.tight_layout()
	plt.savefig(f"{FIG_TITLE.replace(' ', '.')}.ours.pdf")


#
# main
#

if len(sys.argv) < 2:
	quit(f"USAGE: {sys.argv[0]} speedup|speedup_m|tpb|spb|spbus|spbmc")

option = sys.argv[1]

if option == 'speedup':
	speedup()
elif option == 'speedup_m':
	speedup_m()
elif option == 'tpb':
	tpb()
elif option == 'spb':
	speedup_1e()
elif option == 'spbus':
	speedup_1e_us()
elif option == 'spbmc':
	speedup_1e_memcpy()
else:
	quit("ERR: invalid menu option!")
