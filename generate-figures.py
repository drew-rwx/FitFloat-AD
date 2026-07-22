#!/usr/bin/env python3

# This file is part of FitFloat, a drop-in floating-point array replacement supporting user-specified precision on GPUs with the goal of reducing storage requirements.
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Andrew Rodriguez, and Martin Burtscher
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
# URL: The latest version of this code is available at https://github.com/burtscher/FitFloat/.
#
# Publication: This work is described in detail in the following paper.
# Andrew Rodriguez, and Martin Burtscher. "FitFloat: Read/Write Random-Access Compressed Floating-Point Arrays for GPUs"
#
# Sponsor: This material is based upon work supported by the U.S. National Science Foundation under Grant Number 2403380 and by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under Award Number DE-SC0022223.

#
# imports
#


import glob
from pathlib import Path
import sys
import statistics
import matplotlib.pyplot as plt


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


def list_files(fileglob):
	return glob.glob(fileglob)


def parse_runtimes(files, runtime_dict):
	for file in files:

		bits = file.removesuffix(".results")
		bits = bits.split("..")
		bits = bits[-1]

		expo = int(bits[:2])
		mant = int(bits[2:])
		bits = expo	+ mant + 1

		data = read_file(file).splitlines()
		data = [d for d in data if d.startswith('~')]
		datat = list(enumerate(data))
		data = [float(d.split()[-2]) for d in data]

		runtimes = list()

		if DATATYPE == "float":
			runtimes.append(data[0]) # accuracy
			runtimes.append(data[1]) # adam
			runtimes.append(data[2]) # aidw
			runtimes.append(data[3]) # attention
			runtimes.append(statistics.geometric_mean(data[4:7])) # bilateral
			runtimes.append(statistics.geometric_mean(data[7:12])) # bincount
			runtimes.append(data[12]) # bscholes
			if expo >= 6:
				runtimes.append(statistics.geometric_mean(data[13:17])) # bsearch
			runtimes.append(data[17]) # car
			runtimes.append(data[18]) # chi2
			runtimes.append(data[19]) # fhd
			runtimes.append(data[20]) # adam OPT

		if DATATYPE == "double":
			runtimes.append(data[0]) # adv
			runtimes.append(data[1]) # asta
			runtimes.append(data[2]) # burger

		if bits not in runtime_dict:
			runtime_dict[bits] = list()

		runtime_dict[bits].append(runtimes)


def get_geomean_speedups(uvm_speedups, uvm_fitf_runtimes, uvm_ieee_runtimes):
	if len(uvm_fitf_runtimes.keys()) != len(uvm_ieee_runtimes.keys()):
		quit(f"ERROR: FitFloat runtimes do not align with IEEE runtimes!")

	bits_in_use = list(uvm_fitf_runtimes.keys())
	bits_in_use.sort()

	for bits in bits_in_use:
		fitf_runtimes_list = uvm_fitf_runtimes[bits]
		ieee_runtimes_list = uvm_ieee_runtimes[bits]

		if len(fitf_runtimes_list) != len(ieee_runtimes_list):
			quit(f"ERROR: FitFloat runtimes do not align with IEEE runtimes for bit count {bits}!")

		for fitf_runtimes, ieee_runtimes in zip(fitf_runtimes_list, ieee_runtimes_list):
			if len(fitf_runtimes) != len(ieee_runtimes):
				quit(f"ERROR: FitFloat runtimes do not align with IEEE runtimes for bit count {bits}!")

			speedups_per_benchmark = list()

			for fitf, ieee in zip(fitf_runtimes, ieee_runtimes):
				speedups_per_benchmark.append(ieee / fitf)

			if bits not in uvm_speedups:
				uvm_speedups[bits] = list()

			uvm_speedups[bits].append(statistics.geometric_mean(speedups_per_benchmark))

	for bits, bits_list in uvm_speedups.items():
		if len(bits_list) == 1:
			sp = bits_list[0]
		else:
			sp = statistics.geometric_mean(bits_list)

		uvm_speedups[bits] = sp


def print_geomean_speedups(uvm_speedups, title):
	print(f"*** {title} ***")
	print("FitFloat Bits: Speedup")
	speedups = list(uvm_speedups.items())
	speedups.reverse()
	for bits, sp in speedups:
		print(f"{bits}: {sp:.2f}x")
	print()


def generate_geomean_figure(geomean_speedups, filename):
	FIG_SIZE_WIDTH = 5
	FIG_SIZE_HEIGHT = 5

	fig, ax = plt.subplots(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))

	ax.set_ylabel("Number of Bits")
	ax.set_xlabel("Speedup")

	for bits, sp in geomean_speedups.items():
		ax.barh(bits, sp, color="red", edgecolor="black", height=0.45)

	ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
	ax.set_axisbelow(True)

	if DATATYPE == "float":
		plt.ylim(16, 33)
	else:
		plt.ylim(32, 65)

	plt.tight_layout()
	plt.savefig(filename)
	# plt.show()


#
# CLI check
#


if len(sys.argv) != 2:
	quit(f"USAGE: {sys.argv[0]} float|double")

DATATYPE = sys.argv[1]

if DATATYPE != "float" and DATATYPE != "double":
	quit(f"ERROR: must specify float or double, you provided: {DATATYPE}")


#
# Parse results
#


# UVM1, FitFloat fits

files = list_files(f"./results/uvm1/FF..{DATATYPE}..*")
files.sort(reverse=True)
uvm1_fitf_runtimes = dict()
parse_runtimes(files, uvm1_fitf_runtimes)

files = list_files(f"./results/uvm1/IEEE..{DATATYPE}..*")
files.sort(reverse=True)
uvm1_ieee_runtimes = dict()
parse_runtimes(files, uvm1_ieee_runtimes)

# UVM2

files = list_files(f"./results/uvm2/FF..{DATATYPE}..*")
files.sort(reverse=True)
uvm2_fitf_runtimes = dict()
parse_runtimes(files, uvm2_fitf_runtimes)

files = list_files(f"./results/uvm2/IEEE..{DATATYPE}..*")
files.sort(reverse=True)
uvm2_ieee_runtimes = dict()
parse_runtimes(files, uvm2_ieee_runtimes)


#
# Speedups
#

# UVM1, FitFloat fits

uvm1_speedups = dict()
get_geomean_speedups(uvm1_speedups, uvm1_fitf_runtimes, uvm1_ieee_runtimes)
print_geomean_speedups(uvm1_speedups, "Geometric mean speedups, FitFloat arrays fit in global memory")

# UVM2

uvm2_speedups = dict()
get_geomean_speedups(uvm2_speedups, uvm2_fitf_runtimes, uvm2_ieee_runtimes)
print_geomean_speedups(uvm2_speedups, "Geometric mean speedups, Native and FitFloat arrays do not fit in global memory")

#
# Figures
#

generate_geomean_figure(uvm1_speedups, f"./figures/{DATATYPE.capitalize()}.UVM1.Geomean.pdf")

generate_geomean_figure(uvm2_speedups, f"./figures/{DATATYPE.capitalize()}.UVM2.Geomean.pdf")
