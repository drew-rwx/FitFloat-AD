#!/usr/bin/env python3


#
# imports
#


import glob
from pathlib import Path
import sys
import statistics


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

		# print(*datat, sep="\n")
		# print()

		# print(*data, sep="\n")

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

		# print()
		# print(bits)
		# print(*runtimes, sep="\n")


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

			# print(bits, *speedups_per_benchmark, sep="\n", end="\n\n")

			if bits not in uvm_speedups:
				uvm_speedups[bits] = list()

			uvm_speedups[bits].append(statistics.geometric_mean(speedups_per_benchmark))

	for bits, bits_list in uvm_speedups.items():
		if len(bits_list) == 1:
			sp = bits_list[0]
		else:
			sp = statistics.geometric_mean(bits_list)

		uvm_speedups[bits] = sp


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

print(*list(uvm1_speedups.items()), sep="\n", end="\n\n")

# UVM2

uvm2_speedups = dict()
get_geomean_speedups(uvm2_speedups, uvm2_fitf_runtimes, uvm2_ieee_runtimes)

print(*list(uvm2_speedups.items()), sep="\n", end="\n\n")
