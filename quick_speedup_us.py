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


IEEE_FILE = sys.argv[1]
FF_FILE = sys.argv[2]
FF_S_FILE = sys.argv[3]

ieee_data = read_file_by_lines(IEEE_FILE)
ff_data = read_file_by_lines(FF_FILE)
ff_s_data = read_file_by_lines(FF_S_FILE)

for ieee_l, ff_l, ff_s_l in zip(ieee_data, ff_data, ff_s_data):

	d1 = ieee_l.split(',')
	d2 = ff_l.split(',')
	d3 = ff_s_l.split(',')

	d1.pop(-1)
	d2.pop(-1)
	d3.pop(-1)

	bits1 = int(d1[0])
	bits2 = int(d2[0])
	bits3 = int(d3[0])

	if bits1 != bits2 or bits2 != bits3 or len(d1) != len(d2): quit("Lines do not match!")

	expo1 = int(d1[1])
	mant1 = int(d1[2])

	ieee_runtimes = []
	for idx in range(3, len(d1)):
		ieee_runtimes.append(float(d1[idx]))

	ff_runtimes = []
	for idx in range(3, len(d2)):
		ff_runtimes.append(float(d2[idx]))

	if len(ieee_runtimes) != len(ff_runtimes): quit("Runtimes do not match!")

	ff_s_runtimes = []
	for idx in range(3, len(d3)):
		ff_s_runtimes.append(float(d3[idx]))

	speedups = []

	for r1, r2 in zip(ieee_runtimes, ff_runtimes):
		if r1 == 0.0 or r2 == 0.0:
			speedups.append(-1.0)
		else:
			speedups.append(r1 / r2)

	for r1, r2 in zip(ieee_runtimes, ff_s_runtimes):
		if r1 == 0.0 or r2 == 0.0:
			speedups.append(-1.0)
		else:
			speedups.append(r1 / r2)

	# csv print

	print(f"{expo1},{mant1}", end='')

	print(f",{round(speedups[0], 3)}", end='') # init gen
	print(f",{round(speedups[6], 3)}", end='') # init spec
	print(f",{round(speedups[2], 3)}", end='') # add
	print(f",{round(speedups[4], 3)}", end='') # count
	print(f",{round(speedups[3], 3)}", end='') # memcpy ->
	print(f",{round(speedups[5], 3)}", end='') # memcpy <-
	print()
