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


FILE_STEM = sys.argv[1]

files = list_files(FILE_STEM)
files.sort(reverse=True)

for FILE in files:

	data = read_file(FILE)

	data = data.split("!!!!!")

	runtimes = []

	for d in data:

		d = d.splitlines()

		d = [l for l in d if l.startswith("median")]

		for r in d:
			r = r.split()
			r = r[-2]
			r = float(r)

			runtimes.append(r)

	bits = FILE.removesuffix(".results")
	bits = bits.split("..")
	bits = bits[-1]

	expo = int(bits[:2])
	mant = int(bits[2:])
	bits = expo	+ mant + 1

	print(f"{bits},{expo},{mant},", end='')
	for rt in runtimes:
		print(f"{rt},", end='')
	print()
