#!/usr/bin/env sh
##
##  libBICOS: binary correspondence search on multishot stereo imagery
##  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU Lesser General Public License as
##  published by the Free Software Foundation, either version 3 of the
##  License, or (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU Lesser General Public License for more details.
##
##  You should have received a copy of the GNU Lesser General Public License
##  along with this program.  If not, see <https://www.gnu.org/licenses/>.
##

DATAURL=https://robotik.informatik.uni-wuerzburg.de/telematics/bicos/bicos-data.tar.xz

if [ ! -d .git ]
then
	echo "This is supposed to be called from the project root of libBICOS" 1>&2
	exit 1
fi

TARBALL=$(basename $DATAURL)

if [ ! -f $TARBALL ]
then
	if command -v curl > /dev/null; then
		DLTOOL="curl -O"
	elif command -v wget > /dev/null; then
		DLTOOL=wget
	else
		echo "Neither curl nor wget seem to be available" 1>&2
		exit 1
	fi

	eval "$DLTOOL $DATAURL"
fi

if ! sha256sum --check data/sha256sums.txt
then
	echo "Dataset checksum changed!" 1>&2
	exit 1 
fi

tar -xf $TARBALL -C data && rm $TARBALL

