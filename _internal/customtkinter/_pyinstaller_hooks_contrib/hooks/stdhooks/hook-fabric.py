# ------------------------------------------------------------------
# Copyright (c) 2022 PyInstaller Development Team.
#
# This file is distributed under the terms of the GNU General Public
# License (version 2.0 or later).
#
# The full license is available in LICENSE, distributed with
# this software.
#
# SPDX-License-Identifier: GPL-2.0-or-later
# ------------------------------------------------------------------
#
# Fabric is a high level Python (2.7, 3.4+) library designed to execute shell commands remotely over SSH,
# yielding useful Python objects in return
#
# https://docs.fabfile.org/en/latest
#
# Tested with fabric 2.6.0

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files('fabric')
