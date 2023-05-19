#!/usr/bin/env python3
import os
import sys
import fileinput

# Read input arguments
# 1. relative filepath to old kernel config file
# 2. relative filepath to new generated config file
# 3. GPU architecture number (ie. 906)
old_config_file_path = sys.argv[1]
new_config_file_path = sys.argv[2]
gpu_arch = sys.argv[3]

old_config_file = open(old_config_file_path, 'r')
new_config_file = open(new_config_file_path, 'r')
new_configs = new_config_file.read()

gpu_config_label_start = "/*********************************BEGIN gfx" + gpu_arch + " CONFIG**************************/"
gpu_config_label_end = "/*********************************END gfx" + gpu_arch + " CONFIG**************************/"

copy_line = True
contents_to_copy = ""

# Copy old config contents, except for target GPU arch.  Insert new configs for GPU arch, then write to old file
for num, line in enumerate(old_config_file, 1):
    #print line
    if copy_line:
        contents_to_copy += line
    if gpu_config_label_start in line:
        copy_line = False
        contents_to_copy += new_configs
        contents_to_copy += "\n"
    elif gpu_config_label_end in line:
        contents_to_copy += line
        copy_line = True

old_config_file = open(old_config_file_path, 'w')
old_config_file.write(contents_to_copy)

