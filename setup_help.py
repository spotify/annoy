import os
import subprocess


def _get_AVX_macro_list():
    cmd = """gcc -march=native -dM -E - < /dev/null | grep "__AVX512" | sort"""
    #os.system(cmd)
    exec_stdoutput = subprocess.check_output(cmd, shell=True)

    AVX_macro_list = []
    for i,line in enumerate(exec_stdoutput.decode().split(os.linesep)):
        term_list = line.split(" ")
        if term_list[0] == "#define":
            AVX_macro_list.append(term_list[1])

    return AVX_macro_list


def get_unuse_AVX512_options():
    args = []
    AVX_macro_list = _get_AVX_macro_list()
    for m in AVX_macro_list:
        option = "-U" + m
        args.append(option)
    return args



