"""Preserve the frozen controllers' complete subprocess API during CPU binding.

Version2 was inspected before launch and omitted STDOUT from its subprocess
namespace. Keep that frozen draft and bind the complete API in this version.
"""
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import run_anomaly_transformer_explicit_cpu_binding_v2 as previous

ROOT,read,sha,write=previous.ROOT,previous.read,previous.sha,previous.write
author,restored=previous.author,previous.restored
subprocess=previous.subprocess
bound_popen,verify=previous.bound_popen,previous.verify


def full_subprocess_namespace(**overrides):
    return SimpleNamespace(**dict(vars(subprocess),**overrides))


def main():
    FunctionType(previous.main.__code__,dict(vars(previous),SimpleNamespace=full_subprocess_namespace))()


if __name__=='__main__':main()
