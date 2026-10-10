"""Restore caller cwd after the original full loader/graph audit, including errors."""
import os

from scripts.flow_matching import audit_adopted_gan_native_v1 as base
from scripts.flow_matching import audit_rcgan_mnist_native_v1 as native


def graph_audit(asset, case, output):
    previous = os.getcwd()
    try:
        return native.graph_audit(asset, case, output)
    finally:
        os.chdir(previous)


def main():
    base.graph_audit = graph_audit
    base.main()


if __name__ == '__main__':
    main()
