"""Supply the diagnostic AST namespace explicitly; preserve both failed attempts."""
import ast
from scripts.flow_matching import audit_rcgan_mmd_native_v1 as base
from scripts.flow_matching import audit_rcgan_mmd_native_v2 as diagnostic


def main():
    base.ast = ast
    diagnostic.main()


if __name__ == '__main__':
    main()
