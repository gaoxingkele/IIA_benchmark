"""Reviewed cold-start CUDA initialization; frozen GRASP implementation unchanged."""
from scripts.flow_matching import grasp_protocol_runner as original


def initialize_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA unavailable; no CPU/budget fallback')
    # is_available() does not create the allocator. Explicit-device peak reset
    # in the preserved runner requires initialization on a cold worker process.
    torch.cuda.init()
    torch.cuda.set_device(0)


def main():
    initialize_cuda()
    original.main()


if __name__ == '__main__':
    main()
