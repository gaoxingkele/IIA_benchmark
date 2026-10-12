"""Cold full400 seed1105 producer using the unchanged complete-state worker."""
from scripts.flow_matching.run_cfm_checkpointed_light_parent_v2 import lightweight_namespace


def parent_namespace():
    return lightweight_namespace()


if __name__ == '__main__':
    parent_namespace()['main']()
