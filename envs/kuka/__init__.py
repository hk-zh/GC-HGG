from .kuka_env import KukaEnvWrapper



def make_env(args):
    """
    Utility function for vectorized env.

    :param env_cls: (str) the environment class to instantiate
    :param seed: (int) the inital seed for RNG
    :param **env_options: additional arguments to pass to the environment
    """

    env = KukaEnvWrapper(args)
    return env
