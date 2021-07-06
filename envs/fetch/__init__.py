from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .custom import CustomGoalEnv

# TODO: change this file for new env handling!
def make_env(args):
	return {
		'vanilla': VanillaGoalEnv,
		'fixobj': FixedObjectGoalEnv,
		'interval': IntervalGoalEnv,
		'custom': CustomGoalEnv,
	}[args.goal](args)