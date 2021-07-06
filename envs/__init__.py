import gym
import envs.fetch as fetch_env
import envs.hand as hand_env
import envs.kuka as kuka_env
from .utils import goal_distance, goal_distance_obs

Robotics_envs_id = [
    'FetchReach-v1',
    'FetchPush-v1',
    'FetchSlide-v1',
    'FetchPickAndPlace-v1',
    'FetchPushNew-v1',
    'FetchCurling-v1',
    'FetchPushObstacle-v1',
    'FetchPickObstacle-v1',
    'FetchPushNoObstacle-v1',
    'FetchPickNoObstacle-v1',
    'FetchPushLabyrinth-v1',
    'FetchPickAndThrow-v1',
    'FetchPickAndSort-v1',
    'HandManipulateBlock-v0',
    'HandManipulateEgg-v0',
    'HandManipulatePen-v0',
    'HandReach-v0'
]
Kuka_envs_id = [
    'KukaReach-v1',
    'KukaPickAndPlaceObstacle-v1',
    'KukaPickAndPlaceObstacle-v2',
    'KukaPickNoObstacle-v1',
    'KukaPickNoObstacle-v2',
    'KukaPickThrow-v1',
    'KukaPushLabyrinth-v1',
    'KukaPushLabyrinth-v2',
    'KukaPushSlide-v1',
    'KukaPushNew-v1'
]


def make_env(args):
    assert (args.env in Robotics_envs_id or args.env in Kuka_envs_id)
    if args.env[:4] == 'Kuka':
        return kuka_env.make_env(args)
    elif args.env[:5] == 'Fetch':
        return fetch_env.make_env(args)
    else:
        return hand_env.make_env(args)



def clip_return_range(args):
    gamma_sum = 1.0 / (1.0 - args.gamma)
    return {
        'FetchReach-v1': (-gamma_sum, 0.0),
        'FetchPush-v1': (-gamma_sum, 0.0),
        'FetchPushNew-v1': (-gamma_sum, 0.0),
        'FetchSlide-v1': (-gamma_sum, 0.0),
        'FetchPickAndPlace-v1': (-gamma_sum, 0.0),
        'FetchPickObstacle-v1': (-gamma_sum, 0.0),
        'FetchPickNoObstacle-v1': (-gamma_sum, 0.0),
        'FetchPushLabyrinth-v1': (-gamma_sum, 0.0),
        'FetchPickAndThrow-v1': (-gamma_sum, 0.0),
        'FetchPickAndSort-v1': (-gamma_sum, 0.0),
        'HandManipulateBlock-v0': (-gamma_sum, 0.0),
        'HandManipulateEgg-v0': (-gamma_sum, 0.0),
        'HandManipulatePen-v0': (-gamma_sum, 0.0),
        'HandReach-v0': (-gamma_sum, 0.0),
        'KukaReach-v1': (-gamma_sum, 0.0),
        'KukaPickAndPlaceObstacle-v1': (-gamma_sum, 0.0),
        'KukaPickAndPlaceObstacle-v2': (-gamma_sum, 0.0),
        'KukaPickNoObstacle-v1': (-gamma_sum, 0.0),
        'KukaPickNoObstacle-v2': (-gamma_sum, 0.0),
        'KukaPickThrow-v1': (-gamma_sum, 0.0),
        'KukaPushLabyrinth-v1': (-gamma_sum, 0.0),
        'KukaPushLabyrinth-v2': (-gamma_sum, 0.0),
        'KukaPushSlide-v1': (-gamma_sum, 0.0),
        'KukaPushNew-v1': (-gamma_sum, 0.0)
    }[args.env]
