# Curriculum-guided Hindsight Goal Generation under Kuka-Environment

It is based on the implementation of the HGG paper [Exploration via Hindsight Goal Generation](http://arxiv.org/abs/1906.04279) accepted by NeurIPS 2019.



## Requirements
1. Ubuntu 16.04 or macOS Catalina 10.15.7 (newer versions also work well) 
2. Python 3.5.2 (newer versions such as 3.6.8 should work as well, 3.8 or higher is not suggested)
3. MuJoCo == 2.00 (see instructions on https://github.com/openai/mujoco-py)
4. Install gym from https://github.com/Hongkuan-Zhou/gym.git. Certain environment specifications and parameters are set there. 


```bash
git clone https://github.com/Hongkuan-Zhou/gym.git
cd gym
pip install -e . 
```

6. Install requirements with pip install -r requirements.txt
```bash
pip install -r requirements.txt
```

7. Trained policies can be downloaded here: 
https://syncandshare.lrz.de/getlink/fiNF8o2gZeushcLZEcjxMVVu/policies.zip
  
8. Videos about Kuka Environments can be found here:
https://sites.google.com/view/kuka-environment/

## New Kuka Environments
![image](https://github.com/Hongkuan-Zhou/C-HGG/blob/main/Image/KukaReach.png)
![image](https://github.com/Hongkuan-Zhou/C-HGG/blob/main/Image/KukaPush.png)
![image](https://github.com/Hongkuan-Zhou/C-HGG/blob/main/Image/KukaPickNoObstacle.png)
![image](https://github.com/Hongkuan-Zhou/C-HGG/blob/main/Image/KukaPickObstacle.png)
## Training under different environments

The following commands are used to train the agent in different environments with HGG, HER, G-HGG, C-HGG.
Note that new Kuka Environments are introduced.

### Fetch Environments
```bash
## FetchPushLabyrinth

# HER (with EBP)
python train.py --tag 000 --learn normal --env FetchPushLabyrinth-v1 --goal custom 
# HGG (with HER, EBP and STOP condition)
python train.py --tag 010 --learn hgg --env FetchPushLabyrinth-v1 --goal custom --stop_hgg_threshold 0.3
# GC-HGG
python train.py --tag 020 --learn hgg --env FetchPushLabyrinth-v1 --goal custom --stop_hgg_threshold 0.3 --graph True --n_x 31 --n_y 31 --n_z 11 --curriculum True
# HER+GoalGAN
python train.py --tag 030 --learn normal+goalGAN --env FetchPushLabyrinth-v1 --goal custom
# C-HER
python train.py --tag 040 --learn normal --env FetchPushLabyrinth-v1 --goal custom --curriculum True --batch_size 64 --buffer_size 500


## FetchPickObstacle

python train.py --tag 100 --learn normal --env FetchPickObstacle-v1 --goal custom 
python train.py --tag 110 --learn hgg --env FetchPickObstacle-v1 --goal custom --stop_hgg_threshold 0.3
python train.py --tag 120 --learn hgg --env FetchPickObstacle-v1 --goal custom --graph True --n_x 31 --n_y 31 --n_z 11 --stop_hgg_threshold 0.3 --curriculum True
python train.py --tag 140 --learn normal+goalGAN --env FetchPickObstacle-v1 --goal custom
# hgg + route
python train.py --tag 111 --learn hgg --env FetchPickObstacle-v1 --goal custom --stop_hgg_threshold 0.5 --route True 
# C-HER
python train.py --tag 150 --learn normal --env FetchPickObstacle-v1 --goal custom --curriculum True --batch_size 64 --buffer_size 500

## FetchPickNoObstacle
python train.py --tag 200 --learn normal --env FetchPickNoObstacle-v1 --goal custom 
python train.py --tag 210 --learn hgg --env FetchPickNoObstacle-v1 --goal custom --stop_hgg_threshold 0.3
python train.py --tag 220 --learn hgg --env FetchPickNoObstacle-v1 --goal custom --graph True --n_x 31 --n_y 31 --n_z 11 --stop_hgg_threshold 0.3 --curriculum True
python train.py --tag 240 --learn normal+goalGAN --env FetchPickNoObstacle-v1 --goal custom
# C-HER
python train.py --tag 250 --learn normal --env FetchPickNoObstacle-v1 --goal custom --curriculum True --batch_size 64 --buffer_size 500

## FetchPickAndThrow
python train.py --tag 300 --learn normal --env FetchPickAndThrow-v1 --goal custom 
python train.py --tag 310 --learn hgg --env FetchPickAndThrow-v1 --goal custom --stop_hgg_threshold 0.9
python train.py --tag 320 --learn hgg --env FetchPickAndThrow-v1 --goal custom --graph True --n_x 51 --n_y 51 --n_z 7 --stop_hgg_threshold 0.9 --curriculum True
python train.py --tag 340 --learn normal+goalGAN --env FetchPickAndThrow-v1 --goal custom

## FetchPush
python train.py --tag 1010 --goal custom --learn hgg --env FetchPush-new-v1 --stop_hgg_threshold 0.3 --epoch 20

# FetchReach
python train.py --tag 1310 --goal custom --learn hgg --env FetchReach-v1 --stop_hgg_threshold 0.3 --epoch 10
# C-HER
python train.py --tag 1320 --goal custom --learn normal --env FetchReach-v1 --curriculum True --batch_size 64 --buffer_size 500 --epoch 10


```
### Kuka Environments
```bash
## KukaReach
python train.py --tag 400 --learn normal --env KukaReach-v1 
python train.py --tag 410 --learn hgg --env KukaReach-v1 --stop_hgg_threshold 0.3
#CHER
python train.py --tag 450 --learn normal --env KukaReach-v1 --curriculum True --batch_size 64 --buffer_size 500 --epoch 10

## KukaPickAndPlaceObstacle
python train.py --tag 510 --learn hgg --env KukaPickAndPlaceObstacle-v1 --stop_hgg_threshold 0.3
python train.py --tag 520 --learn hgg --env KukaPickAndPlaceObstacle-v1 --graph True --n_x 31 --n_y 31 --n_z 15 --stop_hgg_threshold 0.9 --curriculum True
#CHER
python train.py --tag 550 --learn normal --env KukaPickAndPlaceObstacle-v1 --curriculum True --batch_size 64 --buffer_size 500

## KukaPickNoObstacle
python train.py --tag 610 --learn hgg --env KukaPickNoObstacle-v1 --stop_hgg_threshold 0.3
python train.py --tag 620 --learn hgg --env KukaPickNoObstacle-v1 --graph True --n_x 31 --n_y 31 --n_z 21 --stop_hgg_threshold 0.9 --curriculum True
#CHER
python train.py --tag 650 --learn normal --env KukaPickNoObstacle-v1 --curriculum True --batch_size 64 --buffer_size 500

## KukaPickThrow
python train.py --tag 710 --learn hgg --env KukaPickThrow-v1 --stop_hgg_threshold 0.3 --epoch 30
python train.py --tag 720 --learn hgg --env KukaPickThrow-v1 --graph True --n_x 31 --n_y 31 --n_z 21 --stop_hgg_threshold 0.9 --epoch 30

## KukaPushLabyrinth
python train.py --tag 820 --learn hgg --env KukaPushLabyrinth-v1 --graph True --n_x 51 --n_y 51 --n_z 7 --stop_hgg_threshold 0.9 --curriculum True
#CHER
python train.py --tag 850 --learn normal --env KukaPushLabyrinth-v1 --curriculum True --batch_size 64 --buffer_size 500

## KukaPushSlide
python train.py --tag 910 --learn hgg --env KukaPushSlide-v1 --stop_hgg_threshold 0.3 --epoch 20

## KukaPushNew

#HER
python train.py --tag 1000 --learn normal --env KukaPushNew-v1 --epoch 20
#HGG
python train.py --tag 1010 --learn hgg --env KukaPushNew-v1 --stop_hgg_threshold 0.3 --epoch 20
#GC-HGG
python train.py --tag 1020 --learn hgg --env KukaPushNew-v1 --stop_hgg_threshold 0.3 --epoch 20 --graph True --n_x 31 --n_y 31 --n_z 11 --curriculum True
#CHER
python train.py --tag 1050 --learn normal --env KukaPushNew-v1 --epoch 20 --curriculum True

```

### Hand Manipulate Environments
```bash
# HandReach
python train.py --tag 1110 --learn hgg --env HandReach-v0 --stop_hgg_threshold 0.3 --epoch 20
python train.py --tag 1110 --learn hgg --env HandReach-v0 --stop_hgg_threshold 0.3 --epoch 20 --curriculum True

# HandManipulateEgg
python train.py --tag 1210 --learn hgg --env HandManipulateEgg-v0 --stop_hgg_threshold 0.3 --epoch 20
python train.py --tag 1210 --learn hgg --env HandManipulateEgg-v0 --stop_hgg_threshold 0.3 --epoch 20 --curriculum True

# HandManipulateBlock
python train.py --tag 1410 --learn hgg --env HandManipulateBlock-v0 --stop_hgg_threshold 0.3 --epoch 20
python train.py --tag 1410 --learn hgg --env HandManipulateBlock-v0 --stop_hgg_threshold 0.3 --epoch 20 --curriculum True

#HandManipulatePen
python train.py --tag 1510 --learn hgg --env HandManipulatePen-v0 --stop_hgg_threshold 0.3 --epoch 20
python train.py --tag 1510 --learn hgg --env HandManipulatePen-v0 --stop_hgg_threshold 0.3 --epoch 20 --curriculum True
```

## Playing 

To look at the agent solving the respective task according to his learned policy, issue the following command:

### Kuka Environments
```bash
# Scheme: python play.py --env env_id --goal custom --play_path log_dir --play_epoch <epoch number, latest or best>

# KukaReach
python play.py --env KukaReach-v1 --play_path policies/KukaReach/400-ddpg-KukaReach-v1-normal --play_epoch best

# KukaPickAndPlaceObstacle
python play.py --env KukaPickAndPlaceObstacle-v1 --play_path policies/KukaPickAndPlaceObstacle/520-ddpg-KukaPickAndPlaceObstacle-v1-hgg-graph-stop --play_epoch best
python play.py --env KukaPickAndPlaceObstacle-v1 --play_path policies/KukaPickAndPlaceObstacle/510-ddpg-KukaPickAndPlaceObstacle-v1-hgg-stop --play_epoch best

# KukaPickNoObstacle
python play.py --env KukaPickNoObstacle-v1 --play_path policies/KukaPickNoObstacle/610-ddpg-KukaPickNoObstacle-v1-hgg-stop --play_epoch best
python play.py --env KukaPickNoObstacle-v1 --play_path policies/KukaPickNoObstacle/620-ddpg-KukaPickNoObstacle-v1-hgg-graph-stop --play_epoch best

# KukaPickThrow
python play.py --env KukaPickThrow-v1 --play_path policies/KukaPickThrow/710-ddpg-KukaPickThrow-v1-hgg-stop --play_epoch best
python play.py --env KukaPickThrow-v1 --play_path policies/KukaPickThrow/720-ddpg-KukaPickThrow-v1-hgg-graph-stop --play_epoch best

# KukaPushLabyrinth
python play.py --env KukaPushLabyrinth-v1 --play_path policies/KukaPushLabyrinth/810-ddpg-KukaPushLabyrinth-v1-hgg-stop --play_epoch best
python play.py --env KukaPushLabyrinth-v1 --play_path policies/KukaPushLabyrinth/820-ddpg-KukaPushLabyrinth-v1-hgg-graph-stop --play_epoch best


#KukaPushNew
python play.py --env KukaPushNew-v1 --play_path policies/KukaPushNew/1010-ddpg-KukaPushNew-v1-hgg-stop --play_epoch best

```
### Fetch Environments
```bash
# FetchPushLabyrinth
# GC-HGG
python play.py --env FetchPushLabyrinth-v1 --goal custom --play_path policies/FetchPushLabyrinth/020-ddpg-FetchPushLabyrinth-v1-hgg-graph-stop-curriculum --play_epoch best
# HGG
python play.py --env FetchPushLabyrinth-v1 --goal custom --play_path policies/FetchPushLabyrinth/010-ddpg-FetchPushLabyrinth-v1-hgg-stop --play_epoch best
# HER
python play.py --env FetchPushLabyrinth-v1 --goal custom --play_path policies/FetchPushLabyrinth/000-ddpg-FetchPushLabyrinth-v1-normal --play_epoch best

# FetchPickObstacle
python play.py --env FetchPickObstacle-v1 --goal custom --play_path policies/FetchPickObstacle/100-ddpg-FetchPickObstacle-v1-normal --play_epoch best
python play.py --env FetchPickObstacle-v1 --goal custom --play_path policies/FetchPickObstacle/110-ddpg-FetchPickObstacle-v1-hgg-stop --play_epoch best
python play.py --env FetchPickObstacle-v1 --goal custom --play_path policies/FetchPickObstacle/120-ddpg-FetchPickObstacle-v1-hgg-graph-stop-curriculum --play_epoch best

# FetchPickNoObstacle
python play.py --env FetchPickNoObstacle-v1 --goal custom --play_path policies/FetchPickNoObstacle/200-ddpg-FetchPickNoObstacle-v1-normal --play_epoch best
python play.py --env FetchPickNoObstacle-v1 --goal custom --play_path policies/FetchPickNoObstacle/210-ddpg-FetchPickNoObstacle-v1-hgg-stop --play_epoch best
python play.py --env FetchPickNoObstacle-v1 --goal custom --play_path policies/FetchPickNoObstacle/220-ddpg-FetchPickNoObstacle-v1-hgg-graph-stop-curriculum --play_epoch best

# FetchPickAndThrow
python play.py --env FetchPickAndThrow-v1 --goal custom --play_path policies/FetchPickAndThrow/300-ddpg-FetchPickAndThrow-v1-normal --play_epoch best
python play.py --env FetchPickAndThrow-v1 --goal custom --play_path policies/FetchPickAndThrow/310-ddpg-FetchPickAndThrow-v1-hgg-stop  --play_epoch best
python play.py --env FetchPickAndThrow-v1 --goal custom --play_path policies/FetchPickAndThrow/320-ddpg-FetchPickAndThrow-v1-hgg-mesh-stop  --play_epoch best

# FetchPushNew
python play.py --env FetchPushNew-v1 --goal custom --play_path policies/FetchPushNew/1011-ddpg-FetchPushNew-v1-hgg-stop-curriculum --play_epoch best
python play.py --env FetchPushNew-v1 --goal custom --play_path policies/FetchPushNew/1012-ddpg-FetchPushNew-v1-hgg-stop --play_epoch best

# FetchReach
python play.py --env FetchReach-v1 --goal custom --play_path policies/FetchReach/1310-ddpg-FetchReach-v1-hgg-stop --play_epoch best

```
### Hand Environments
```bash
# HandManipulateEgg
python play.py --env HandManipulateEgg-v0 --play_path policies/HandManipulateEgg/1210-ddpg-HandManipulateEgg-v0-hgg-stop --play_epoch best
# HandReach
python play.py --env HandReach-v0 --play_path policies/HandReach/1110-ddpg-HandReach-v0-hgg-stop --play_epoch best
# HandManipulateBlock
python play.py --env HandManipulateBlock-v0 --play_path policies/HandManipulateBlock/1410-ddpg-HandManipulateBlock-v0-hgg-stop --play_epoch best
# HandManipulatePen
python play.py --env HandManipulatePen-v0 --play_path policies/HandManipulatePen/1510-ddpg-HandManipulatePen-v0-hgg-stop  --play_epoch best
```

## Running commands from HGG paper

Run the following commands to reproduce our main results shown in section 5.1 of the HGG paper.

```bash
python train.py --tag='HGG_fetch_push' --env=FetchPush-v1
python train.py --tag='HGG_fetch_pick' --env=FetchPickAndPlace-v1
python train.py --tag='HGG_hand_block' --env=HandManipulateBlock-v0
python train.py --tag='HGG_hand_egg' --env=HandManipulateEgg-v0
```
