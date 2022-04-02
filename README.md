# Hindsight Goal Generation Based on Graph-Based Diversity and Proximity


## Requirements
1. Ubuntu 16.04 or macOS Catalina 10.15.7 (newer versions also work well) 
2. Python 3.5.2 (newer versions such as 3.6.8 should work as well, 3.8 or higher is not suggested)
3. MuJoCo == 2.00 (see instructions on https://github.com/openai/mujoco-py)


6. Install requirements with pip install -r requirements.txt
```bash
pip install -r requirements.txt
```
  
7. Videos about Kuka Environments can be found here:
https://videoviewsite.wixsite.com/gc-hgg
   
8. parallel implementation of GC-HGG can be found in branch concurrency.

## New Kuka Environments
<img width="1920" alt="KukaReach" src="https://user-images.githubusercontent.com/57254021/124595467-57bb0380-de61-11eb-8597-7e83e4c140d0.png">
<img width="1920" alt="KukaPickNoObstacle" src="https://user-images.githubusercontent.com/57254021/124595454-52f64f80-de61-11eb-9287-64482f531d41.png">
<img width="1920" alt="KukaPickObstacle" src="https://user-images.githubusercontent.com/57254021/124595460-55f14000-de61-11eb-9c60-09e2230e555a.png">
<img width="1240" alt="KukaPush" src="https://user-images.githubusercontent.com/57254021/141085082-7c14281b-668b-408a-8f9b-5e364104c36e.png">


## Training under different environments

The following commands are used to train the agent in different environments with HGG, HER, G-HGG, C-HGG.
Note that new Kuka Environments are introduced.

### Kuka Environments
```bash
## KukaReach
python train.py --tag 400 --learn normal --env KukaReach-v1 
#GC-HGG
python train.py --tag 410 --learn hgg --env KukaReach-v1 --curriculum True --stop_hgg_threshold 0.3
#CHER
python train.py --tag 450 --learn normal --env KukaReach-v1 --curriculum True --batch_size 64 --buffer_size 500 --epoch 10

## KukaPickAndPlaceObstacle

#HGG
python train.py --tag 510 --learn hgg --env KukaPickAndPlaceObstacle-v1 --stop_hgg_threshold 0.3
#GC-HGG
python train.py --tag 520 --learn hgg --env KukaPickAndPlaceObstacle-v1 --graph True --n_x 11 --n_y 11 --n_z 7 --stop_hgg_threshold 0.9 --curriculum True
#G-HGG
python train.py --tag 530 --learn hgg --env KukaPickAndPlaceObstacle-v1 --graph True --n_x 11 --n_y 11 --n_z 7 --stop_hgg_threshold 0.9
#CHER
python train.py --tag 550 --learn normal --env KukaPickAndPlaceObstacle-v1 --curriculum True --batch_size 64 --buffer_size 500

## KukaPickNoObstacle

#HGG
python train.py --tag 610 --learn hgg --env KukaPickNoObstacle-v1 --stop_hgg_threshold 0.3
#GC-HGG
python train.py --tag 620 --learn hgg --env KukaPickNoObstacle-v1 --graph True --n_x 31 --n_y 31 --n_z 15 --stop_hgg_threshold 0.5 --curriculum True
#G-HGG
python train.py --tag 630 --learn hgg --env KukaPickNoObstacle-v1 --graph True --n_x 31 --n_y 31 --n_z 15 --stop_hgg_threshold 0.5
#CHER
python train.py --tag 650 --learn normal --env KukaPickNoObstacle-v1 --curriculum True --batch_size 64 --buffer_size 500

## KukaPushNew

#HER
python train.py --tag 1000 --learn normal --env KukaPushNew-v1 --epoch 10
#HGG
python train.py --tag 1010 --learn hgg --env KukaPushNew-v1 --stop_hgg_threshold 0.3 --epoch 10
#GC-HGG
python train.py --tag 1020 --learn hgg --env KukaPushNew-v1 --stop_hgg_threshold 0.3 --epoch 10 --graph True --n_x 5 --n_y 11 --n_z 7 --curriculum True
#G-HGG
python train.py --tag 1030 --learn hgg --env KukaPushNew-v1 --stop_hgg_threshold 0.3 --epoch 10 --graph True --n_x 5 --n_y 11 --n_z 7
#CHER
python train.py --tag 1050 --learn normal --env KukaPushNew-v1 --epoch 10 --curriculum True 

```


## Playing 

To look at the agent solving the respective task according to his learned policy, issue the following command:
### Kuka Environments
```bash
# Scheme: python play.py --env env_id --goal custom --play_path log_dir --play_epoch <epoch number, latest or best>

# KukaReach
python play.py --env KukaReach-v1 --play_path log/400-ddpg-KukaReach-v1-normal --play_epoch best

# KukaPickAndPlaceObstacle
python play.py --env KukaPickAndPlaceObstacle-v1 --play_path log/520-ddpg-KukaPickAndPlaceObstacle-v1-hgg-graph-stop-curriculum --play_epoch best

# KukaPickNoObstacle
python play.py --env KukaPickNoObstacle-v1 --play_path log/620-ddpg-KukaPickNoObstacle-v1-hgg-graph-stop-curriculum --play_epoch best

#KukaPushNew
python play.py --env KukaPushNew-v1 --play_path log/1020-ddpg-KukaPushNew-v1-hgg-graph-stop-curriculum --play_epoch best
```
