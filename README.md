# RRTX-py
Python implementation of RRT-X [1], which is an asymtotically optimal single-query sampling-based replanning algorithm for dynamic environments.

## Requirements
- Anaconda3

## Make and activate conda environment
```
$ conda env create -f env.yml
$ conda activate rrtx
```
## Run RRT-X
```
$ python experiments_for_paper.py --exp-name {exp_name}
$ (ex) python experiments_for_paper.py --exp-name dynamic_2d_debug
```
## Make video for the experiment
```
$ python experiments_for_paper.py --exp-name {exp_name} --file-ctr {exp_start_counter} --start-move-at-ctr {robot_moving_start_counter} --max-file-ctr {exp_end_counter}
$ (ex) python make_video.py --exp-name dynamic_2d_debug --file-ctr 0 --max-file-ctr 400 --start-move-at-ctr 200 
```
## Refrences
[1] Otte M, Frazzoli E. RRTX: Asymptotically optimal single-query sampling-based motion planning with quick replanning. The International Journal of Robotics Research. 2016;35(7):797-822. doi:10.1177/0278364915594679
