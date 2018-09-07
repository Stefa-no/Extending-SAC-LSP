#!/bin/sh
# Start training with N actors, a Learner, and a PS
python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --her_type=1 --goal_type=0 --log_dir="./sac/data/ant-her1-goal0"  |
python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --her_type=1 --goal_type=1   --log_dir="./sac/data/ant-her1-goal1"  |
python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --her_type=2 --goal_type=0   --log_dir="./sac/data/ant-her2-goal0"  |
python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --her_type=3 --goal_type=0   --log_dir="./sac/data/ant-her3-goal0"  

