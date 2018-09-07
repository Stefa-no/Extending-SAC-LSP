#!/bin/sh
# Start training with N actors, a Learner, and a PS

# Set addresses here
pses=localhost:9000 # Comma-separated list of parameter servers
workers=localhost:9001,localhost:9002,localhost:9003,localhost:9004,localhost:9005,localhost:9006,localhost:9007,localhost:9008,localhost:9009 # Comma-separated list of workers

# Visible CUDA devices set to None to prevent PS from using the GPU
CUDA_VISIBLE_DEVICES='' python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5p"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=ps --task_index=0 &
CUDA_VISIBLE_DEVICES='' python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5l"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=0 &
CUDA_VISIBLE_DEVICES=0 python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5w1"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=1 &
CUDA_VISIBLE_DEVICES=0 python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5w2"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=2 &
CUDA_VISIBLE_DEVICES=0 python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5w3"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=3 &
CUDA_VISIBLE_DEVICES=0 python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5w4"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=4 &
CUDA_VISIBLE_DEVICES=0 python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5w5"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=5 &
CUDA_VISIBLE_DEVICES=0 python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5w6"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=6 &
CUDA_VISIBLE_DEVICES=0 python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5w7"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=7 &
CUDA_VISIBLE_DEVICES=0 python ./examples/mujoco_all_sac.py --domain=ant --policy=lsp --env=ant  --log_dir="./sac/data/ant-test-5w8"  \
    --ps_hosts=$pses --worker_hosts=$workers --job_name=worker --task_index=8 &

# Clean up with ''' for pid in $(ps -ef | grep "python" | awk '{print $2}'); do kill -9 $pid; done ''' or similar
