#python tracking/train.py --script ostrack --config momn_3_ts_all_300_384 --mode multiple --nproc_per_node 5 --use_wandb 1
#python tracking/test.py --tracker_name ostrack --tracker_param momn_3_ts_all_300_384 --dataset tnl2k --debug 0 --threads 10 --num_gpus 5
#python tracking/test.py --tracker_name ostrack --tracker_param momn_3_ts_all_300_384 --dataset lasot --debug 0 --threads 10 --num_gpus 5

#python tracking/train.py --script ostrack_re --config momn_3_ts_re_la --mode multiple --nproc_per_node 5 --use_wandb 1
#python tracking/test.py --tracker_name ostrack_re --tracker_param momn_3_ts_re_la --dataset tnl2k --debug 0 --threads 10 --num_gpus 5
#python tracking/test.py --tracker_name ostrack_re --tracker_param momn_3_ts_re_la --dataset lasot --debug 0 --threads 10 --num_gpus 5

#python tracking/train.py --script ostrack --config momn_3_all_300 --mode multiple --nproc_per_node 5 --use_wandb 1
#python tracking/train.py --script ostrack --config momn_3_all_v_300 --mode multiple --nproc_per_node 5 --use_wandb 1

python tracking/test.py --tracker_name ostrack --tracker_param momn_3_all_300 --dataset tnl2k --debug 0 --threads 10 --num_gpus 5
python tracking/test.py --tracker_name ostrack --tracker_param momn_3_all_300 --dataset lasot --debug 0 --threads 10 --num_gpus 5

python tracking/test.py --tracker_name ostrack --tracker_param momn_3_all_v_300 --dataset tnl2k --debug 0 --threads 10 --num_gpus 5
python tracking/test.py --tracker_name ostrack --tracker_param momn_3_all_v_300 --dataset lasot --debug 0 --threads 10 --num_gpus 5