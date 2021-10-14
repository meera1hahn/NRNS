python -W ignore run.py \
    --behavioral_cloning \
    --dataset 'gibson' \
    --bc_type 'gru'\
    --path_type 'curved' \
    --difficulty 'easy' \
    --gibson_bc_model_path 'gibson_bc_gru.pt'

