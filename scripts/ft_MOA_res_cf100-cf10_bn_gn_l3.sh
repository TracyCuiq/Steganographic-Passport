
gpu0='0'
gpu1='1'
gpu2='2'
gpu3='3'

arch0='alexnet'
arch1='resnet'

config0='passport_configs/alexnet_passport_l3.json'
config1='passport_configs/resnet18_passport_l3.json'

dataset0='cifar10'
dataset1='cifar100'
dataset2='caltech-101'
dataset3='caltech-256'

norm0='bn'
norm1='gn'

exp_id_bn=17
exp_id_gn=18


. scripts/base/ft_MOA.sh ${gpu1} ${arch1} ${dataset1} ${dataset0} ${norm0} ${config1} ${exp_id_bn}

sleep 10

. scripts/base/ft_MOA.sh ${gpu1} ${arch1} ${dataset1} ${dataset0} ${norm1} ${config1} ${exp_id_gn}

