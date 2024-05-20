
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



. scripts/base/train_MOA.sh ${gpu1} ${arch1} ${dataset1} ${norm0} ${config1}

sleep 10

. scripts/base/train_MOA.sh ${gpu1} ${arch1} ${dataset1} ${norm1} ${config1}

