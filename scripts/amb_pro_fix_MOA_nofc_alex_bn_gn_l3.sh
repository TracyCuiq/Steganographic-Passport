
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

exp_id_bn_cf10=46
exp_id_bn_cf100=27
exp_id_bn_cal101=23
exp_id_bn_cal256=36

type='fake2-'
att_per='fix'


. scripts/base/amb_pro1_MOA.sh ${gpu0} ${arch0} ${dataset0} ${norm0} ${config0} ${exp_id_bn_cf10} ${type} ${att_per}

sleep 10

. scripts/base/amb_pro1_MOA.sh ${gpu0} ${arch0} ${dataset1} ${norm0} ${config0} ${exp_id_bn_cf100} ${type} ${att_per}

sleep 10

. scripts/base/amb_pro1_MOA.sh ${gpu0} ${arch0} ${dataset2} ${norm0} ${config0} ${exp_id_bn_cal101} ${type} ${att_per}

sleep 10

. scripts/base/amb_pro1_MOA.sh ${gpu0} ${arch0} ${dataset3} ${norm0} ${config0} ${exp_id_bn_cal256} ${type} ${att_per}