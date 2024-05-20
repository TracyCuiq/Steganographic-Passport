
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

exp_id_cifar10=25
exp_id_cifar100=15
exp_id_cal101=18
exp_id_cal256=24

type='fake2-'
att_per='ERB'



#. scripts/base/amb_pro_ERB_MOA.sh ${gpu0} ${arch1} ${dataset0} ${norm0} ${config1} ${exp_id_cifar10} ${type} ${att_per}
#sleep 10

#. scripts/base/amb_pro_ERB_MOA.sh ${gpu0} ${arch1} ${dataset1} ${norm0} ${config1} ${exp_id_cifar100} ${type} ${att_per}
#sleep 10

#. scripts/base/amb_pro_ERB_MOA.sh ${gpu1} ${arch1} ${dataset2} ${norm0} ${config1} ${exp_id_cal101} ${type} ${att_per}
#sleep 10

. scripts/base/amb_pro_ERB_MOA.sh ${gpu1} ${arch1} ${dataset3} ${norm0} ${config1} ${exp_id_cal256} ${type} ${att_per}