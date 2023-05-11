# mygra_code
night of 511, finished the vit and cubf100 and cubl100 part and run it successfully.

正式训练不要忘记将train.py中if epoch % 1 == 0:改回10

python REFILLED/pretrain.py --data_name CUB-f100 --network_name vit --n_training_epochs 2 --batch_size 64 --vit_num_classes 200 --vit_image_size 224 --vit_patch_size 16

将模型修改为 CUB-l100_vit_teacher.model 没错，虽然是使用的f100进行训练，但是还是要使用l100作为名字，注意就行

python REFILLED/main.py --data_name CUB-l100 --teacher_network_name vit --student_network_name wide_resnet --batch_size 64 --vit_num_classes 200 --vit_image_size 224 --vit_patch_size 16
