# KD on CommonVoice Fr

## Preparation
pre-trained teacher models: ngongotaha:/hdd1/yan/tea_models.tar.gz
results folders: ngongotaha:/hdd1/yan/results.tar.gz

Please copy the files to the your cluster. The training has completed 2 epochs, so it can start from there.


## Installation
1. please install the newest version of speechbrain.
2. pip install h5py


## Save teachers inference results
python save_teachers.py hparams/save_teachers_fr.yaml --output_folder /hdd1/yan/kd_results_fr/teachers_save --data_folder /datasets/commonvoice/fr/cv-corpus-6.1-2020-12-11/fr --tea_models_dir ./tea_path.txt

1. The output file will be around 1TB, so please make sure the location of '--output_folder' has enough space.
2. '--tea_models_dir ./tea_path.txt' is the paths of teacher models. See the example file 'tea_path.txt'.


## KD
Please see commands.sh. there are 7 experiments in total.
Example:
python train_kd.py hparams/train_kd_sgd_fr.yaml --data_folder /datasets/commonvoice/fr/cv-corpus-6.1-2020-12-11/fr --pretrain_st_dir /home/yan/kd_commonvoice_fr/tea_models/st --tea_infer_dir /hdd1/yan/kd_results_fr/teachers_save/ --output_folder /home/yan/kd_commonvoice_fr/results/weighted_015drop_1lr --strategy weighted

'--pretrain_st_dir' is the directory of student model. After uncompressing tea_models.tar.gz, you can find tea_models/st/
'--tea_infer_dir' is the directory of inference result file.
