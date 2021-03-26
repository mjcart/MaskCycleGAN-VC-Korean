# MaskCycleGAN-VC
Implementation of Kaneko et al.'s MaskCycleGAN-VC model for non-parallel voice conversion.

## VCC2018 Dataset

The authors of the paper used the dataset from the Spoke task of [Voice Conversion Challenge 2018 (VCC2018)](https://datashare.ed.ac.uk/handle/10283/3061). This is a dataset of non-parallel utterances from 6 male and 6 female speakers. Each speaker utters approaximately 80 sentences.

Download the dataset from the command line.
```
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip?sequence=2&isAllowed=y
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip?sequence=3&isAllowed=y
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_reference.zip?sequence=5&isAllowed=y
```

Unzip the dataset file.
```
mkdir vcc2018
apt-get install unzip
unzip vcc2018_database_training.zip?sequence=2 -d vcc_2018/
unzip vcc2018_database_evaluation.zip?sequence=3 -d vcc_2018/
unzip vcc2018_database_reference.zip?sequence=5 -d vcc_2018/
```

## Data Preprocessing

To expedite training, preprocess the dataset by converting waveforms to melspectograms, then save the spectrograms as pickle files `<speaker_id>normalized.pickle` and normalization statistics (mean, std) as npz files `<speaker_id>_norm_stats.npz`. Convert waveforms to spectrograms using the [melgan vocoder](https://github.com/descriptinc/melgan-neurips) to ensure that you can decode voice converted spectrograms to waveform and listen to your samples during inference.

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_training \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2
```

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_evaluation \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_evaluation \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4
```

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_reference \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_reference \
  --speaker_ids VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2
```

## Training

Train MaskCycleGAN-VC to convert between <speaker_id_A> and <speaker_id_B>. You should start to get excellent results after only several hundred epochs.
```
python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_<speaker_id_A>_<speaker_id_B> \
    --seed 0 \  # random seed
    --save_dir results/ \  # save directory for all logs and model checkpoints
    --preprocessed_data_dir vcc2018_training_preprocessed/ \  # directory with preprocessed data
    --speaker_A_id <speaker_id_A> \
    --speaker_B_id <speaker_id_B> \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \  # pushes decoded wavforms to tensorboard
    --num_epochs 6172 \
    --batch_size 1 \
    --lr 5e-4 \
    --decay_after 1e4 \
    --sample_rate 22050 \  # sampling rate of MelGAN vocoder (do not change)
    --num_frames 64 \  # length of Mel-spectrograms
    --max_mask_len 25 \  # mask of length 0-25 is applied to the Mel-spectrogram
    --gpu_ids 0 \
```

To continue training from a previous checkpoint in the case that training is suspended, add the argument `--continue train` while keeping all others the same. The model saver class will automatically load the most recently saved checkpoint and resume training.




