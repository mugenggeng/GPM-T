# GPM-T
The code for GPM-T and provide a demo of GPM-T+DCAN. We will upload the code as soon as possible.

### Environment

We use Miniconda3 to manage our python environments.

```sh
conda create -n dcan python=3.7
conda activate dcan
conda install matplotlib tqdm joblib h5py
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

### Data preparation



**ActivityNet v1.3**: 
We use the TSN feature extracted by the two-stream network. 
The frame interval is set to 16.
Using linear interpolation, each video feature sequence is rescaled to L = 100 snippets.

**THUMOS-14**: 
We use the TSN feature extracted by the TSN network.
The FPS of the feature is consistent with the original videos.
This feature is stored through HDF5 for rapid reading.

---

The TSN features for both datasets are available on the website. We also provide our **download links** to long-term support.

Baidu Netdisk:

[ActivityNet v1.3 and THUMOS-14](https://pan.baidu.com/s/1uY2nnLOBJ71mPl2KwBSBug), code: uvpk

Zenodo: 

[ActivityNet v1.3](https://zenodo.org/record/6650813) , [THUMOS-14](https://zenodo.org/record/6652094)

---

In order to use the downloaded video features, the **feature_path** attributes in the two **opt.py** need to be modified respectively.
### Training and Testing

For example on THUMOS-14. 

```sh
cd anet_thumos
```

Training model using 4 GPUS (ids=0,1,2,3).

```sh
Python train.py --gpus 0,1,2,3
```

Testing model on a sepific epoch. 

```sh
python test.py --checkpoint_path ./save/xxx-xx/ --test_epoch 4
```

The 'xxx-xx' is the training **work directory** in the 'save' directory named by a timestamp, such as "./save/20220116-1417". 

For ActivityNet v1.3, the training and testing procedure is similar to THUMOS-14. 

