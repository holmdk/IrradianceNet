# IrradianceNet
This is the official implementation of the [IrradianceNet paper]().

### Sample 1
![Alt Text](results/convlstm/video_sample1.gif)

### Sample 2
![Alt Text](results/convlstm/video_sample2.gif)


## 1. Installation
This project was conducted using `pytorch==1.8.1`. It might also work with previous/future versions, but this is obviously not guaranteed.
I have attached the requirements.txt of my conda environment. Run the following two commands to get started.

```
git clone https://github.com/holmdk/IrradianceNet.git
```

Next, install virtual environment (conda but others should work too)
```
conda install --file requirements.txt
```

## 2. Configuration
Please read the following descriptions before you start running any code. If you follow all steps you should be able to predict satellite-derived solar irradiance using our pretrained ConvLSTM model relatively quickly.   
- If you are interested in the final product of solar irradiance forecasts, we have only supplied all the ingredients for a small period (2016-05-01 to 2016-05-05), but extending this is relatively straightforward and explained in detail later. 
- If you are only interested in the effective cloud albedo, then you can use our setup for the entire period of 1983-01-01 - 2017-12-31 straight away.
> The results shown in our paper are based on the entire test dataset (2016-2017), and this small subset is **only meant for other researchers to quickly validate or replicate our results or for prototyping new models.** Nevertheless, the results here largely align with the ones demonstrated in the paper.

### 2.1 Input data
As we are not the owners of the original Effective Cloud Albedo dataset, you will have to download the data from the [following site](https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002_01). Perform the following steps:
1. Sign up to the website
2. Select both a) the Effective cloud albedo (CAL) and b) Surface incoming shortwave radiation (SIS) with  "Instantaneous" 
3. Download the following time interval (2016-05-01 to 2016-05-05) for both a) and b)
4. See the next section on Pre-processing on how to proceed from these raw datasets.
5. Download [clear sky irradiance dataset](https://drive.google.com/file/d/16__58FmYl31wxuwcUcvS9Z52Zrex8YcT/view?usp=sharing)

- If you want to use meta-features (like in the original paper), most are relatively straightforward to generate and download. We give a description of these in the paper. If you want to use these meta-features, you need to append them to the Dataloader and also set `in_channels=n`, where n is the number of meta-features in the ConvLSTM model. Using all meta-features mentioned in the paper will boost performance of the ConvLSTM models by reducing MAE (over the entire test set) by roughly 5-10, but the version proposed here is still superior to the state-of-the-art TVL-1 Optical Flow. 


### 2.2 Pre-processing
The usage of pre-processing is relatively limited and will be described in detail in a future publication. We list the primary steps here along with a reference to the various scripts needed so you can essentially replicate our methodology from scratch. The primary steps are:
1. Having followed the Input data section, you should put the downloaded 'CAL' and 'SIS' files into a folder as `/data/SARAH/`.
2. Modify the `/src/data/process_raw_data.py` script if you want a different name, time horizon, etc. in the config dict listed in the script.
3. Run the `/src/data/process_raw_data.py` script.
4. You should now have all the data required to run inference.  

__\*Note__ the raw dataset size is multiple terabytes so there is a high probability you will run our of memory if you run this script for a large portion of the dataset.  
__\*\*Note__ there are many ways to derive clear sky irradiance, which is the component needed to convert CAL to SIS. Here, I have attached for the period 2016-05-01 to 2016-05-05 using Ineiched Clear Sky model. If you want more clear-sky data this should be trivial using the popular [pvlib library](https://pvlib-python.readthedocs.io/en/stable/).

### 2.3 Pre-trained model
We are supplying a pre-trained model specifically optimized for 2-step ahead prediction (1-hour), but the model can still be used for longer time horizons by setting n to be the number of future frames you want to predict using the flag `n_future_frames=n`. If you are interested in models optimized for more time steps let me know (up to 8 future frames).  

You can download our high-resolution model [here](https://drive.google.com/file/d/1fAbgjOavED_BArz00gzoLnp0KXXujGyf/view?usp=sharing). After downloading, put the model into the `/models/full_res` folder.


## 3. Usage (inference)

If you want to run the ConvLSTM model with default settings, run the following:

```python 
python test.py
```

If you want the benchmark TVL-1 Optical Flow, run the following:

```python 
python test.py --model_name opt_flow
```

There are other arguments you can change as well, but this is only recommended for expert users.


## 4. Results

2016-05-01 to 2016-05-05
| Model | MAE (k) |  RMSE (k)  |   MAE (SIS) |  RMSE (SIS)  |  
| --- | --- | --- | --- | --- |
| ConvLSTM (high-res, only CAL) | 0.0674 | 0.1032 |  46.81  | 68.09 | 
| TVL-1 Optical Flow (high-res, only CAL) | 0.0782 | 0.1459 | 58.38 | 93.99 |


**In the figures below, we**
- Demonstrate predictions in the top row, ground truth in middle row and the pixel-wise absolute error in the bottom row
- Column 1 is one-step ahead (30 minutes), and column 2 is two-steps ahead (1 hour)

## 5. Visualization

#### Example 1
<img src="/results/convlstm/batch_0006.png" alt="drawing" width="400"/>

#### Example 2
<img src="/results/convlstm/batch_0008.png" alt="drawing" width="400"/>
