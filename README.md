# IrradianceNet


## Installation
This project was conducted using `pytorch==1.8.1`. It might also work with previous/future versions, but this is obviously not guaranteed.
I have attached the requirements.txt of my conda environment, which you can use by calling `conda install --file requirements.txt`

## Configuration
Please read the following descriptions before you start running any code.

### Input data
As we are not the owners of the original Effective Cloud Albedo dataset, you will have to download the data from the [following site](https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002_01). Sign up to the website, and make sure to select "Instantaneous" when you download the "Effective cloud albedo" dataset. See the next section on Pre-processing on how to proceed from this raw dataset.

- If you want to use meta-features (like in the original paper), most are relatively straightforward to generate but for topography you need to download it from the [following site](google.com). These are optional features that you can disable/enable by setting the flag `enable_metafeatures=True` and `in_channels=n`, where n is the number of meta-features. Using all meta-features mentioned in the paper will boost performance of the ConvLSTM models by reducing MAE (over the entire test set) by roughly 10. 

### Pre-processing
The usage of pre-processing is relatively limited and will be described in detail in a future publication. The primary steps are:
- Before the dataset can be used in the `AlbedoLoader` and `AlbedoPatchLoader` module, it will need to be converted to a `torch.tensor` and saved to a torch file called `lres_test_subset.pt` and `test_subset.pt`, respectively. I have attached a script for this functionality called `process_raw_data.py`.
- **Importantly**, if you want to use the exact same dates as we do in this repo here, I have attached a `test_subset_timestamps.csv` that you can extract from the raw dataset BEFORE saving it to a torch file. The results shown in our paper are based on the entire test dataset, however, and this subset is **only meant for other researchers to quickly validate or replicate our results or for prototyping new models.**

### Pre-trained model
We are supplying a pre-trained model specifically optimized for 2-step ahead prediction (1-hour), but the model can still be used for longer time horizons by setting n to be the number of future frames you want to predict using the flag `n_future_frames=n`. If you are interested in models optimized for more time steps let me know (up to 8 future frames).

## Usage (inference)

Run the `test.py` script followed by a few args in case you want to change the default settings.
