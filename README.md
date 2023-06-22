
## What can a cook in Italy teach a mechanic in India? Action Recognition Generalisation Over Scenarios and Locations

This is the official resource for the paper "What can a cook in Italy teach a mechanic in India? Action Recognition Generalisation Over Scenarios and Locations" including the dataset (ARGO1M) and the code.

<img src="resources/gif_new.gif" width="600">

## BibTeX

If you use the ARGO1M dataset and/or our CIR method code, please cite:

<code>@inproceedings{Plizzari2023,
  title={What can a cook in Italy teach a mechanic in India? Action Recognition Generalisation Over Scenarios and Locations},
  author={Plizzari, Chiara and Perrett, Toby and Caputo, Barbara and Damen, Dima},
  booktitle={preprint},
  year={2023}}</code>
  
## Requirements
We provide modified training scripts for CIR to replicate paper results. To install dependencies:

<code>conda env create -f environment.yml</code> 


## Dataset: ARGO1M

### How to download ARGO1M

Our annotated clips making up ARGO1M, are curated from videos of the large-scale Ego4D dataset. Before using ARGO1M, you thus need to sign the EGO4D License Agreement. Here are the three steps to follow for downloading the dataset:

1. Go to [ego4ddataset.com](https://ego4d-data.org/docs/start-here/#download-data) to review and execute the EGO4D License Agreement, and you will be emailed a set of AWS access credentials when your license agreement is approved, which will take 48hrs.

2.  The datasets are hosted on Amazon S3 and require credentials to access. AWS CLI uses the credentials stored in the home directory file: <code>~/.aws/credentials</code>. If you already have credentials configured then you can skip this step. If not, then:
  - Install the AWS CLI from: [AWS CLI](https://aws.amazon.com/cli/)
  - Open a command line and type <code>aws configure</code>
  - Leave the default region blank, and enter your AWS access id and secret key when prompted.

  The CLI requires python >= 3.8. Please install the prerequisites via <code>python setup.py install</code> (easyinstall) at the repo root, or via <code>pip install -r requirements.txt</code>. 
  
3. Download the dataset using the following command: <code>python code/scripts/download_all.py --flag DEST_DIR</code>, where <code>flag</code> is either <code>ffcv</code> or <code>csv</code>.


You can directly download our FFCV encodings for all ARGO1M splits as well as the CSV files described below.

### CSV 

We provide the .csv files for all the proposed splits. 

Those contain the following entries: 

- <code>uid</code>: uid of the video clip; 
- <code>scenario_idx</code>: scenario label (index-scenario association in <a href="https://github.com/Chiaraplizz/ARGO1M-What-can-a-cook/blob/main/data/index_scenario.txt">index_scenario.txt</a>);
- <code>location_idx</code>: location label (index-location association in <a href="https://github.com/Chiaraplizz/ARGO1M-What-can-a-cook/blob/main/data/index_location.txt">index_location.txt</a>); 
- <code>label</code>: action label (index-action association in <a href="https://github.com/Chiaraplizz/ARGO1M-What-can-a-cook/blob/main/data/index_verb.txt">index_verb.txt</a>);
- <code>timestamp</code>: starting timestamp;
- <code>timeframe</code>: starting timeframe;
- <code>narration</code>: narration; 
- <code>action_start_feature_idx</code>: starting feature index for SlowFast pre-extracted features;
- <code>action_end_feature_idx</code>: ending feature index for SlowFast pre-extracted features.

### FFCV 

To speed up training, we used <a href="https://ffcv.io/">FFCV</a> encodings of both training and test sets for each of the proposed splits. 

We also provide the scripts for extracting them using the given <a href="s3://ego4d-bristol/public/ARGO1M/">CSV</a> files. After downloading Ego4D SlowFast features, you can extract FFCVs by running: 

<code>python /scripts/dataset_ffcv_encode.py --config /configs/{config_file}.yaml --split {split_name}</code>

### Code structure

We designed the code in such a way that makes it easier to try your own methods and losses on top of it.
Suppose you want to introduce a new module called MyModule in our pipeline.
- You can define MyModule in <code>models.py</code>.

- In the corresponding <code>config.yaml</code>, you can add to <code>model_types</code> MyModule, with corresponding attributes in <code>model_names</code>, <code>model_lrs</code>, <code>model_use_train</code>, <code>model_use_eval</code> and <code>step</code>.

- In <code>model_inputs</code>, you can specify the input to MyModule, by prepending the model name that has provided the output, in the form {"arg":"other_model_name.output_name"}, e.g. {"input_logits":"mlp.logits"}.
  
- You can do the same by adding your new loss MyLoss in <code>loss_types</code>, along with the corresponding <code>loss_names</code>, and by specifying the corresponding <code>loss_inputs</code> in the form {"arg":"other_model_name.output_name"}, e.g. {"logits":"mlp.logits"}.


### Steps for training

The folder <code>scripts</code> contains code and bash scripts to reproduce the paper results. To re-create CIR results:

1. Modify <code>config</code> internal paths to match the location of FFCV data.

2. Run <code>python run.py --config configs/config_run/run_CIR.yaml</code>

### License 

All files in this repository are copyright by us and published under the Creative Commons Attribution-NonCommerial 4.0 International License, found <a href="http://en.wikipedia.org/wiki/Main_Page">here</a>. This means that you must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. You may not use the material for commercial purposes.
