### Steps to reproduce results:
1. Download Amazon reviews datasets from https://jmcauley.ucsd.edu/data/amazon/links.html
   and preprocess them following this paper https://dl.acm.org/doi/10.1145/3580305.3599519.
   Place the preprocessed datasets under data/
1. Setup the python environment using the commands given below.
1. Run prompt_llm.py to gather LLM_c's clusterings for Amazon reviews 
   datasets. Then, run parse_clusters.py to parse the outputs.
1. Run preprocess.py to create two files for each dataset:
    idx2text.json and clusterings.json
1. Run main.py for clustering using the commands given below.


#### Setup conda environment
```
conda create -n llm_cluster python=3.10.9
source activate llm_cluster
pip install pytz pandas tqdm matplotlib pyarrow pydot
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.35.2
pip install accelerate==0.25.0
pip install awscli boto3 botocore==1.31.63 --upgrade 
pip install peft==0.4.0
pip install bitsandbytes==0.40.0
pip install scikit-learn==1.2.2
pip install sentencepiece==0.1.99
pip install protobuf
pip install evaluate
```

#### Table 2 (showing commands for Arts dataset)
```
python unsupervised.py --dataset Arts --val_size 1000 --test_size 3000
python main.py --dataset Arts --set_enc_type nia --train_size 1 --val_size 1000 --test_size 3000 --loss_type scl
python main.py --pretrain 1 --dataset Arts --set_enc_type sia_hid_mean --val_size 1000 --test_size 3000 --loss_type triplet_neutral
python main.py --dataset Arts --set_enc_type sia_hid_mean --train_size 1 --val_size 1000 --test_size 3000 --loss_type triplet_neutral --load_ckpt_path '../outputs/Arts/pretrain_sia_hid_mean|triplet_neutral|margin:0.3|cutoff:0.0|C:0.15|r:0.5|tau:0.5|max_items:None|max_clusters:None|train_size:0.8|model_name:google-flan-t5-base/clus_checkpoint_best.bin' --lr 5e-5
```

#### Vary loss functions
```
python main.py --dataset Arts --set_enc_type sia_first --train_size 3000 --val_size 1000 --test_size 3000 --loss_type triplet_neutral
python main.py --dataset Arts --set_enc_type sia_first --train_size 3000 --val_size 1000 --test_size 3000 --loss_type scl 
python main.py --dataset Arts --set_enc_type sia_first --train_size 3000 --val_size 1000 --test_size 3000 --loss_type cross_entropy
python main.py --dataset Arts --set_enc_type sia_first --train_size 3000 --val_size 1000 --test_size 3000 --loss_type triplet 
python main.py --dataset Arts --set_enc_type sia_first --train_size 3000 --val_size 1000 --test_size 3000 --loss_type basic
```


#### Vary set encoders
```
python main.py --dataset Arts --set_enc_type nia --train_size 3000 --val_size 1000 --test_size 3000 --loss_type triplet_neutral
python main.py --dataset Arts --set_enc_type fia --train_size 3000 --val_size 1000 --test_size 3000 --loss_type triplet_neutral
python main.py --dataset Arts --set_enc_type sia_hid_mean --train_size 3000 --val_size 1000 --test_size 3000 --loss_type triplet_neutral 
python main.py --dataset Arts --set_enc_type sia_mean --train_size 3000 --val_size 1000 --test_size 3000 --loss_type triplet_neutral
```

#### Finetuning ablation study
```
python main.py --dataset Arts --set_enc_type sia_hid_mean --train_size 3000 --val_size 1000 --test_size 3000 --loss_type triplet_neutral --load_ckpt_path '../outputs/Arts/pretrain_sia_hid_mean|triplet_neutral|margin:0.3|cutoff:0.0|C:0.15|r:0.5|tau:0.5|max_items:None|max_clusters:None|train_size:0.8|model_name:google-flan-t5-base/clus_checkpoint_best.bin' --lr 5e-5
```
