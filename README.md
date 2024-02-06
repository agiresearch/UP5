# UFO: Unbiased Foundation Model for Fairness-aware Recommendation
## Abstract

Recent advances in Foundation Models such as Large Language Models (LLMs) have propelled them to the forefront of Recommender Systems (RS). Despite their utility, there is a growing concern that LLMs might inadvertently perpetuate societal stereotypes, resulting in unfair recommendations. Since fairness is critical for RS as many users take it for decision-making and demand fulfillment, this paper focuses on user-side fairness for LLM-based recommendation where the users may require a recommender system to be fair on specific sensitive features such as gender or age. In this paper, we dive into the extent of unfairness exhibited by LLM-based recommender models based on both T5 and LLaMA backbones, and discuss appropriate methods for promoting equitable treatment of users in LLM-based recommendation models. We introduce a novel Counterfactually-Fair-Prompt (CFP) method towards Unbiased Foundation mOdels (UFO) for fairness-aware LLM-based recommendation. 
Experiments are conducted on two real-world datasets, MovieLens-1M and Insurance, and compared with both matching-based and sequential-based fairness-aware recommendation models. Results show that CFP achieves better recommendation performance with a high level of fairness.
<img width="801" alt="Screen Shot 2024-02-05 at 10 26 34 PM" src="https://github.com/agiresearch/UP5/assets/28013619/089a6cfb-6545-40f0-9236-ddcb1cdc1f86">


## Environment
```
torch==1.11.0
transformers==4.21
sklearn==1.1.2
numpy==1.23.1
```

## pretrain P5 on MovieLens and Insurance
Train.csv is the original Insurance dataset without any preprocessing
### process data first:
```
cd pretrain/
python data_preprocessing_insurance.py --data_dir data_location_of_your_file

movielens dataset does not need preprocessing.
```
### pretrain P5 on MovieLens
```
python -m torch.distributed.launch \
    --master_port 12323 \
    main.py \
    --distributed --multiGPU \
        --seed 2022 \
        --batch_size 16 \
        --warmup_prop 0.1 \
        --lr 1e-3 \
        --clip 1.0 \
        --model_type 't5-small' \
        --epoch 10 \
        --gpu '5,6,7,8' \
        --logging_step 5000 \
        --logging_dir 'pretrain_t5_small_movie.log' \
```
### pretrain P5 on Insurance on sequential data
```
python -m torch.distributed.launch \
        --master_port 12323 \
        main.py \
        --distributed --multiGPU \
        --seed 2022 \
        --task insurance \
        --batch_size 16 \
        --warmup_prop 0.05 \
        --lr 1e-3 \
        --clip 1.0 \
        --model_type 't5-small' \
        --epoch 20 \
        --gpu '0,1' \
        --logging_step 5000 \
        --logging_dir 'pretrain_t5_small_insurance_sequential.log' \
```
### pretrain P5 on Insurance on direct data
```
python -m torch.distributed.launch \
        --master_port 12323 \
        main.py \
        --distributed --multiGPU \
        --seed 2022 \
        --task insurance \
        --batch_size 16 \
        --warmup_prop 0.05 \
        --lr 1e-3 \
        --clip 1.0 \
        --model_type 't5-small' \
        --epoch 20 \
        --gpu '0,1' \
        --logging_step 5000 \
        --insurance_type direct\
        --logging_dir 'pretrain_t5_small_insurance_direct.log' \
```

## single-attribute counterfactually-fair prompt tuning
```
cd prefix
mkdir log
mkdir trained_prefix
```

### insurance direct marital_status attribute
```
python prefix_adversarial.py --gpu 1 --task insurance --P5_pretrained_dir ../pretrain_direct_insurance_t5-small.pt --adversarial_epoch 20 --save_steps 500 --adversarial_logging_steps 1000 --batch_size 16 --rs_step 20 --logging_dir log/insurance/direct_marital.log --unbiased_model_dir trained_prefix/insurance/direct_marital.pt --prefix_lr 1e-4 --dis_lr 5e-5 --together_discriminator_update_steps 10 --rec_update_steps 10 --discriminator_weight 1 --sole_discriminator_update_steps 0 --discriminator_logging_step 10000 --feature user_marital --prefix_length 5 --discriminator_batch_size 16 --from_scratch --insurance_type direct --template_id 5-8 --use_trained_initialization --initialized_prefix_dir trained_prefix/insurance/attentiontuningp5_direct_marital_unbiased_prefix_model_reconly_good.pt --freeze_partial

AUC 0.529783, hits@1 for 5-8 is 0.8234560199625702, hits@3 is 0.9213973799126638, hits@5 is 0.9563318777292577
```
### insurance sequential marital_status attribute
```
python prefix_adversarial.py --gpu 1 --task insurance --P5_pretrained_dir ../pretrain_insurance_t5-small.pt --adversarial_epoch 10 --adversarial_logging_steps 1000 --batch_size 16 --rs_step 10 --logging_dir log/insurance/sequential_marital.log --unbiased_model_dir trained_prefix/insurance/sequential_marital.pt --prefix_lr 1e-4 --dis_lr 5e-5 --together_discriminator_update_steps 10 --rec_update_steps 10 --discriminator_weight 1 --sole_discriminator_update_steps 10 --discriminator_logging_step 10000 --feature user_marital --prefix_length 15 --discriminator_batch_size 16 --from_scratch --template_id 2-1

AUC 0.511286, hits@1 for 2-1 is 0.8113405670283514, hits@3 is 0.9065453272663633, hits@5 is 0.9488974448722436
```

### insurance sequential age attribute
```
python prefix_adversarial.py --gpu 1 --task insurance --P5_pretrained_dir ../pretrain_insurance_t5-small.pt --adversarial_epoch 2 --adversarial_logging_steps 1000 --batch_size 16 --rs_step 20 --logging_dir log/insurance/sequential_age.log --unbiased_model_dir trained_prefix/insurance/sequential_age.pt --prefix_lr 5e-5 --dis_lr 5e-5 --together_discriminator_update_steps 10 --rec_update_steps 10 --discriminator_weight 1 --sole_discriminator_update_steps 0 --discriminator_logging_step 10000 --feature user_age --prefix_length 5 --discriminator_batch_size 16 --from_scratch --template_id 2-1 --use_trained_initialization --initialized_prefix_dir pretrained_trained_prefix/insurance/sequential_age.pt

AUC 0.514246, hits@1 for 2-1 is 0.8139375214555441, hits@3 is 0.9316855475454857, hits@5 is 0.96738757294885
```
### insurance sequential occupation attribute
```
python prefix_adversarial.py --gpu 7 --task insurance --P5_pretrained_dir ../pretrain_insurance_t5-small.pt --adversarial_epoch 5 --adversarial_logging_steps 1000 --batch_size 16 --rs_step 20 --logging_dir log/insurance/sequential_occupation.log --unbiased_model_dir trained_prefix/insurance/sequential_occupation.pt --prefix_lr 1e-4 --dis_lr 5e-5 --together_discriminator_update_steps 10 --rec_update_steps 10 --discriminator_weight 1 --sole_discriminator_update_steps 0 --discriminator_logging_step 10000 --feature user_occupation --prefix_length 30 --discriminator_batch_size 16 --from_scratch --template_id 2-1

AUC 50.82, hits@1 for 2-1 is 0.8261648745519713, hits@3 is 0.9265232974910395, hits@5 is 0.9580645161290322
```

### movielens age attribute
```
python prefix_adversarial.py --task movie --gpu 1 --adversarial_epoch 20 --save_steps 100 --adversarial_logging_steps 10 --batch_size 32 --from_scratch --rs_step 20 --logging_dir log/movie/movie_age.log --unbiased_model_dir trained_prefix/movie/movie_age.pt --prefix_lr 1e-4 --dis_lr 1e-4 --together_discriminator_update_steps 20 --rec_update_steps 15 --discriminator_weight 100 --sole_discriminator_update_steps 10 --feature user_age --initial_discriminator_epoch 5 --discriminator_logging_step 10000 --prefix_length 10

AUC 0.529198, hits@1 for 2-1 is 0.3122516556291391,hits@3 is 0.5117549668874172, hits@5 is 0.5890728476821192, hits@10 is 0.6769867549668874
```

### movielens occupation attribute
```
python prefix_adversarial.py --task movie --gpu 1 --adversarial_epoch 20 --save_steps 100 --adversarial_logging_steps 10 --batch_size 32 --from_scratch --rs_step 20 --logging_dir log/movie/movie_occupation.log --unbiased_model_dir trained_prefix/movie/movie_occupation.pt --prefix_lr 1e-4 --dis_lr 1e-4 --together_discriminator_update_steps 20 --rec_update_steps 15 --discriminator_weight 10 --sole_discriminator_update_steps 10 --feature user_occupation --initial_discriminator_epoch 5 --discriminator_logging_step 10000 --prefix_length 10

AUC 0.518756, hits@1 for 2-1 is 0.3162251655629139,hits@3 is 0.5043046357615895, hits@5 is 0.5826158940397351, hits@10 is 0.6675496688741722
```

## Combine-attribute, training Prompt Mixture
This can only be trained after single-attribute prompts are trained.
```
mkdir combine_trained_prefix

python prefix_combine_adversarial_insurance.py --task insurance --gpu 0 \
		--age_prefix_pretrained_dir your_trained_model_dir \
		--marital_prefix_pretrained_dir your_trained_model_dir \
		--occupation_prefix_pretrained_dir your_trained_model_dir \
		--prefix_length combined_attribute_length \
		--pretrained_dir ../pretrain_insurance_t5-small.pt \
		--age_prefix_length 10 --marital_prefix_length 5 --occupation_prefix_length 30 \
		--adversarial_epoch 10 \
        --prefix_lr 1e-4 \
        --dis_lr 5e-5 \
        --rs_step 10 \
        --rec_update_steps 10 \
        --together_discriminator_update_steps 10 \
        --adversarial_logging_steps 200 \
        --sole_discriminator_update_steps 10 \
        --save_steps 1000 \
        --unbiased_model_dir combine_trained_prefix/insurance/prompt_mixture.pt \
        --logging_dir log/insurance/prompt_mixture.log \
        --from_scratch \
        --discriminator_logging_step 1000
```
