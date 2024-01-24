
---
language: en
tags:
- sagemaker
- bart
- summarization
license: apache-2.0
datasets:
- samsum
model-index:
- name: bart-large-cnn-samsum
  results:
  - task: 
      name: Abstractive Text Summarization
      type: abstractive-text-summarization
    dataset:
      name: "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization" 
      type: samsum
    metrics:
       - name: Validation ROGUE-1
         type: rogue-1
         value: 42.621
       - name: Validation ROGUE-2
         type: rogue-2
         value: 21.9825
       - name: Validation ROGUE-L
         type: rogue-l
         value: 33.034
       - name: Test ROGUE-1
         type: rogue-1
         value: 41.3174
       - name: Test ROGUE-2
         type: rogue-2
         value: 20.8716
       - name: Test ROGUE-L
         type: rogue-l
         value: 32.1337
widget:
- text: | 
    Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
    Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
    Jeff: ok.
    Jeff: and how can I get started? 
    Jeff: where can I find documentation? 
    Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face 
---
## `bart-large-cnn-samsum`
This model was trained using Amazon SageMaker and the new Hugging Face Deep Learning container.
For more information look at:
- [🤗 Transformers Documentation: Amazon SageMaker](https://huggingface.co/transformers/sagemaker.html)
- [Example Notebooks](https://github.com/huggingface/notebooks/tree/master/sagemaker)
- [Amazon SageMaker documentation for Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)
- [Python SDK SageMaker documentation for Hugging Face](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)
- [Deep Learning Container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers)
## Hyperparameters
    {
    "dataset_name": "samsum",
    "do_eval": true,
    "do_predict": true,
    "do_train": true,
    "fp16": true,
    "learning_rate": 5e-05,
    "model_name_or_path": "facebook/bart-large-cnn",
    "num_train_epochs": 3,
    "output_dir": "/opt/ml/model",
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "predict_with_generate": true,
    "sagemaker_container_log_level": 20,
    "sagemaker_job_name": "huggingface-pytorch-training-2024-01-24-01-02-41-566",
    "sagemaker_program": "run_summarization.py",
    "sagemaker_region": "us-east-1",
    "sagemaker_submit_directory": "s3://sagemaker-us-east-1-802575742115/huggingface-pytorch-training-2024-01-24-01-02-41-566/source/sourcedir.tar.gz",
    "seed": 7
}
## Usage
    from transformers import pipeline
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    conversation = '''Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
    Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
    Jeff: ok.
    Jeff: and how can I get started? 
    Jeff: where can I find documentation? 
    Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face                                           
    '''
    summarizer(conversation)
## Results
| key | value |
| --- | ----- |
| eval_rouge1 | 43.1754 |
| eval_rouge2 | 22.2026 |
| eval_rougeL | 33.6383 |
| eval_rougeLsum | 40.1594 |
