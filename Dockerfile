FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

##############################################################################
# ML Utilities
##############################################################################
RUN pip install --no-cache-dir \
    transformers==4.25.1 \
    accelerate>=0.12.0 \
    datasets>=1.8.0 \
    sentencepiece!=0.1.92 \
    evaluate==0.3.0 \
    scipy \
    protobuf==3.20.0 \
    scikit-learn \
    seaborn \
    ipython \
    wandb \
    tqdm \
    azure-datalake-store==0.0.51 \
    azure-storage-queue==12.1.5 \
    mlflow==1.26.0 \
    azureml-mlflow==1.43.0 \
    azureml-dataprep==4.2.2 \
    azureml-dataprep-native==38.0.0 \
    azureml-dataprep-rslex==2.8.1