# J-MADRAL

Repository of the J-MADRAL model, presented in the
```
Multi-Aspect Joint Retrieval for E-Commerce: Bridging Product Catalogs and Customer Reviews
```
short paper currently under review at ACM CIKM '26 conference.


## Datasets and Models ##

All the datasets used and every model trained in this work are available through Huggingface:

* `Datasets`
    - `Amazon ESCI`: [https://huggingface.co/datasets/J-MADRAL/AmazonESCI](https://huggingface.co/datasets/J-MADRAL/AmazonESCI)
    - `Search ESCI`: [https://huggingface.co/datasets/J-MADRAL/SearchESCI](https://huggingface.co/datasets/J-MADRAL/SearchESCI)
    - `TREC Product Search 2023`: [https://huggingface.co/datasets/J-MADRAL/TREC_Product_Search_2023](https://huggingface.co/datasets/J-MADRAL/TREC_Product_Search_2023)


* `Data Required for Training`: [https://huggingface.co/datasets/J-MADRAL/TrainingData](https://huggingface.co/datasets/J-MADRAL/TrainingData)


* `Models`
    - `P-BiBERT`: [https://huggingface.co/J-MADRAL/P-BiBERT](https://huggingface.co/J-MADRAL/P-BiBERT)
    - `P-MADRAL`: [https://huggingface.co/J-MADRAL/P-MADRAL](https://huggingface.co/J-MADRAL/P-MADRAL)
    - `R-BiBERT`: [https://huggingface.co/J-MADRAL/R-BiBERT](https://huggingface.co/J-MADRAL/R-BiBERT)
    - `R-MADRAL`: [https://huggingface.co/J-MADRAL/R-MADRAL](https://huggingface.co/J-MADRAL/R-MADRAL)
    - `J-BiBERT`: [https://huggingface.co/J-MADRAL/J-BiBERT](https://huggingface.co/J-MADRAL/J-BiBERT)
    - `J-MADRAL`: [https://huggingface.co/J-MADRAL/J-MADRAL](https://huggingface.co/J-MADRAL/J-MADRAL)


## Re-training the Models ##

It is possible to retrain the models from scratch using the provided scripts in the `scripts` folder.

**Note**: Make sure to set the paths to the required input files at the start of the scripts.


## Python Virtual Environment ###

To create the Python virtual environment required to execute the code, refer to the `environment.yaml` file 
provided in this repository. For reference, these are the main packages required by the code:

```
accelerate        1.12.0
datasets          4.5.0
ir_measures       0.4.3
numpy             2.2.6
pyserini          1.2.0
python            3.10.19
scipy             1.15.3
torch             2.11.0+cu126
torchvision       0.26.0+cu126
transformers      5.7.0
```