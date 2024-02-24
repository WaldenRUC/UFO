# UFO: a Unified and Flexible Framework for Evaluating Factuality of Large Language Models

## Introduction
Official implementation of our proposed unified and flexible factuality evaluation pipeline.

## Getting Started

You can set up all libraries and dependencies by:
```
pip install -r requirements.txt
```

## Usage

To obtain the dataset statistics as described in our paper, run the `stat.py` script. This will output the necessary statistical data relevant to our research.

```
python stat.py
```

## Obtaining Data of Baseline Evaluation Metrics
To get data from all baseline evaluation metrics, use the `baseline.py` script. 
This will provide you with a comprehensive set of data points derived from the baseline metrics we've studied.
```
python baseline.py
```


## Evaluating in Specific Scenarios

For evaluations in specific scenarios, utilize the `eval.sh` bash script. 
This script should be configured to test under the scenarios defined a given scenario. 
You may need to manually set specific evaluation scenarios, OpenAI's API key, evaluated LLMs, datasets, etc.
```
bash eval.sh
```

## Output

For demonstration, we show 10 samples for each dataset and evaluated LLMs in the directory `output/`.