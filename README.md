# Dances with Drifts: Efficient and Effective In-Context Adaptation to Concept Drift for Structured Data

This repository contains the code for the paper "Dances with Drifts: Efficient and Effective In-Context Adaptation to Concept Drift for Structured Data".
<img src="https://github.com/zjiaqi725/FLAIR/blob/main/fig/Overview.png" width="1000">  

## FLAIR Overview

We propose a novel online adaptation framework called FLAIR, which can produce predictions adaptively under different concepts without retraining. Different from the existing approaches, \name presents a new online in-context adaptation paradigm that not only addresses the concept drift challenge inherent in AI-powered databases but also supports various downstream tasks. The key intuition behind \name draws from the in-context learning paradigm of large language models (LLMs), seamlessly integrating `contextual cues' from database environments to dynamically generate outputs that are acutely attuned to the current context.
<p align="center">
<img src="https://github.com/zjiaqi725/FLAIR/blob/main/fig/Framework.png" width="500">  
</p>
As illustrated in the figure, \name realizes in-context adaptation based on two cascaded modules. First, *the task featurization module (TFM)* is a customizable component that extracts informative and task-specific features for subsequent processing. Second, *the dynamic decision engine (DDE)* leverages the extracted features to deliver contextualized predictions given contextual information about the current concept.

## Implementation

#### Environment

Clone the repository locally, and then create a virtual environment:

```bash
- conda create --name FLAIR python=3.8+
- conda activate FLAIR
```

Install the required packages:

```bash
- cd FLAIR
- pip install -r requirements.txt
- python src/init/initialize.py
```

#### Install PostgreSQL (in Linux):

```bash
- cd FLAIR
- wget https://ftp.postgresql.org/pub/source/v13.1/postgresql-13.1.tar.bz2
- tar xvf postgresql-13.1.tar.bz2 && cd postgresql-13.1
- patch -s -p1 < ../pg_modify.patch
- ./configure --prefix=/usr/local/pgsql/13.1 --enable-depend --enable-cassert --enable-debug CFLAGS="-ggdb -O0"
- make -j 64 && sudo make install
- echo 'export PATH=/usr/local/pgsql/13.1/bin:$PATH' >> ~/.bashrc
- echo 'export LD_LIBRARY_PATH=/usr/local/pgsql/13.1/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
- source ~/.bashrc
```

We follow the PostgreSQL setup as [ALECE](https://github.com/pfl-cs/ALECE).

#### Data and workload preparation

* Download the dataset and move it into `./data/STATS/data/`, which can be found in [Benchmark](https://drive.google.com/file/d/1la2GrR0F32GGmKE7TnNujx4K9-esS6wK/view?usp=sharing).
* Download the workload and move it into `./data/STATS/workload/dist_shift_mild/` and `./data/STATS/workload/dist_shift_severe/`, which can be found in [Benchmark](https://drive.google.com/file/d/1la2GrR0F32GGmKE7TnNujx4K9-esS6wK/view?usp=sharing).

#### Offline Training

FLAIR is trained in two stages. In the first stage, the DDE module $\mathcal{M}_{DDE}$ undergoes a one-off meta-training phase across various task distributions.

We have released the [checkpoint file](https://drive.google.com/file/d/1jzbdo3SFrVx9zp954ejdfq9AtncRivb8/view?usp=sharing) for our DDE module, please move it into `./src/MetaDDE/models_diff/`. You can also train your own DDE from desired prior task distributions.

In the second stage, the $\mathcal{M}_{TFM}$ module is trained to extract informative task features that are critical for the specific tasks at hand.
Run the following scripts for two dynamic scenarios: 

```bash
- python main.py --model FLAIR --data STATS --wl_type dist_shift_mild --tfm_train 1
- python main.py --model FLAIR --data STATS --wl_type dist_shift_severe --tfm_train 1
```

#### Online Inference and Adaptation

Once trained, FLAIR is ready for deployment in a real-time environment, performing concurrent online inference and adaptation under evolving concepts.

```bash
- python main.py --model FLAIR --data STATS --wl_type dist_shift_mild --adapt_reg True --stack_size 80
- python main.py --model FLAIR --data STATS --wl_type dist_shift_severe --adapt_reg True --stack_size 80
```

## Acknowledgments

This project is based on the following open-source projects:[ALECE](https://github.com/pfl-cs/ALECE), [TabPFN](https://github.com/automl/TabPFN). Thanks for their great work!
