# Henrik's master thesis
By Henrik Gjestang

## Outline
This repository is split into three parts; code, essay and thesis. In addition to run the experiments it is assumed to exist a folder containing the datasets. The code is split into one directory for each dataset, and additional testing and utils folder for packages etc.
Draft of thesis is found in thesis/build/thesis.pdf

```bash
$ tree -dL 2
.
├── code
│   ├── cifar10
│   ├── hyper-kvasir
│   ├── kvasir
│   ├── kvasir-seg
│   ├── pillcam
│   ├── testing
│   └── utils
├── data
│   ├── cifar10
│   ├── hyper-kvasir
│   ├── kvasir-v2
│   ├── kvasir-v2-unlabeled
│   └── PillCam-Augere
├── essay
│   └── figures
└── thesis
    ├── build
    ├── duo
    ├── figures
    └── ifimaster
```

## Requirements
Everything is run on Ubuntu 18.04 within a conda environment (see code/environment.yml for all dependencies).
