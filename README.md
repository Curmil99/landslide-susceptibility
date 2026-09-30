# Creating a Landslide Susceptibility Map

This project uses machine learning and geospatial environmental data to model landslide susceptibility and generate a landslide susceptibility map.

**Project website:** [GitHub Pages](https://curmil99.github.io/landslide-susceptibility/)  
**Source code:** [GitHub Repository](https://github.com/Curmil99/landslide-susceptibility)

## Description
The project uses landslide and non-landslide point data together with eleven landslide-controlling factors to model spatial landslide susceptibility. The workflow consists of two main parts: `prepare_data.py` prepares the spatial input data and extracts the raster values at the point locations, while `landslide_workflow.ipynb` contains the machine learning workflow.
A Random Forest classifier is used as the main model. The workflow includes hyperparameter optimisation, random and spatial evaluation, model interpretation, and the final application of the trained model to the study area to generate a landslide susceptibility map.

## Getting Started

### Dependencies

The workflow requires Python 3 and the following packages:

- geopandas
- matplotlib
- numpy
- pandas
- rasterio
- scikit-learn

Install the required packages with:

```bash
pip install geopandas matplotlib numpy pandas rasterio scikit-learn
```

A Jupyter-compatible environment is required to run `landslide_workflow.ipynb`, for example Visual Studio Code with the Jupyter extension.

### Data

The input data used in this project are based on the dataset provided by
Samodra et al. (2024), which contains landslide and non-landslide point
data as well as eleven raster datasets representing landslide-controlling
factors.

**Original dataset:**  
[Samodra et al. (2024) – Spatial datasets for benchmarking machine learning-based landslide susceptibility models](https://data.mendeley.com/datasets/vrtx3w6mjd/1)


Due to the size of the raster datasets, the complete input data are not
included directly in this repository. They can be downloaded here:

[Download the input data from HeiBOX] https://heibox.uni-heidelberg.de/d/ea671984d0b74e1cafa8/ 

After downloading, place the input data in the data/raw/ directory. If the directory does not yet exist, create it first.

### Project Structure

The project is organised as follows:

```text
project/
├── README.md
├── Documentation.md
│
├── code/
│   ├── prepare_data.py
│   ├── landslide_workflow.ipynb
│   └── landslide_workflow.pdf
│
└── data/
    ├── raw/
    │   └── [input data]
    ├── processed/
    │   ├── landslide_ml.csv
    │   └── landslide_susceptibility.tif
    └── sample/
        └── [sample data]
```


### Executing the Workflow

The workflow consists of two main steps. Run the following commands from the project root directory.

1. Run `prepare_data.py` to prepare the spatial input data and create the dataset used for model training:

```bash
python code/prepare_data.py
```

This creates `landslide_ml.csv` in the `data/processed/` directory.

2. Open `landslide_workflow.ipynb` in a Jupyter-compatible environment and execute the notebook from top to bottom.

The notebook performs the machine learning workflow, including feature preparation, Random Forest training and optimisation, model evaluation and interpretation, and the final spatial prediction. The resulting landslide susceptibility map is saved as `landslide_susceptibility.tif` in the `data/processed/` directory.
No user interaction is required while the individual calculations are running.




## Authors

Milan Barth

milan.barth@stud.uni-heidelberg.de

## Acknowledgments

Generative AI was used to assist with code development, debugging, and refinement.

DeepL was used for translation and wording



