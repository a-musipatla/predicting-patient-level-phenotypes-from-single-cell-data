# PREDICTING PATIENT-LEVEL PHENOTYPES FROM SINGLE-CELL DATA

Deep learning approaches have seen increased use in several functional genomics applications, often meeting or exceeding the performance of state-of-the-art methodologies. Over the course of this project, we intend to use a convolutional neural network architecture to predict patient-level phenotypes from B cell mass cytometry data.

## Getting Started

1. Install dependencies using `pip install -r requirements.txt`
    - The `cytoflow` python package requires [SWIG](http://www.swig.org/index.php), which you can install [following these instructions](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/). 
2. Create directory `data/` in the main level of this repository. Use this to store your flow cytometry data. 
