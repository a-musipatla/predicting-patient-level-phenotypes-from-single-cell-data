# PREDICTING PATIENT-LEVEL PHENOTYPES FROM SINGLE-CELL DATA

Deep learning approaches have seen increased use in several functional genomics applications, often meeting or exceeding the performance of state-of-the-art methodologies. Over the course of this project, we intend to use a convolutional neural network architecture to predict patient-level phenotypes from B cell mass cytometry data.

## Getting Started

1. Install dependencies using `pip install -r requirements.txt`
    - The `cytoflow` python package requires [SWIG](http://www.swig.org/index.php), which you can install [following these instructions](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/). 
2. Create directory `data/` in the main level of this repository. Use this to store your flow cytometry data. 

## Running the Models

1. Running `python scripts/dnn/bcell_driver.py` will load and run a DNN on the B-cell cytometry data. Command line flags:
    - `-v`: Full output verbosity
    - `-p`: Display all plots

## Additional Resources
Selected additional resources and documentation to assist in running this code.
- Flow Cytometry Analysis
    - [What is Flow Cytometry?](https://www.antibodies-online.com/resources/17/1247/what-is-flow-cytometry-facs-analysis/)
    - [A Beginner’s Guide To Analyzing and Visualizing Mass Cytometry Data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5765874/)
    - [Interpreting flow cytometry data: a guide for the perplexed](http://depts.washington.edu/flowlab/Cell%20Analysis%20Facility/Interpreting%20Flow%20Data.pdf)
    - Packages
        - [`cytoflow` documentation](https://cytoflow.readthedocs.io/en/stable/)
        - [`FlowCytometryTools` documentation](https://eyurtsev.github.io/FlowCytometryTools/)

