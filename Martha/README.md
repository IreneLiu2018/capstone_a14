# HDSI Faculty Allocation Tool

This repository contains the files necessary to run the Version 1.0 of the HDSI Sankey Diagram tool that was created using Latent Dirichlet Allocation
on faculty's publications abstracts.
The run.py file is in the src folder along with the test data and LDA.py model </br>
*It is necessary to cd into src to be able to run the following:*

- To obtain the preprocessed data file, run python run.py process_data
- To obtain the trained lda model, run python run.py model
- To prepare the dashboard, run python run.py viz_prepare
- To run the dashboard, run python run.py dashboard
