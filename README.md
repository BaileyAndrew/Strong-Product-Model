# The Strong Product Model for Network Inference without Independence Assumptions

This is the repository containing the associated code for the paper presented at AISTATS 2025.  TODO: Link to paper, once up on conference website.

Note: To generate the results on real data, you need to actually have the data on your computer ;)  We used two datasets
- [COIL](https://cave.cs.columbia.edu/repository/COIL-20)
- [Cell Cycle](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-2805), in particular you can find an [already-processed version of this dataset used to test the (similar) scBiGLasso algorithm here](https://github.com/luisacutillo78/Scalable_Bigraphical_Lasso/tree/main/CCdata).

Here's a quick overview of the files:

- `print_versions.ipynb`: Shows the packages used for the strong product model code
- `strong_product_model.py`: The code of our actual model
- `utilities.py`: Some additional functions our model relies on
- `synthetic.ipynb`: Code used to generate figures from paper about performance on synthetic data
- `synthetic_rebuttal.ipynb`: During the rebuttal, reviewers asked some interesting questions!  This is the notebook we used to get answers to them; it is admittedly somewhat messy!
- `coil.ipynb`, `cell-cycle.ipynb`: Code used to generate figures from paper about performance on real data
- `test.ipynb`: This is the notebook we prototyped our model in; it's unlikely to be very useful to others
