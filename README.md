# semopy

**semopy** is an umbrella Python package that includes numerous Structural Equation Modelling (SEM) techniques. 

## Features
  - Write down a model description in a user-friendly syntax
  - Estimate model's parameters using a variety of objective functions
  - Calculate numerous statistics and fit indices
  - Estimate model's parameters in presence of ordinal variables
  - A vast number of settings to fit a researcher's needs
  - Fast and accurate

## Installation
**semopy** is available at PyPi and can be installed by typing the following line into terminal:

`pip install semopy`

## Syntax
To specify SEM models, The **semopy** uses the syntax, which is natural to describe regression models in R. The syntax supports three operator symbols characterising relationships between variables:

- ~ to specify structural part,
- =~ to specify measurement part,
- ~~ to specify common variance between variables.

For example, let a linear equation in the structural part of SEM model take the form:

`y = β1 x1 + β2 x2 + ε` 

Then, in **semopy** syntax it becomes:

`y ~ x1 + x2`

Parameters β1, β2 are to be estimated by **semopy**. In some cases a user might want to fix some of parameters to particular value. For instance, let's assume that we want β1 to stay equal to 2.0 and we are only interested in estimating β2:

`y ~ 2*x1 + x2`


Likewise, if a latent variable η is explained by manifest variables y1, y2, y3, then in **semopy** syntax it can be written down this way:

`eta =~ y1 + y2 + y3`

It is also possible to specify a type of variable. If variable x2 is ordinal, we can inform package about it by using a special operator "is":

`x2 is ordinal`


## Quickstart
For Jupyter notebooks examples with real data see [Examples](/examples) directory.

The pipeline for working with SEM models in **semopy** consists of three steps:
1. Specifying a model
2. Loading a dataset to the model
3. Estimating parameters of the model.

Two main objects required for scpecifying and estimating an SEM model are *Model* and *Optimizer*.

*Model* is responsible for setting up a model from the proposed SEM syntax:
~~~
# The first step
from semopy import Model
mod = """ x1 ~ x2 + x3
          x3 ~ x2 + eta1
          eta1 =~ y1 + y2 + y3
          eta1 ~ x1
      """
model = Model(mod)
~~~
Then a dataset should be provided; at this step the initial values of parameters are calculated:
~~~
# The second step
from pandas import read_csv
data = read_csv("my_data_file.csv", index_col=0)
model.load_dataset(data)
~~~


To estimate parameters of the model an *Optimizer* object should be initialised and estimation executed:
~~~
# The third step
from semopy import Optimizer
opt = Optimizer(model)
objective_function_value = opt.optimize()
~~~

The default objective function for estimating parameters is the likelihood function and the optimisation method is SLSQP (Sequential Least-Squares Quadratic Programming). However, the *semopy* supports a wide range of other objective functions and optimisation schemes being specified as parameters in the *optimize* method.

Finally, user can inspect parameters' estimates:

~~~
from semopy.inspector import inspect
inspect(opt)
~~~

## Documentation
All methods and classes have a docstring provided and a **Sphinx**-generated documentation is also available at [ReadTheDocs](ReadTheDocs.io). For more details see an article.


## Authors

* **Mescheryakov A. Georgy** - *Programmer* - [Herrberg](https://bitbucket.org/herrberg) - undegraduate student, SPbSTU
* **Igolkina A. Anna** - *Supervisor* - [iganna](https://github.com/iganna) - PhD, SPbSTU

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.