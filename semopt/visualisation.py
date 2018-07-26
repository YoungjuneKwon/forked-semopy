import graphviz as gv
from .model import Model


def visualise(model: Model, structural_part=True, measurement_part=False):
    """Visualisation of SEM model via graphviz library.

    Keyword arguments:
    model            -- A SEM model.
    structural_part  -- Should structural part be visualised?
    measurement_part -- Should measurement part be visualised?
    """
    g = gv.Digraph(format='jpg')
    if structural_part:
        g.node_attr.update(color='red', shape='box')
        for i, j in model.parameters['Beta']:
            lval, rval = model.beta_names[0][i], model.beta_names[0][j]
            g.edge(rval, lval)
    if measurement_part:
        g.node_attr.update(color='black', shape='circle')
        for i, j in model.parameters['Lambda']:
            lval, rval = model.lambda_names[0][i], model.lambda_names[0][j]
            g.edge(lval, rval)
    g.render(view=True)
        