import pyomo.environ as pyo
from pyomo.dae import *

model = pyo.ConcreteModel()
model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
model.obj = pyo.Objective(expr=2 * model.x[1] ** 2 + 3 * model.x[2] ** 2, sense=1)
model.constraints1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)
pyo.SolverFactory('ipopt').solve(model)
solutions = [model.x[i]() for i in range(1, 3)]
print("solutins is :" + str(solutions))
