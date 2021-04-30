import numpy as np
from pyomo.environ import *
from pyomo.dae import *

model = ConcreteModel()
model.x = Var(RangeSet(1, 4), bounds=(1, 25))
model.cons1 = Constraint(rule=lambda model: 40 == model.x[1] ** 2 + model.x[2] ** 2 + model.x[3] ** 2 + model.x[4] ** 2)
model.cons2 = Constraint(rule=lambda model: 25 <= model.x[1] * model.x[2] * model.x[3] * model.x[4])
model.obj = Objective(expr=model.x[1] * model.x[4] * (model.x[1] + model.x[2] + model.x[3]) + model.x[3],
                      sense=minimize)
SolverFactory('ipopt').solve(model)
solutions = [model.x[i]() for i in range(1, 5)]
print("solutins is :" + str(solutions))
