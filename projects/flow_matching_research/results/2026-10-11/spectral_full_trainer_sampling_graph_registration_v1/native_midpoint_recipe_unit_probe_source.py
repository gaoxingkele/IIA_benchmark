import json
import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

class UnitField(ModelWrapper):
    def __init__(self):
        super().__init__(None)
        self.calls=0
    def forward(self,x,t,**extras):
        self.calls+=1
        return torch.ones_like(x)

torch.set_num_threads(2)
field=UnitField()
solver=ODESolver(field)
grid=torch.linspace(0.,1.-1e-5,500)
result=solver.sample(x_init=torch.zeros(1),time_grid=grid,method='midpoint',step_size=2./500)
assert field.calls==500
assert torch.isclose(result[0],grid[-1],rtol=1e-5,atol=1e-5)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,native_grid_points=len(grid),native_field_calls=field.calls,
    native_midpoint_step_size=2./500,result=float(result[0]),scalar_unit_probe_only=True,
    full_model_or_batch_executed=False,benchmark_performance=False)))
