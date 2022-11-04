from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

player_model = BayesianNetwork(
    [
        ("C1", "D1"),
        ("C1", "C2"),
        ("C1", "D3"),
        ("C2", "D2"),
        ("D1", "D2"),
        ("D2", "D3")
    ]
)
cpd_c1 = TabularCPD(
    variable="C1",
    variable_card=5,
    values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
)
cpd_c2 = TabularCPD(
    variable="C2",
    variable_card=5,
    values=[[0, 0.25, 0.25, 0.25, 0.25],
            [0.25, 0, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25, 0]],
    evidence=["C1"],
    evidence_card=5
)
cpd_d1 = TabularCPD(
    variable="D1",
    variable_card=5,
    values=[]

)