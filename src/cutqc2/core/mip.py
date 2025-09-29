import os
from typing import TYPE_CHECKING

import gurobipy as gp

if TYPE_CHECKING:
    from cutqc2.core.dag import DAGEdge


class MIPCutSearcher:
    def __init__(  # noqa: PLR0913
        self,
        *,
        n_vertices: int,
        edges: list[tuple[int, int]],
        id_to_dag_edge: dict[int, "DAGEdge"],
        num_subcircuit: int,
        max_subcircuit_width: int,
        num_qubits: int,
        max_cuts: int,
    ):
        self.check_graph(n_vertices, edges)

        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.id_to_dag_edge = id_to_dag_edge
        self.num_subcircuit = num_subcircuit
        self.max_subcircuit_width = max_subcircuit_width
        self.num_qubits = num_qubits
        self.max_cuts = max_cuts

        # Count the number of input qubits directly connected to each vertex
        self.vertex_weight = {
            i: edge.weight() for i, edge in self.id_to_dag_edge.items()
        }

        if all(
            env_var in os.environ
            for env_var in (
                "GUROBI_WLSACCESSID",
                "GUROBI_WLSSECRET",
                "GUROBI_LICENSEID",
            )
        ):
            env = gp.Env(
                params={
                    "WLSACCESSID": os.environ["GUROBI_WLSACCESSID"],
                    "WLSSECRET": os.environ["GUROBI_WLSSECRET"],
                    "LICENSEID": int(os.environ["GUROBI_LICENSEID"]),
                }
            )
            self.model = gp.Model(name="cut_searching", env=env)
        else:
            self.model = gp.Model(name="cut_searching")

        self.model.params.OutputFlag = 0  # turning off solver logging

        # all optimization variables except `total_cuts`are indexed by
        # subcircuit as the last index in a multi-level dictionary.
        self.vars = {}
        self.add_variables()

        self.add_constraints()

    def add_variables(self):
        """
        Add all optimization variables to the model.
        We follow the same notation is in section 4.1 of the CutQc paper
        """

        # Total number of cuts - add 0.1 for numerical stability
        self.vars["total_cuts"] = self.model.addVar(
            lb=0, ub=self.max_cuts + 0.1, vtype=gp.GRB.INTEGER
        )

        # Add all other variables in bulk, one per subcircuit
        indexes = range(self.num_subcircuit)  # shorthand

        # Indicator variables - y[v][c]
        #   1 if vertex v is in subcircuit c
        #   0 otherwise
        self.vars["y"] = {}
        for v in range(self.n_vertices):
            self.vars["y"][v] = self.model.addVars(
                indexes, lb=0.0, ub=1.0, vtype=gp.GRB.BINARY
            )

        # Indicator variables - x[e][c]
        #   1 if edge e is cut by subcircuit c
        #   0 otherwise
        self.vars["x"] = {}
        for e in range(self.n_edges):
            self.vars["x"][e] = self.model.addVars(
                indexes, lb=0.0, ub=1.0, vtype=gp.GRB.BINARY
            )

        # Eq (4): Number of original input qubits in each subcircuit
        self.vars["alpha"] = self.model.addVars(
            indexes, lb=0.1, ub=self.max_subcircuit_width, vtype=gp.GRB.INTEGER
        )
        # Eq (5): Number of initialization qubits in each subcircuit
        self.vars["rho"] = self.model.addVars(
            indexes, lb=0, ub=self.max_subcircuit_width, vtype=gp.GRB.INTEGER
        )
        # Eq (6): Number of measurement qubits in each subcircuit
        self.vars["O"] = self.model.addVars(
            indexes, lb=0, ub=self.max_subcircuit_width, vtype=gp.GRB.INTEGER
        )

        self.model.update()

    def add_constraints(self):
        """
        Add all optimization variable constraints to the model.
        We follow the same notation is in section 4.1 of the CutQc paper
        """

        # Eq (4): The number of original input qubits in each subcircuit
        # depends simply on the weight factors (in {0, 1, 2}) for the vertices
        # in the subcircuit.
        self.model.addConstrs(
            self.vars["alpha"][c]
            == gp.quicksum(
                self.vertex_weight[i] * self.vars["y"][i][c]
                for i in range(self.n_vertices)
            )
            for c in range(self.num_subcircuit)
        )

        # Eq (5): A subcircuit requires initialization qubits when, for some
        # edge that is cut, the downstream vertex is in that subcircuit.
        self.model.addConstrs(
            self.vars["rho"][c]
            == gp.quicksum(
                self.vars["x"][i][c] * self.vars["y"][self.edges[i][1]][c]
                for i in range(self.n_edges)
            )
            for c in range(self.num_subcircuit)
        )

        # Eq (6): A subcircuit requires measurement qubits when, for some
        # edge that is cut, the upstream vertex is in that subcircuit.
        self.model.addConstrs(
            self.vars["O"][c]
            == gp.quicksum(
                self.vars["x"][i][c] * self.vars["y"][self.edges[i][0]][c]
                for i in range(self.n_edges)
            )
            for c in range(self.num_subcircuit)
        )

        # Eq (8): Every vertex should be assigned to exactly one subcircuit
        for v in range(self.n_vertices):
            self.model.addConstr(
                gp.quicksum([self.vars["y"][v][i] for i in range(self.num_subcircuit)])
                == 1,
            )

        # Eq (9): The total number of qubits in each subcircuit
        self.model.addConstrs(
            self.vars["alpha"][c] + self.vars["rho"][c] <= self.max_subcircuit_width
            for c in range(self.num_subcircuit)
        )

        # Eq (10): x[e][c] = y[e_a][c] XOR y[e_b][c]
        #   i.e x[e][c] = 1 (edge e is cut by subcircuit c) if and only if one
        #   and only one of the vertices e_a,e_b of edge e is in subcircuit c
        for e in range(self.n_edges):
            e_a, e_b = self.edges[e]
            for c in range(self.num_subcircuit):
                x_e_c = self.vars["x"][e][c]
                y_ea_c = self.vars["y"][e_a][c]
                y_eb_c = self.vars["y"][e_b][c]

                # Eq (11) - linear constraints corresponding to Eq (10)
                self.model.addConstr(x_e_c <= y_ea_c + y_eb_c)
                self.model.addConstr(x_e_c >= y_ea_c - y_eb_c)
                self.model.addConstr(x_e_c >= y_eb_c - y_ea_c)
                self.model.addConstr(x_e_c <= 2 - y_ea_c - y_eb_c)

        # Eq (12): Symmetry-breaking constraint
        #   Vertex v cannot be assigned to subcircuits > v
        for v in range(self.num_subcircuit):
            for c in range(v + 1, self.num_subcircuit):
                self.model.addConstr(self.vars["y"][v][c] == 0)

        # Eq (13): The number of cuts
        self.model.addConstr(
            2 * self.vars["total_cuts"]
            == gp.quicksum(
                [
                    self.vars["x"][i][subcircuit]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                ]
            )
        )

        """
        We cannot model Eq (14) in Gurobi because it is not linear, due to the
        term 2^f(i).

        We choose instead to minimize the total number of cuts while still
        satisfying all the constraints that pertain to
        `self.max_subcircuit_width`
        """
        self.model.setObjective(self.vars["total_cuts"], gp.GRB.MINIMIZE)

        self.model.update()

    def solve(self, threads: int = 48, time_limit: int = 30) -> bool:
        self.model.params.threads = threads
        self.model.Params.TimeLimit = time_limit
        self.model.optimize()

        if self.model.solcount > 0:
            self.subcircuits = []

            for i in range(self.num_subcircuit):
                subcircuit = []
                for j in range(self.n_vertices):
                    if abs(self.vars["y"][j][i].x) > 1e-4:  # noqa: PLR2004
                        subcircuit.append(self.id_to_dag_edge[j])
                self.subcircuits.append(subcircuit)

            # We should have assigned all vertices to some subcircuit
            assert (
                sum([len(subcircuit) for subcircuit in self.subcircuits])
                == self.n_vertices
            )

            cut_edges_idx = []
            self.cut_edges_pairs = []

            for i in range(self.num_subcircuit):
                for j in range(self.n_edges):
                    if abs(self.vars["x"][j][i].x) > 1e-4 and j not in cut_edges_idx:  # noqa: PLR2004
                        cut_edges_idx.append(j)
                        u, v = self.edges[j]

                        self.cut_edges_pairs.append(
                            (self.id_to_dag_edge[u], self.id_to_dag_edge[v])
                        )
            return True
        return False

    @staticmethod
    def check_graph(n_vertices: int, edges: list[int]):
        """
        Check that the incoming vertices and edges are valid.
         1. edges must include all vertices
         2. all (u, v) must be ordered and smaller than n_vertices
        """

        vertices = {i for (i, _) in edges}
        vertices |= {i for (_, i) in edges}
        assert vertices == set(range(n_vertices))
        for u, v in edges:
            assert u < v
            assert u < n_vertices
