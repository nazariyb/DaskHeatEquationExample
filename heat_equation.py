import numpy as np

class HeatEquation:
    def __init__(self, rho : float, capacity : float, coef : float, nodes: int = 10, length: float = 50):
        self.a = coef / (rho * capacity)
        self.nodes = nodes
        self.u = np.zeros((self.nodes, self.nodes, self.nodes))
        
        self.dx = length / self.nodes
        self.dy = length / self.nodes
        self.dz = length / self.nodes

        self.u[1, :, :] = 200
        self.u[-2, :, :] = 200

    def calculate(self, delta_t: float):
        w = self.u.copy()

        for i in range(1, self.nodes - 1):
            for j in range(1, self.nodes - 1):
                for k in range(1, self.nodes - 1):
                    if i == 1 and j == 1 and k == 1:
                        continue

                    dd_ux = (w[i+1, j, k] - 2 * w[i, j, k] + w[i-1, j, k]) / self.dx**2
                    dd_uy = (w[i, j+1, k] - 2 * w[i, j, k] + w[i, j-1, k]) / self.dy**2
                    dd_uz = (w[i, j, k+1] - 2 * w[i, j, k] + w[i, j, k-1]) / self.dz**2

                    derivative = dd_ux + dd_uy + dd_uz
                    
                    self.u[i, j, k] = w[i, j, k] + self.a * delta_t * derivative
                    # print(f"new temp at [{i}, {j}, {k}] = {self.u[i, j, k]}, self.a = {self.a}, delta_t = {delta_t}, derivative = {derivative}")

        return self.u
