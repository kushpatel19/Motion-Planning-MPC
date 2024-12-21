import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.animation import PillowWriter

# System parameters
dt = 0.1  # Sampling time (s)
r_obstacle = 1.5  # Obstacle radius (m)
r_robot = 0.1     # Robot radius (m)
wheel_length = 0.1  # Length of each wheel (m)
wheel_width = 0.05  # Width of each wheel (m)

# Unicycle model: State [x, y, theta], Control [v, omega]
n_x = 3  # Number of states
n_u = 2  # Number of controls

# Obstacle position
obstacle_pos = np.array([-2, -2.25])

# Initial and target states
x_ref = np.array([0, 0, 0])  # Target state [x, y, theta]
x0 = np.array([-5, -5, np.pi/4])   # Initial state [x, y, theta]

# Cost weights
Q = np.diag([10, 10, 1])  # State cost
R = np.diag([0.1, 0.1])   # Control cost
P = 100 * np.eye(n_x)     # Terminal cost


# State and control bounds
x_min = np.array([-5, -5, -np.pi])
x_max = np.array([5, 5, np.pi])
u_min = np.array([0, -np.pi])  # Non-negative velocity
u_max = np.array([1, np.pi])

# Prediction horizons for MPC-DC
N_values = [8]
trajectories = []

def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def unicycle_dynamics(x, u):
    return ca.vertcat(
        u[0] * ca.cos(x[2]),
        u[0] * ca.sin(x[2]),
        u[1]
    )

def mpcDC_simulation(N):
    x_traj, u_traj = [x0], []
    current_state = x0
    max_iterations = 200
    total_cost_DC = 0
    i = 0

    while np.linalg.norm(current_state[:2] - x_ref[:2]) > 0.1 and i < max_iterations:
        X = ca.MX.sym('X', n_x * (N + 1))
        U = ca.MX.sym('U', n_u * N)

        obj = 0
        g = []
        lbg = []
        ubg = []

        g.append(X[:n_x] - current_state)
        lbg.extend([0] * n_x)
        ubg.extend([0] * n_x)

        for k in range(N):
            xk = X[k * n_x:(k + 1) * n_x]
            uk = U[k * n_u:(k + 1) * n_u]
            xk_next = X[(k + 1) * n_x:(k + 2) * n_x]

            xk_next_rk4 = rk4_step(unicycle_dynamics, xk, uk, dt)
            g.append(xk_next - xk_next_rk4)
            lbg.extend([0] * n_x)
            ubg.extend([0] * n_x)

            g.append(xk)
            lbg.extend(x_min)
            ubg.extend(x_max)

            dist_to_obstacle = ca.sqrt((xk[0] - obstacle_pos[0])**2 + (xk[1] - obstacle_pos[1])**2)
            g.append(dist_to_obstacle - (r_obstacle + r_robot))
            lbg.append(0)
            ubg.append(ca.inf)

            obj += ca.mtimes((xk - x_ref).T, ca.mtimes(Q, (xk - x_ref))) + ca.mtimes(uk.T, ca.mtimes(R, uk))

        obj += ca.mtimes((X[N * n_x:(N + 1) * n_x] - x_ref).T, ca.mtimes(P, (X[N * n_x:(N + 1) * n_x] - x_ref)))

        g.append(U)
        lbg.extend(u_min.tolist() * N)
        ubg.extend(u_max.tolist() * N)

        nlp = {'f': obj, 'x': ca.vertcat(X, U), 'g': ca.vertcat(*g)}
        solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
        sol = solver(lbg=lbg, ubg=ubg)

        x_sol = sol['x'][:n_x * (N + 1)].full().flatten()
        u_sol = sol['x'][n_x * (N + 1):].full().flatten()

        current_state = rk4_step(unicycle_dynamics, current_state, u_sol[:n_u], dt)
        x_traj.append(np.array(current_state).flatten())
        u_traj.append(u_sol[:n_u])
        total_cost_DC += sol['f']
        i += 1
        print(f"Timestep {i}: Current State: {current_state}")

    print(f"Total steps: {i}, Total cost(MPC-DC): {total_cost_DC}")
    return x_traj

# Define MPC-CBF simulation
def mpcCBF_simulation(N, gamma):
    x_traj, u_traj = [x0], []
    current_state = x0
    max_iterations = 200
    total_cost_CBF = 0
    i = 0

    while np.linalg.norm(current_state[:2] - x_ref[:2]) > 0.1 and i < max_iterations:
        X = ca.MX.sym('X', n_x * (N + 1))
        U = ca.MX.sym('U', n_u * N)

        obj = 0
        g = []
        lbg = []
        ubg = []

        g.append(X[:n_x] - current_state)
        lbg.extend([0] * n_x)
        ubg.extend([0] * n_x)

        for k in range(N):
            xk = X[k * n_x:(k + 1) * n_x]
            uk = U[k * n_u:(k + 1) * n_u]
            xk_next = X[(k + 1) * n_x:(k + 2) * n_x]

            xk_next_rk4 = rk4_step(unicycle_dynamics, xk, uk, dt)
            g.append(xk_next - xk_next_rk4)
            lbg.extend([0] * n_x)
            ubg.extend([0] * n_x)

            g.append(xk)
            lbg.extend(x_min)
            ubg.extend(x_max)

            # CBF constraint
            h = (xk[0] - obstacle_pos[0])**2 + (xk[1] - obstacle_pos[1])**2 - (r_obstacle + r_robot)**2
            #h_del = 2*(xk[0] - obstacle_pos[0]) * (uk[0] * ca.cos(xk[2])) + 2*(xk[1] - obstacle_pos[1]) * (uk[0] * ca.sin(xk[2]))
            h_next = (xk_next[0] - obstacle_pos[0])**2 + (xk_next[1] - obstacle_pos[1])**2 - (r_obstacle + r_robot)**2

            cbf_constraint = h_next - h + gamma * h
            g.append(cbf_constraint)
            lbg.append(0)
            ubg.append(ca.inf)

            obj += ca.mtimes((xk - x_ref).T, ca.mtimes(Q, (xk - x_ref))) + ca.mtimes(uk.T, ca.mtimes(R, uk))

        obj += ca.mtimes((X[N * n_x:(N + 1) * n_x] - x_ref).T, ca.mtimes(P, (X[N * n_x:(N + 1) * n_x] - x_ref)))

        g.append(U)
        lbg.extend(u_min.tolist() * N)
        ubg.extend(u_max.tolist() * N)

        nlp = {'f': obj, 'x': ca.vertcat(X, U), 'g': ca.vertcat(*g)}
        solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
        sol = solver(lbg=lbg, ubg=ubg)

        x_sol = sol['x'][:n_x * (N + 1)].full().flatten()
        u_sol = sol['x'][n_x * (N + 1):].full().flatten()

        current_state = rk4_step(unicycle_dynamics, current_state, u_sol[:n_u], dt)
        x_traj.append(np.array(current_state).flatten())
        u_traj.append(u_sol[:n_u])
        total_cost_CBF += sol['f']
        i += 1
        print(f"Timestep {i}: Current State: {current_state}")

    print(f"Total steps: {i}, Total cost(MPC-CBF): {total_cost_CBF}")
    return x_traj


# Simulate trajectories for MPC-CBF
gamma_values = [0.1, 0.20, 0.3, 1.0]
for gamma in gamma_values:
    trajectories.append(mpcCBF_simulation(8, gamma))

# Simulate trajectories for MPC-DC
for N in N_values:
    trajectories.append(mpcDC_simulation(N))

# Visualization setup
fig, ax = plt.subplots()
fig.suptitle("Satic Obstacle Avoidance with MPC-CBF for Different γ", fontsize=16)

ax.set_xlim(-6, 2)
ax.set_ylim(-6, 2)
ax.set_aspect('equal')

# Add obstacle
obstacle = Circle(obstacle_pos, r_obstacle, color='slategray', alpha=0.5, label="Obstacle")
ax.add_artist(obstacle)

# Add initial and goal positions
ax.scatter(x0[0], x0[1], color='blue', s=50, label="Initial Position")
ax.scatter(x_ref[0], x_ref[1], color='green', s=50, label="Goal Position")

# Initialize robot visuals
robots = []
traj_lines = []
colors = ['red', 'orange', 'purple', 'cyan', 'magenta']
styles = ['-', '-', '-', '-', '--']  # Dashed for MPC-DC, solid for MPC-CBF

labels = [
    f"MPC-CBF (N=8, γ={gamma})" for gamma in gamma_values
] + [
    f"MPC-DC (N={N})" for N in N_values
]

for color, style, label in zip(colors, styles, labels):
    traj_line, = ax.plot([], [], style, label=label, color=color)
    traj_lines.append(traj_line)

ax.legend()

# Animation
def update(frame):
    updates = []
    for traj, traj_line in zip(trajectories, traj_lines):
        if frame < len(traj):
            traj_line.set_data(
                [state[0] for state in traj[:frame + 1]],
                [state[1] for state in traj[:frame + 1]]
            )
            updates.append(traj_line)
    return updates

# Create and save the animation
ani = FuncAnimation(fig, update, frames=100, blit=True, interval=100)
ani.save("mpc_cbf_gamma.gif", writer=PillowWriter(fps=10))
plt.show()
