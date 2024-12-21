import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import pickle

# System parameters
dt = 0.1  # Sampling time
N = 10    # Prediction horizon
r_obstacle = 1.5  # Obstacle radius
r_robot = 0.1  # Robot radius

# System matrices
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
B = np.array([[0.5 * dt**2, 0],
              [0, 0.5 * dt**2],
              [dt, 0],
              [0, dt]])
n_x = A.shape[1]
n_u = B.shape[1]

# Dynamic obstacle parameters
obstacle_pos = np.array([-2, -3.])  # Initial position
obstacle_vel =  np.array([0.0,0.2])  # Constant velocity

# Initial and target states
x_ref = np.array([0, 0, 0, 0])  # Target state
x0 = np.array([-5, -5, 0, 0])   # Initial state

# Cost weights
Q = np.diag([10, 10, 10, 10])  # State cost
R = np.eye(n_u)              # Control cost
P = 100 * np.eye(n_x)        # Terminal cost

# State and control bounds
x_min = -5
x_max = 5
u_min = -1
u_max = 1

# Trajectory storage
x_traj = [x0]
u_traj = []
obstacle_traj = [obstacle_pos.copy()]  # To store obstacle positions
current_state = x0
total_cost = 0
i = 0

# MPC loop
while np.linalg.norm(current_state[:2] - x_ref[:2]) > 0.1:
    # Update obstacle position
    obstacle_pos += obstacle_vel * dt
    obstacle_traj.append(obstacle_pos.copy())

    # Define decision variables
    X = ca.MX.sym('X', n_x * (N + 1))
    U = ca.MX.sym('U', n_u * N)

    # Objective and constraints
    obj = 0
    g = []
    lbg = []
    ubg = []

    # Initial state constraint
    g.append(X[:n_x] - current_state)
    lbg.extend([0] * n_x)
    ubg.extend([0] * n_x)

    for k in range(N):
        # State and control at step k
        xk = X[k * n_x:(k + 1) * n_x]
        uk = U[k * n_u:(k + 1) * n_u]
        xk_next = X[(k + 1) * n_x:(k + 2) * n_x]

        # Dynamics constraint
        g.append(xk_next - (ca.mtimes(A, xk) + ca.mtimes(B, uk)))
        lbg.extend([0] * n_x)
        ubg.extend([0] * n_x)

        # State constraints
        g.append(xk)
        lbg.extend([x_min] * n_x)
        ubg.extend([x_max] * n_x)

        # Dynamic obstacle avoidance constraint
        obstacle_future_pos = obstacle_pos + obstacle_vel * k * dt
        dist_to_obstacle = ca.sqrt((xk[0] - obstacle_future_pos[0])**2 +
                                   (xk[1] - obstacle_future_pos[1])**2)
        g.append(dist_to_obstacle - (r_obstacle + r_robot))
        lbg.append(0)
        ubg.append(ca.inf)

        # Objective function
        obj += ca.mtimes((xk - x_ref).T, ca.mtimes(Q, (xk - x_ref))) + ca.mtimes(uk.T, ca.mtimes(R, uk))

    # Terminal cost
    obj += ca.mtimes((X[N * n_x:(N + 1) * n_x] - x_ref).T, ca.mtimes(P, (X[N * n_x:(N + 1) * n_x] - x_ref)))

    # Control constraints
    g.append(U)
    lbg.extend([u_min] * N * n_u)
    ubg.extend([u_max] * N * n_u)

    # Solve the optimization problem
    nlp = {'f': obj, 'x': ca.vertcat(X, U), 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
    sol = solver(lbg=lbg, ubg=ubg)

    # Extract solution
    x_sol = sol['x'][:n_x * (N + 1)].full().flatten()
    u_sol = sol['x'][n_x * (N + 1):].full().flatten()

    # Update state
    current_state = x_sol[n_x:2 * n_x]
    x_traj.append(current_state)
    u_traj.append(u_sol[:n_u])
    total_cost += sol['f']

    i += 1
    print(f"Timestep {i}: Current State: {current_state}, Obstacle Position: {obstacle_pos}")

print(f"Total steps: {i}, Total cost: {total_cost}")

# Visualization with animation
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("MPC-DC with Dynamic Obstacles and Live Data", fontsize=16)

# Subplot 1: Trajectory
ax_traj = axes[0, 0]
ax_traj.set_xlim(-6, 2)
ax_traj.set_ylim(-6, 2)
ax_traj.set_aspect('equal')
ax_traj.set_title("Live Trajectory")
ax_traj.set_xlabel("X Position")
ax_traj.set_ylabel("Y Position")
obstacle = plt.Circle(obstacle_traj[0], r_obstacle, color='red', alpha=0.5, label="Obstacle")
ax_traj.add_artist(obstacle)
goal = ax_traj.scatter(x_ref[0], x_ref[1], color='green', label="Goal")
robot = plt.Circle((x0[0], x0[1]), r_robot, color='blue', label="Robot")
ax_traj.add_artist(robot)
traj_line, = ax_traj.plot([], [], 'b--', label="Trajectory")
ax_traj.legend()

# Subplot 2: State evolution
ax_states = axes[0, 1]
ax_states.set_title("State Evolution")
ax_states.set_xlabel("Timestep")
ax_states.set_ylabel("State Values")
lines_states = [ax_states.plot([], [], label=f"x{i+1}")[0] for i in range(n_x)]
ax_states.legend()
ax_states.set_xlim(0, i + 1)
ax_states.set_ylim(x_min - 1, x_max + 1)

# Subplot 3: Control inputs
ax_controls = axes[1, 0]
ax_controls.set_title("Control Inputs")
ax_controls.set_xlabel("Timestep")
ax_controls.set_ylabel("Control Values")
lines_controls = [ax_controls.plot([], [], label=f"u{i+1}")[0] for i in range(n_u)]
ax_controls.legend()
ax_controls.set_xlim(0, i + 1)
ax_controls.set_ylim(u_min - 0.5, u_max + 0.5)

# Subplot 4: Placeholder for future data
axes[1, 1].axis("off")

# Update function for animation
def update(frame):
    # Update robot and trajectory
    robot.center = (x_traj[frame][0], x_traj[frame][1])
    traj_line.set_data([x[0] for x in x_traj[:frame+1]], [x[1] for x in x_traj[:frame+1]])

    # Update obstacle
    obstacle.center = obstacle_traj[frame]

    # Update state plots
    for j, line in enumerate(lines_states):
        line.set_data(range(frame + 1), [x[j] for x in x_traj[:frame+1]])

    # Update control input plots
    if frame > 0:
        for j, line in enumerate(lines_controls):
            line.set_data(range(frame), [u[j] for u in u_traj[:frame]])

    return [robot, traj_line, obstacle, *lines_states, *lines_controls]

# Create animation
ani = FuncAnimation(fig, update, frames=len(x_traj), interval=100, blit=True)

# Save the animation as MP4 or GIF using Matplotlib's built-in writer
ani.save("mpcDC_dynamic_obstacle.gif", writer=PillowWriter(fps=10))

# Show animation
plt.tight_layout()
plt.show()

# # Save trajectory
# trajectory = {'x': x_traj, 'u': u_traj, 'obstacle': obstacle_traj}
# with open(f'mpc_dc_dynamic_obstacle.pkl', 'wb') as file:
#     pickle.dump(trajectory, file)
