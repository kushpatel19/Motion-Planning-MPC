import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.animation import PillowWriter

# System parameters
dt = 0.1  # Sampling time (s)
N = 8    # Prediction horizon
r_obstacle = 1.5  # Obstacle radius (m)
r_robot = 0.2     # Robot radius (m)
wheel_length = 0.2  # Length of each wheel (m)
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

# Trajectory storage
x_traj, u_traj = [x0], []
current_state = x0
total_cost_CBF = 0
i = 0
max_iterations = 200  # Limit on MPC iterations

def rk4_step(f, x, u, dt):
    """
    Perform a single step of RK4 integration.

    Args:
        f (function): The dynamics function.
        x (np.array): Current state.
        u (np.array): Current control input.
        dt (float): Time step.

    Returns:
        np.array: Next state after integration.
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def unicycle_dynamics(x, u):
    """
    Unicycle model dynamics using CasADi symbolics.

    Args:
        x (ca.MX): Current state [x, y, theta].
        u (ca.MX): Control input [v, omega].

    Returns:
        ca.MX: State derivative [dx/dt, dy/dt, dtheta/dt].
    """
    return ca.vertcat(
        u[0] * ca.cos(x[2]),
        u[0] * ca.sin(x[2]),
        u[1]
    )

# MPC loop
gamma = 0.5
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
# Convert CasADi matrices to NumPy arrays when appending to trajectory

# Visualization setup
fig, ax = plt.subplots()
fig.suptitle("Satic Obstacle Avoidance with MPC-CBF", fontsize=16)

ax.set_xlim(-6, 2)
ax.set_ylim(-6, 2)
ax.set_aspect('equal')
obstacle = Circle(obstacle_pos, r_obstacle, color='slategray', alpha=0.5, label="Obstacle")
ax.add_artist(obstacle)
initial_pos = ax.scatter(x0[0], x0[1], color='blue', s=50, label="Initial Position")
goal_pos = ax.scatter(x_ref[0], x_ref[1], color='green', s=50, label="Goal Position")

robot_body = Circle((x0[0], x0[1]), r_robot, color='red', zorder=2, label="Robot")

# Create two wheels with rectangles
wheel_left = Rectangle((0, 0), wheel_length, wheel_width, color='black', zorder=3)
wheel_right = Rectangle((0, 0), wheel_length, wheel_width, color='black', zorder=3)

# Create orientation axes
robot_heading_x = Line2D([], [], color='blue', lw=2)
robot_heading_y = Line2D([], [], color='green', lw=2)

traj_line, = ax.plot([], [], 'b--', label="Trajectory")
ax.legend()
ax.legend(handles=[obstacle, initial_pos, goal_pos, robot_body, traj_line])

ax.add_patch(robot_body)
ax.add_patch(wheel_left)
ax.add_patch(wheel_right)
ax.add_line(robot_heading_x)
ax.add_line(robot_heading_y)

def init():
    """Initialize the animation."""
    robot_body.center = (x0[0], x0[1])
    wheel_left.set_xy((x0[0], x0[1]))
    wheel_right.set_xy((x0[0], x0[1]))
    robot_heading_x.set_data([], [])
    robot_heading_y.set_data([], [])
    traj_line.set_data([], [])
    return robot_body, wheel_left, wheel_right, robot_heading_x, robot_heading_y, traj_line

def update(frame):
    """Update the animation with the current state."""
    x, y, theta = x_traj[frame]
    robot_body.center = (x, y)

    # Wheel positions based on robot orientation
    dx = r_robot * np.sin(theta)  # Offset in y-direction (perpendicular to heading)
    dy = -r_robot * np.cos(theta)  # Offset in x-direction (perpendicular to heading)

    # Update wheel positions
    wheel_left.set_xy((
        x - dx - wheel_length / 2 * np.cos(theta),
        y - dy - wheel_length / 2 * np.sin(theta),
    ))
    wheel_right.set_xy((
        x + dx - wheel_length / 2 * np.cos(theta),
        y + dy - wheel_length / 2 * np.sin(theta),
    ))

    # Update wheel orientations to match robot's heading
    wheel_left.angle = np.degrees(theta)
    wheel_right.angle = np.degrees(theta)

    # Update orientation axes
    axis_length = 0.5
    robot_heading_x.set_data(
        [x, x + axis_length * np.cos(theta)],
        [y, y + axis_length * np.sin(theta)],
    )
    robot_heading_y.set_data(
        [x, x - axis_length * np.sin(theta)],
        [y, y + axis_length * np.cos(theta)],
    )

    # Update trajectory line
    traj_line.set_data([state[0] for state in x_traj[:frame + 1]],
                       [state[1] for state in x_traj[:frame + 1]])
    return robot_body, wheel_left, wheel_right, robot_heading_x, robot_heading_y, traj_line

# Create and display the animation
ani = FuncAnimation(fig, update, frames=len(x_traj), init_func=init, interval=100, blit=True)

# Save the animation as MP4 or GIF using Matplotlib's built-in writer
ani.save("mpc_cbf_static.gif", writer=PillowWriter(fps=10))

plt.show()
