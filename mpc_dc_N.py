
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
x0 = np.array([-5, -5, 0])   # Initial state [x, y, theta]

# Cost weights
Q = np.diag([10, 10, 1])  # State cost
R = np.diag([0.1, 0.1])   # Control cost
P = 100 * np.eye(n_x)     # Terminal cost

# State and control bounds
x_min = np.array([-5, -5, -np.pi])
x_max = np.array([5, 5, np.pi])
u_min = np.array([0, -np.pi])  # Non-negative velocity
u_max = np.array([1, np.pi])

# Prediction horizons
N_values = [7, 15, 30]
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
    total_cost_DC = 0
    max_iterations = 200
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


# Simulate trajectories for all N values
for N in N_values:
    trajectories.append(mpcDC_simulation(N))

# Visualization setup
fig, ax = plt.subplots()
fig.suptitle("Satic Obstacle Avoidance with MPC-DC for Different Prediction horizon", fontsize=10)

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
robot_headings = []
traj_lines = []
colors = ['red', 'orange', 'purple']

for i, (N, color) in enumerate(zip(N_values, colors)):
    robot_body = Circle((x0[0], x0[1]), r_robot, color=colors[i], zorder=2)
    wheel_left = Rectangle((0, 0), wheel_length, wheel_width, color='black', zorder=3)
    wheel_right = Rectangle((0, 0), wheel_length, wheel_width, color='black', zorder=3)
    robot_heading_x = Line2D([], [], color='blue', lw=2)
    robot_heading_y = Line2D([], [], color='green', lw=2)
    traj_line, = ax.plot([], [], f'--', label=f"MPC-DC (N = {N})")

    ax.add_patch(robot_body)
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)
    ax.add_line(robot_heading_x)
    ax.add_line(robot_heading_y)

    robots.append((robot_body, wheel_left, wheel_right))
    robot_headings.append((robot_heading_x, robot_heading_y))
    traj_lines.append(traj_line)

ax.legend()

def init():
    for robot, traj_line in zip(robots, traj_lines):
        robot_body, wheel_left, wheel_right = robot
        robot_body.center = (x0[0], x0[1])
        wheel_left.set_xy((x0[0], x0[1]))
        wheel_right.set_xy((x0[0], x0[1]))
        traj_line.set_data([], [])
    return [item for sublist in robots for item in sublist] + traj_lines

def update(frame):
    updates = []
    for traj, robot, robot_heading, traj_line in zip(trajectories, robots, robot_headings, traj_lines):
        if frame < len(traj):
            x, y, theta = traj[frame]
            robot_body, wheel_left, wheel_right = robot
            robot_heading_x, robot_heading_y = robot_heading

            robot_body.center = (x, y)
            dx = r_robot * np.sin(theta)
            dy = -r_robot * np.cos(theta)

            wheel_left.set_xy((x - dx - wheel_length / 2 * np.cos(theta), y - dy - wheel_length / 2 * np.sin(theta)))
            wheel_right.set_xy((x + dx - wheel_length / 2 * np.cos(theta), y + dy - wheel_length / 2 * np.sin(theta)))
            wheel_left.angle = np.degrees(theta)
            wheel_right.angle = np.degrees(theta)

            axis_length = 0.5
            # Update orientation axes
            robot_heading_x.set_data([x, x + axis_length * np.cos(theta)],
                                     [y, y + axis_length * np.sin(theta)])
            robot_heading_y.set_data([x, x - axis_length * np.sin(theta)],
                                     [y, y + axis_length * np.cos(theta)])

            # Update trajectory line
            traj_line.set_data([state[0] for state in traj[:frame + 1]], 
                               [state[1] for state in traj[:frame + 1]])

            updates.extend([robot_body, wheel_left, wheel_right, robot_heading_x, robot_heading_y, traj_line])
    return updates

# Create and display the animation
ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=100)
# Save the animation as MP4 or GIF using Matplotlib's built-in writer
ani.save("mpc_dc_diff_N.gif", writer=PillowWriter(fps=10))

plt.show()
