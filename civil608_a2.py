# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# --- Kalman Filter Parameters (Global for all pedestrians) ---
dt = 1.0  # Time step

# State Transition Matrix (F)
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1]])

# Observation Matrix (H)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Process Noise Covariance (Q)
# Tunable: Represents uncertainty in the motion model (e.g., unmodeled accelerations)
Q_diag_factor_pos = 0.05 # variance for position components (increased slightly)
Q_diag_factor_vel = 0.02 # variance for velocity components (increased slightly)
Q = np.diag([Q_diag_factor_pos, Q_diag_factor_pos, Q_diag_factor_vel, Q_diag_factor_vel])

# Measurement Noise Covariance (R)
# Tunable: Represents uncertainty in the measurements
measurement_noise_std_dev = 0.15 # (decreased slightly, assuming more precise measurements)
R = np.array([[measurement_noise_std_dev**2, 0],
              [0, measurement_noise_std_dev**2]])

# --- Input Data for Multiple Pedestrians ---
# Each element in the list is a trajectory for one pedestrian.
# Each trajectory is a NumPy array of (x, y) coordinates for 9 observed frames.

observed_trajectories_all_peds = [
    # Pedestrian 1
    np.array([
        [0.0, 0.0], [0.8, 0.1], [1.9, 0.3], [3.1, 0.7], [4.0, 1.2],
        [5.2, 1.8], [6.0, 2.5], [6.7, 3.3], [7.5, 4.0]
    ]),
    # Pedestrian 2
    np.array([
        [1.0, 5.0], [1.2, 4.5], [1.5, 4.0], [1.9, 3.6], [2.4, 3.3],
        [3.0, 3.1], [3.7, 3.0], [4.5, 2.9], [5.3, 2.8]
    ]),
    # Pedestrian 3
    np.array([
        [10.0, 2.0], [9.5, 2.2], [8.9, 2.5], [8.2, 2.9], [7.5, 3.4],
        [6.8, 4.0], [6.0, 4.5], [5.3, 4.9], [4.5, 5.2]
    ])
]

# Ground Truth Future Trajectories for 12 frames for each pedestrian
# This is what actually happened, for comparison.
ground_truth_future_all_peds = [
    # Pedestrian 1 Future (12 frames)
    np.array([
        [8.3, 4.7], [9.1, 5.4], [10.0, 6.1], [10.8, 6.8], [11.5, 7.5], [12.2, 8.2],
        [12.9, 8.9], [13.6, 9.6], [14.3, 10.3], [15.0, 11.0], [15.7, 11.7], [16.4, 12.4]
    ]),
    # Pedestrian 2 Future (12 frames)
    np.array([
        [6.1, 2.7], [6.9, 2.6], [7.7, 2.5], [8.5, 2.4], [9.3, 2.3], [10.1, 2.2],
        [10.9, 2.1], [11.7, 2.0], [12.5, 1.9], [13.3, 1.8], [14.1, 1.7], [14.9, 1.6]
    ]),
    # Pedestrian 3 Future (12 frames)
    np.array([
        [3.7, 5.5], [2.9, 5.8], [2.1, 6.1], [1.3, 6.4], [0.5, 6.7], [-0.3, 7.0],
        [-1.1, 7.3], [-1.9, 7.6], [-2.7, 7.9], [-3.5, 8.2], [-4.3, 8.5], [-5.1, 8.8]
    ])
]

num_predict_frames = 12

# --- Lists to store results for all pedestrians ---
all_peds_filtered_states_history = []
all_peds_predicted_trajectories = []
all_peds_last_filtered_states = []


# --- Process each pedestrian ---
for ped_idx, observed_trajectory in enumerate(observed_trajectories_all_peds):
    num_observed_frames = observed_trajectory.shape[0]

    # --- Kalman Filter Initialization for current pedestrian ---
    x_hat = np.zeros(4).reshape(4, 1)
    if num_observed_frames > 0:
        x_hat[0] = observed_trajectory[0, 0]
        x_hat[1] = observed_trajectory[0, 1]
        if num_observed_frames > 1:
            x_hat[2] = (observed_trajectory[1, 0] - observed_trajectory[0, 0]) / dt
            x_hat[3] = (observed_trajectory[1, 1] - observed_trajectory[0, 1]) / dt
    P = np.eye(F.shape[0]) * 100.0 # Reset P for each pedestrian

    current_ped_filtered_states = []
    for i in range(num_observed_frames):
        # Prediction Step
        x_hat_minus = F @ x_hat
        P_minus = F @ P @ F.T + Q
        # Update Step
        z = observed_trajectory[i, :].reshape(2, 1)
        innovation_covariance = H @ P_minus @ H.T + R
        K = P_minus @ H.T @ np.linalg.inv(innovation_covariance)
        innovation = z - H @ x_hat_minus
        x_hat = x_hat_minus + K @ innovation
        P = (np.eye(F.shape[0]) - K @ H) @ P_minus
        current_ped_filtered_states.append(x_hat.copy())
    
    all_peds_filtered_states_history.append(current_ped_filtered_states)
    all_peds_last_filtered_states.append(x_hat.copy()) # Store the final state for this ped

    # --- Predict Future Frames for current pedestrian ---
    current_ped_predicted_future = []
    x_hat_predict = x_hat.copy() # Start from the last filtered state
    P_predict = P.copy()       # Start from the last covariance
    for _ in range(num_predict_frames):
        x_hat_predict = F @ x_hat_predict
        P_predict = F @ P_predict @ F.T + Q # Propagate uncertainty (optional for just getting points)
        current_ped_predicted_future.append(x_hat_predict[:2].flatten())
    all_peds_predicted_trajectories.append(np.array(current_ped_predicted_future))


# --- Visualization using Matplotlib ---
plt.figure(figsize=(14, 10))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot for each pedestrian
for ped_idx in range(len(observed_trajectories_all_peds)):
    observed_traj = observed_trajectories_all_peds[ped_idx]
    gt_future_traj = ground_truth_future_all_peds[ped_idx]
    predicted_traj = all_peds_predicted_trajectories[ped_idx]
    
    # Label only for the first pedestrian to avoid legend clutter
    # label_suffix = f" " if ped_idx == 0 else ""
    label_suffix = ''

    # 1. Plot observed historical trajectories (GREY)
    plt.plot(observed_traj[:, 0], observed_traj[:, 1], 'o-', color='grey', linewidth=1.5, markersize=5, alpha=0.8, label="Observed History Trajectories" + label_suffix if ped_idx == 0 else "")
    plt.scatter(observed_traj[0, 0], observed_traj[0, 1], color='grey', marker='D', s=70, edgecolors='black', zorder=3, label="Starting Points (Obs)" + label_suffix if ped_idx == 0 else "") # Start point

    # 2. Plot ground truth future trajectories (GREEN)
    # Combine last observed with first ground truth for a continuous line
    if observed_traj.shape[0] > 0 and gt_future_traj.shape[0] > 0:
        full_gt_path = np.vstack([observed_traj[-1, :], gt_future_traj])
        plt.plot(full_gt_path[:, 0], full_gt_path[:, 1], 's--', color='green', linewidth=1.5, markersize=5, label="Ground Truth Future Trajectories" + label_suffix if ped_idx == 0 else "")
    elif gt_future_traj.shape[0] > 0: # If only future GT is available (unlikely for this setup)
         plt.plot(gt_future_traj[:, 0], gt_future_traj[:, 1], 's--', color='green', linewidth=1.5, markersize=5, label="Ground Truth Future Trajectories" + label_suffix if ped_idx == 0 else "")


    # 3. Plot predicted future trajectories (BLUE)
    if predicted_traj.shape[0] > 0:
        # Combine last observed with first predicted for a continuous line
        if observed_traj.shape[0] > 0 :
            last_obs_pt = observed_traj[-1,:]
            # If filtered states were plotted, use the last filtered point instead:
            # if all_peds_filtered_states_history[ped_idx]:
            #    last_obs_pt = all_peds_filtered_states_history[ped_idx][-1][:2].flatten()
            
            full_pred_path = np.vstack([last_obs_pt, predicted_traj])
            plt.plot(full_pred_path[:, 0], full_pred_path[:, 1], '^-', color='blue', linewidth=1.5, markersize=5, label="Predicted Future Trajectories by Kalman Filter" + label_suffix if ped_idx == 0 else "")
        else: # If no observed history (unlikely for this setup)
            plt.plot(predicted_traj[:, 0], predicted_traj[:, 1], '^-', color='blue', linewidth=1.5, markersize=5, label="Predicted Future Trajectories by Kalman Filter" + label_suffix if ped_idx == 0 else "")

  


plt.xlabel("X Position (meters)", fontsize=14)
plt.ylabel("Y Position (meters)", fontsize=14)
plt.title(f"Multi-Pedestrian Trajectory Prediction ({num_predict_frames}-Frame Ahead)", fontsize=16)
plt.legend(fontsize=9, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.show()
## save the plot as svg file
plt.savefig("multi_pedestrian_trajectory_prediction.svg", format='svg', bbox_inches='tight')

# --- Output some information for the first pedestrian as an example ---
if observed_trajectories_all_peds:
    N = 3
    for ped_idx_to_print in range(N):

        print(f"--- Details for Pedestrian {ped_idx_to_print+1} ---")
        if len(observed_trajectories_all_peds[ped_idx_to_print]) > 0:
            last_obs = observed_trajectories_all_peds[ped_idx_to_print][-1]
            print(f"Last observed point: ({last_obs[0]:.2f}, {last_obs[1]:.2f})")

        # if all_peds_filtered_states_history and len(all_peds_filtered_states_history[ped_idx_to_print]) > 0:
            # last_filt_pos = all_peds_filtered_states_history[ped_idx_to_print][-1][:2].flatten()
            # print(f"Last filtered position: ({last_filt_pos[0]:.2f}, {last_filt_pos[1]:.2f})")

        if all_peds_predicted_trajectories and len(all_peds_predicted_trajectories[ped_idx_to_print]) > 0:
            first_pred = all_peds_predicted_trajectories[ped_idx_to_print][0]
            print(f"First predicted future point: ({first_pred[0]:.2f}, {first_pred[1]:.2f})")
            print("\nPredicted future trajectory (x, y):")
            for i, point in enumerate(all_peds_predicted_trajectories[ped_idx_to_print]):
                print(f"  Frame {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
        else:
            print("\nNo future trajectory was predicted for this pedestrian.")

        if all_peds_last_filtered_states:
            final_state = all_peds_last_filtered_states[ped_idx_to_print].flatten()
            print(f"\nFinal state estimate (x, y, vx, vy) after observations:\n {final_state}")
