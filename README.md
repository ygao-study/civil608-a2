# Multi-Pedestrian Trajectory Prediction using Kalman Filters

This project demonstrates trajectory prediction for multiple pedestrians using individual Kalman filters. Given a short history of observed 2D positions for several pedestrians, the script predicts their future paths for a specified number of frames. It then visualizes:
* The observed historical trajectories.
* The ground truth future trajectories (for comparison).
* The predicted future trajectories generated by the Kalman filters.

## Prerequisites

Before you begin, ensure you have the following installed:
* Python 3.7 or newer
* pip (Python package installer, usually comes with Python)

## Setup and Installation

1.  **Clone the repository or download the files:**
    Make sure you have `civil608_a2.py` and `requirements.txt` in the same project directory.

2.  **Navigate to the project directory:**
    Open your terminal or command prompt and change to the directory where you saved the files.
    ```bash
    cd path/to/your/project_directory
    ```

3.  **Create a virtual environment (Recommended):**
    Using a virtual environment helps manage dependencies and avoids conflicts with other Python projects.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    * On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```
    You should see `(venv)` at the beginning of your terminal prompt.

4.  **Install required packages:**
    Install all necessary libraries using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

Once the setup is complete, you can run the pedestrian trajectory prediction script:

```bash
python civil608_a2.py
```

## Expected output visualization
<image src="expected_output.png" width="500">
