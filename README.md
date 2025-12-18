# GPU Cluster Health Monitor

This project automates the monitoring of GPU nodes on the UMBC `chip-gpu` cluster. It uses Slurm's `scontrol` to auto-discover node hardware, compares nodes against their peer groups to detect missing GPUs (e.g., a node showing 5 GPUs when its peers have 8), and generates visual snapshots of cluster health.

## 📂 File Structure

* **`generate_gpu_status.py`**
* **Main Code:** Queries `scontrol show node`, parses GPU counts/models, and determines if a node is "DEGRADED" (missing cards), "OVER" (extra cards), or "OK".
* **Output:** Prints a text report to the terminal and saves a visualization (PNG) to the `images/` folder.


* **`run_daily_scan.sh`**
* **Wrapper:** A Bash script designed for `cron`. It sets up the `PATH` (to find `scontrol`) and the Python environment before running the generator. Logs output to `scan.log`.


* **`make_gif.py`**
* **Animator:** Stitches all PNG snapshots in the `images/` folder into a chronological GIF (`cluster_history.gif`) to visualize health over time.


* **`cleanup.py`**
* **Cleanup:** deletes all `.png` and `.gif` files from the root and `images/` directories to reset the history.


* **`images/`**
* Directory where the generator saves individual status snapshots.



---

## 🚀 Quick Access Commands

### 1. Activate Python Environment

To run scripts manually during development:

```bash
conda activate BenchmarkingPythonEnvironment

```

### 2. Manual Scan

To force a status check right now:

```bash
./run_daily_scan.sh

```

### 3. Check Automation Logs

To see if the cron job ran successfully or debug errors:

```bash
tail -f scan.log

```

### 4. Git Workflow

To save changes to the repository:

```bash
git add .
git commit -m "Updated scripts"
git push

```

---

## ⏰ Automation (Crontab)

The automation is currently running on **`chip-login1`**.

**Schedule:** Twice daily at **9:00 AM** and **5:00 PM**.


