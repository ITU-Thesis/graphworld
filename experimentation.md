# Info
- The directory /home/data_shares/scara/graphworld/results/mode4/raw/node_classification_[ID]/` is where all the results are being written to, where [ID] is the given ID of the run. This should be changed when we run a new suite of experiments.

# To run experiments:
Step 0: Run `module load singularity` before running anything.
Step 1: Decide what experiment [ID] you are at. You can do this by both checking the `hpc/run_nodeclassification.job` in both beke & daen accounts to see what was run last time, and check the `/home/data_shares/scara/graphworld/results/mode4/raw/node_classification_[ID]/` directories.
Step 2: When you know the ID of your next run, go in to `hpc/run_nodeclassification.job` and change the two lines where it has `node_classification_[ID]`. You then need to increment the [ID] to the corresponding new [ID] that you want to run. The results are then saved in the shared directory.
Step 3: Run the backup script `scripts/perform_backup_[user].bat` on your local machine where [user] is your user. The reason for having individual scripts fpr user is just a lazy solution for now.


# To run the processing
Just change the `RAW_DIR` and `PROCESSED_DIR` in `evaluation/process_results.ipynb` to the correct [ID] which has been ran.


