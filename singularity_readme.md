# Run GraphWorld using Singularity
1. **Build the docker image locally**: Run `scripts/build_local.bat` (Windows) or `scripts/build_local.sh` (Mac/Linux). If you are running the `build_local.sh` script, then run `docker save %tag_name% -o ./graphworld_image.tar` afterwards where `%tag_name` is the tag of the image.
2. **Build .sif file**: Move the `graphworld_image.tar` file to the HPC cluster in the root of the _graphworld_ directory. This can take a couple of minutes as the file is quite large. Hereafter, run `./hpc/build_sif.sh`. This outputs `graphworld.sif` which now can be executed with singularity.
3. **Run HPC**: Finally you can execute the `hpc/run_hpc.job` which executes the `graphworld.sif` in a hpc job.
