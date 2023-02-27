# Run GraphWorld using Singularity
1. **Build the docker image locally**: Run `scripts/build_local.bat` (Windows) or `scripts/build_local.sh` (Mac/Linux). If you are running the `build_local.sh` script, then run `docker save %tag_name% -o ./graphworld_image.tar` afterwards where `%tag_name` is the tag of the image.
2. **Build .sif file**: Move the `graphworld_image.tar` file to the HPC cluster in the root of the _graphworld_ directory. This can take a couple of minutes as the file is quite large. Hereafter, run `build_sif.sh`.
3. **Run the pipeline**: Finally execute `run_pipeline.sh`. This runs the graphworld pipeline.
