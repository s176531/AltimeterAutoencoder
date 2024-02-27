# Requirements

The required packages are all included in the *grid* conda environment used in the grid interpolation. The environment.yml file may be found in the AltimeterGridding repository.

In order to utilize the program in this repository, a data set must be provided. The idea is to use the grid data set produced using the AltimeterGridding pipeline, however, both MEaSUREs and CMEMS data should be usable. The 5 day gap between gridded days in MEaSUREs may provide issues, which are not accounted for.

# Running the code

The main program is executed using the [train.py](train.py) file. This file contains all the necessary settings for running the autoencoder. The code is currently set up to run a single combined netcdf file containing all dates. This may be changed in the *load_data* function within [data_setup.py](src/data_setup.py). The model can automatically be saved by setting the **save_epoch** parameter in [train.py](train.py). These savepoints may be loaded using the *load_model* function from [save_load_model.py](src/save_load_model.py) for either further training or use as is.
