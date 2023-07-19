# Configuration files

## Configuration of transforms

Values of the parameters of the transforms can be set in an external TOML file.
This TOML file is organized as follows:

```toml
[transform_name]
parameter_name = "value"
```

All parameters have a default value, that can be found in the TOML file at 
[src/carotid/utils/default_args.toml](https://github.com/MIAGroupUT/carotid-segmentation/blob/main/src/carotid/utils/default_args.toml).

You can copy-paste this TOML file and modify it to change the values of the parameters. Make sure that
the type of the parameters is consistent with the documentation.

!!! note "Incomplete definition"
    If the value of one parameter is not set, it will be automatically set to its default value.

## Configuration of training procedures

Similarly to the transforms, the training procedure can be parametrized with a TOML file with the same
organization:

```toml
[transform_name]
parameter_name = "value"
```

All parameters have a default value, that can be found in the TOML file at 
[src/carotid/utils/default_train.toml](https://github.com/MIAGroupUT/carotid-segmentation/blob/main/src/carotid/utils/default_train.toml).

You can copy-paste this TOML file and modify it to change the values of the parameters. Make sure that
the type of the parameters is consistent with the documentation.

!!! note "Incomplete definition"
    If the value of one parameter is not set, it will be automatically set to its default value.

