from configparser import ConfigParser

VALID_PIPELINES = ["preprocessing", "reconstruction", "tractography"]


def _parse_subjects(config):
    """
    Parse the subjects in the config file.
    If the subjects are not specified, set them to "all".
    If the subjects are specified, split them by comma and space.
    """
    if "subject" not in config["DATASET"]:
        config["DATASET"]["subject"] = "all"
    elif config["DATASET"]["subject"] == "all":
        pass
    else:
        subjects = config["DATASET"]["subject"].split(", ")
        config["DATASET"]["subject"] = subjects

    return config


def _parse_pipeline(config):
    if "PIPELINE" not in config:
        raise ValueError(
            "[PIPELINE] section is missing in the config file. "
            "Specify atleast one pipeline to run. "
            "Available pipelines: preprocessing, reconstruction, "
            "tractography. "
            "Set the ones you want to run to True. "
            "Running tractography will run reconstruction and preprocessing "
            "as well. "
            "Example:\n"
            "[PIPELINE]\n"
            "preprocess = True\n"
            "reconstruction = False\n"
            "tractography = False"
        )
    else:
        for key in config["PIPELINE"]:
            if key not in VALID_PIPELINES:
                raise ValueError(
                    f"Invalid key {key} in [PIPELINE] section. "
                    "Available pipelines: preprocessing, reconstruction, tractography."
                )
            else:
                if config["PIPELINE"][key] == "True":
                    config["PIPELINE"][key] = True
                elif config["PIPELINE"][key] == "False":
                    config["PIPELINE"][key] = False
                else:
                    raise ValueError(
                        f"Invalid value for {key} in [PIPELINE] section. "
                        "Expected True or False."
                    )
        if config["PIPELINE"]["tractography"]:
            config["PIPELINE"]["reconstruction"] = False
            config["PIPELINE"]["preprocessing"] = False


def _parse_config(config_file):
    config = ConfigParser()
    config.read(config_file)
    # convert to dictionary
    config = config._sections
    config = _parse_subjects(config)
    config = _parse_pipeline(config)
    return config


def _select_pipeline(config):
    """
    Setup the pipeline based on the config file.
    This function will parse the config file and create the pipeline.
    """

    # check if the config file is valid
    if not config:
        raise ValueError("Config file is empty or invalid.")

    # check if the output directory exists
    if not os.path.exists(config["OUTPUT"]["cache"]):
        os.makedirs(config["OUTPUT"]["cache"])

    # check if the data directory exists
    if not os.path.exists(config["DATASET"]["directory"]):
        raise ValueError(
            f"Data directory {config['DATASET']['directory']} does not exist."
        )

    to_run = []
    for key, value in config["PIPELINE"].items():
        if value:
            to_run.append(key)

    # check if the pipeline is valid
    if not to_run:
        raise ValueError(
            "No pipeline specified in the config file. "
            "Specify at least one pipeline to run."
            "Example:\n"
            "[PIPELINE]\n"
            "preprocess = True\n"
            "reconstruction = False\n"
            "tractography = False"
        )
    else:
        return to_run


def _run_pipeline(config):
    """
    Run the pipeline based on the config file.
    """

    # pipeline to initialization function mapping
    pipeline_function = {
        "preprocessing": init_preprocess_wf,
        "reconstruction": init_recon_wf,
        "tractography": init_tracto_wf,
    }

    cache_dir = config["OUTPUT"]["cache"]
    for pipeline in to_run:
        # create the pipeline
        wf = pipeline_function[pipeline](
            output_dir=os.path.join(cache_dir, f"{pipeline}_output"),
            config=config,
        )
        wf.write_graph(
            graph2use="flat",
            dotfilename=os.path.join(cache_dir, "graph.dot"),
            format="svg",
        )
        wf.run()


def main():
    """
    Main function to run the diffusion preprocessing pipeline.
    """

    # get the config file path from the command line argument
    import sys
    import os
    import time
    import tempfile
    from diffusion_pipelines.workflows import (
        init_preprocess_wf,
        init_recon_wf,
        init_tracto_wf,
    )

    if len(sys.argv) != 2:
        print(sys.argv)
        print(
            "Usage: dmriprep-tracto <path_to_config_file> or '-' to read from stdin"
        )
        sys.exit(1)

    config_arg = sys.argv[1]

    # If the argument is '-' or if the given file path doesn't exist,
    # assume the config is coming via stdin
    if config_arg == "-" or not os.path.exists(config_arg):
        config_data = sys.stdin.read()
        print(config_data)
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".cfg"
        ) as tmp_file:
            tmp_file.write(config_data)
            tmp_file.flush()  # ensure content is written to disk
            config_arg = tmp_file.name
            print(f"Temporary config file created at {config_arg}")
            # print the contents of the temporary file
            with open(config_arg, "r") as f:
                print(f.read())

    # Create a timestamp in YYYYMMDD_HHMMSS format
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # parse the config file
    config = _parse_config(config_arg)

    _run_pipeline(_select_pipeline(config))
