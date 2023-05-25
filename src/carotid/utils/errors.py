class MissingRawArgException(Exception):
    pass


class MissingProcessedObjException(Exception):
    pass


class TransformAlreadyRun(Exception):
    def __init__(self, transform_name, output_dir):
        message = f"Transform {transform_name} was already performed in {output_dir}.\n" \
                  f"Use --force option to overwrite it."
        super().__init__(message)


class NoValidSlice(Exception):
    pass


class InvalidArgException(Exception):
    pass
