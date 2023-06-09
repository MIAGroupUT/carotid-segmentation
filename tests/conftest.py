def pytest_addoption(parser):
    parser.addoption("--test_similarity", action="store_true", default=False)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.test_similarity
    if "test_similarity" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("test_similarity", [option_value])
