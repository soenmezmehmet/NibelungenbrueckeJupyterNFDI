import os

def check_path_exists(path):
    'Checks if a path exists with an if statement (always checked)'
    
    if not os.path.exists(path):
        raise Exception("Path " + path + " doesn't exist.")

def assert_path_exists(path):
    'Checks if a path exists with an assert (usually turned off in deployment)'
    
    assert os.path.exists(path), "Path " + path + " doesn't exist."

