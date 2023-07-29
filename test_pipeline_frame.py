import pathlib

from pipeline_dummy import load_data_framewise
import pdb

def test_run():
    pipeline = load_data_framewise('train.csv', pathlib.Path(""))
    output = list(pipeline)
    pdb.set_trace()
    assert True