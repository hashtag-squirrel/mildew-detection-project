# Code adapted from Code Institute's Malaria walkthrough project

from src.data_management import load_pkl_file


def load_test_evaluation(version):
    return load_pkl_file(f'outputs/{version}/evaluation-softmax.pkl')
