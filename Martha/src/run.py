import sys
# import os
import json
# import pandas as pd

sys.path.insert(0, 'src')

# import env_setup
# from etl import get_data
# from features import apply_features

# from model import model_build

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/models')
sys.path.insert(0, 'src/sankey_dash')

from make_dataset import get_data
from process_data import save_cleaned_variables
from train_model import train_and_saved_lda
from prepare_dash import viz_prepare
from launch_dash import run_dashboard


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    # env_setup.make_datadir()
    # env_setup.auth()

    # if 'test' in targets:
    #     with open('config/testdata-params.json') as fh:
    #         data_cfg = json.load(fh)

    #     # make the data target
    #     df_test = pd.DataFrame.from_dict(data_cfg, orient="index").transpose()

    #     print(df_test)

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            etl_cfg = json.load(fh)

        # make the data target
        data = get_data(**etl_cfg)


    if 'process_data' in targets:
        with open('config/process-params.json') as fh:
            process_cfg = json.load(fh)

        save_cleaned_variables(**process_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        train_and_saved_lda(**model_cfg)

    if 'viz_prepare' in targets:
        with open('config/viz-params.json') as fh:
            viz_cfg = json.load(fh)
        viz_prepare(**viz_cfg)

    if 'dashboard' in targets:
        with open('config/dash-params.json') as fh:
            dash_cfg = json.load(fh)
        run_dashboard(**dash_cfg)

    if 'test' in targets:
        with open('config/test-process-params.json') as fh:
            process_cfg = json.load(fh)
        save_cleaned_variables(**process_cfg)

        with open('config/test-model-params.json') as fh:
            model_cfg = json.load(fh)
        train_and_saved_lda(**model_cfg)

        with open('config/test-viz-params.json') as fh:
            viz_cfg = json.load(fh)
        viz_prepare(**viz_cfg)

        with open('config/test-dash-params.json') as fh:
            dash_cfg = json.load(fh)
        run_dashboard(**dash_cfg)

    return 


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
