# %% 
import sys
from utils_ccy.util.directory import import_class

# !git clone -q https://github.com/CalciferZh/AMCParser
sys.path.append('AMCParser')
import amc_parser as amc
import argparse
import yaml

# %%
def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Missing Marker Reconstruction')
    parser.add_argument(
        '--config',
        default='./config/CMU/asfamc.yaml',
        help='path to the default configuration file') # config 파일이 위치하는 디렉토리 
    
    parser.add_argument(
        '--num-joint', type=int, default=31, help='Origin Joint 수') # The number of total joint 
    parser.add_argument(
        '--node-mask',
        default=None,
        help='사용할 노드 리스트') # 사용할 노드 
    parser.add_argument(
        '--max-frame', type=int, default=100, help='데이터의 Frame 길이') # 한 데이터의 Frame 길이  
    parser.add_argument(
        '--raw-data-name', default='CMU', choices=['CMU'], help='Data name') # Raw data의 데이터 이름.
    parser.add_argument(
        '--raw-data-type', default='asfamc', choices=['asfamc', 'c3d', 'bvh'], help='Data Type') # Raw data의 Type.
    parser.add_argument(
        '--raw-data-dir',
        default='./data/CMU/asfamc/',
        help='Raw Data의 경로.') # Raw Data Directory
    parser.add_argument(
        '--missing-args',
        default=dict(),
        help='data missing 관련 argments') # data missing 관련 argments
    parser.add_argument(
        '--save-dir',
        default='./processed_data/CMU/asfamc/',
        help='전처리된 데이터를 저장할 경로') # Raw Data Directory
    parser.add_argument(
        '--pre-processor', default='preprocessor.asfamc.Processor', help='데이터 Processing을 진행할 Python Class') # Preprocessing Method 
    # visulize and debug
    parser.add_argument(
        '--debug',
        default=False,
        help='print logging or not') # Log 출력할지에 대한 여부
    '''
    parser.add_argument(
        '--print-log',
        default=True,
        help='print logging or not') # Log 출력할지에 대한 여부
    '''
    return parser


if __name__ == "__main__":
    ############## parser에 들어있는 argument -> dictionary ##############
    # argument 생성
    parser = get_parser()

    # load arg from config file
    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f) # config 파일에 들어있는 keys. config key로 명명

        # 예외 처리
        key = vars(p).keys() # parser에 들어있는 key값. default key로 명명 
        for k in default_arg.keys():
            if k not in key: # config key가 default key에 들어있지 않으면 
                print('WRONG ARG: {}'.format(k)) # default key에 해당 키가 없음을 알림.
                assert (k in key) 

        parser.set_defaults(**default_arg) # 임의의 개수의 Keyword arguments를 받아서 default key -> config key로 변경

    arg = parser.parse_args() # config key가 반영된 argument

    Processor = import_class(arg.pre_processor)
    processor = Processor(arg)
    processor.start()
    

