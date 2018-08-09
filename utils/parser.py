import argparse
from . import dataloader_utils

def parse_config(config_filepath):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name for this training", type = str)
    parser.add_argument("--color", help="input color, default if RGB", type = str, default = "rgb")
    parser.add_argument("--pretrained", help="input checkpoint path", type = str)
    parser.add_argument("--num_of_stage", help="how many stage of datasets want to use", type = int, default = 1)
    parser.add_argument("--gpu", help="default is yes", type = str, default = "yes")
    parser.add_argument("--train", help="default is yes", type = str, default = "yes")
    parser.add_argument("--lr", help="learning rate, defalut is 1e-4", type = float, default = 1e-4)
    parser.add_argument("--path", help="config file path", type = str)

    parser.add_argument("--start_iteration", help="if you continue training from certain iteration, you should assign a start point", type = int, default = 0 )
    parser.add_argument("--save_img_every_iterations", help="how many iteration between save image", type = int, default = 200 )
    parser.add_argument("--save_each_iteration", help="save model each", type = int, default = 10000 )
    parser.add_argument("-os", help="output stride, default is 16", type = int, default = 16 )
    parser.add_argument("--batch_size", help="batch size, default is 8", type = int, default = 8 )
    parser.add_argument("--subbatch", help="subbatch, default is 1", type = int, default = 1 )
    args = parser.parse_args()

    with open( config_filepath , "r") as input_file:
        task, value = next(iter(input_file)).split('\n')[0].split('=')
        if task != "num_of_stage":
            raise ValueError("first line of config file should be num_of_stage")
        for i in range(int(value)):
            parser.add_argument("--stage{}".format(i), type = str )
            parser.add_argument("--data_argument{}".format(i), type = str)
        args = parser.parse_args()
        setattr(args, task, value)

        for line in input_file:
            if line == '\n':
                continue
            line = line.split('\n')[0]
            task = line.split('=')[0].lower()
            value = ''.join( (str(e) + "=") for e in line.split('=')[1:]).rstrip('=')
            setattr(args, task, value) 

    return args


def get_transform( args_data_argument ):
    compose = []
    funcs = args_data_argument.split(',')
    for func in funcs:
        if func.find('=') == -1:
            compose.append( getattr(dataloader_utils, func)() )
        else:
            fun, param = func.split('=') 
            compose.append( getattr(dataloader_utils, fun)(float(param))  )
    compose = dataloader_utils.Compose(compose)
    return compose