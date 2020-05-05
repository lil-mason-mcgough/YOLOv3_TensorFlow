# coding: utf-8

import yaml


def _replace_none(mystr):
    return 'null' if mystr == 'None' else mystr

def convert_args_to_yaml(config_in_path, config_out_path):
    with open(config_in_path, 'r') as f_in:
        with open(config_out_path, 'w') as f_out:
            f_out.write('---\n')
            for l_in in f_in:
                if l_in.startswith('### parse some params'):
                    break
                elif l_in.startswith('import ') or l_in.startswith('from '):
                    continue
                elif l_in.startswith('#'):
                    f_out.write(l_in)
                elif len(l_in.strip()) == 0:
                    f_out.write('\n')
                else:
                    var_name, var_value = l_in.split('=')[:2]
                    var_name = var_name.strip()
                    var_value = var_value.split('#')
                    if len(var_value) == 2:
                        var_value, var_comment = var_value
                        var_value = var_value.strip()
                        var_value = _replace_none(var_value)
                        var_comment = var_comment.strip()
                        f_out.write('{}: {} # {}\n'.format(var_name, var_value, var_comment))
                    else:
                        var_value = var_value[0].strip()
                        var_value = _replace_none(var_value)
                        f_out.write('{}: {}\n'.format(var_name, var_value))


if __name__ == '__main__':
    config_in_path = './args.py'
    config_out_path = './data/dewalt_escondido/config.yaml'

    convert_args_to_yaml(config_in_path, config_out_path)

    with open(config_out_path) as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)
        print(yaml_dict)
