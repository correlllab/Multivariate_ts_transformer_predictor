import os, sys, yaml, re
sys.path.insert(1, os.path.realpath('../'))

class YamlLoader():
    def __init__(self) -> None:
        # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
        self.loader = yaml.SafeLoader
        self.loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.')
        )


    def load_yaml(self, path: str = None) -> dict:
        with open(path, 'r') as config_file:
            try:
                conf = yaml.safe_load(config_file)
                return conf
            except yaml.YAMLError as e:
                print('ERROR:' + e)

