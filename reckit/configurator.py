import os
import sys
from configparser import ConfigParser
from collections import OrderedDict

class Configurator(object):
    def __init__(self, root_dir, data_dir):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self._sections = OrderedDict()
        self._cmd_args = OrderedDict()

    # Read and add config from ini-style file.
    def add_config(self, cfg_file, section="default"):
        # 初始化文件不存在
        if not os.path.isfile(cfg_file):
            raise FileNotFoundError("File '%s' does not exist." % cfg_file)

        # 解析配置文件
        configparser = ConfigParser()
        configparser.optionxform = str
        configparser.read(cfg_file, encoding="utf-8")
        sections = configparser.sections()

        config_sec = None
        if len(sections) == 0:
            raise ValueError("'%s' is empty!" % cfg_file)
        elif len(sections) == 1:
            config_sec = sections[0]
        elif section in sections:
            config_sec = section
        else:
            raise ValueError("'%s' has more than one sections but there is no "
                             "section named '%s'" % (cfg_file, section))

        sec_name = "%s:%s" % (os.path.basename(cfg_file).split(".")[0], config_sec)
        if sec_name in self._sections:
            sec_name += "_%d" % len(self._sections)

        config_arg = OrderedDict(configparser[config_sec].items())
        # 从命令行更新参数
        for arg in self._cmd_args:
            if arg in config_arg:
                config_arg[arg] = self._cmd_args[arg]
        self._sections[sec_name] = config_arg

    def parse_cmd(self):
        args = sys.argv[1:]
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--"):
                    raise SyntaxError("Commend arg must start with '--', but '%s' is not!" % arg)
                arg_name, arg_value = arg[2:].split("=")
                self._cmd_args[arg_name] = arg_value

        # 覆盖配置文件的参数
        for sec_name, sec_arg in self._sections.items():
            for cmd_argn, cmd_argv in self._cmd_args.items():
                if cmd_argn in sec_arg:
                    sec_arg[cmd_argn] = cmd_argv
    
    def summarize(self):
        """Get a summary of the configurator's arguments.
        
        Returns:
            str: A string summary of arguments.
        """
        if len(self._sections) == 0:
            raise ValueError("Configurator is empty.")
        
        args = self._sections[next(reversed(self._sections.keys()))]
        params_id = '_'.join(["{}={}".format(arg, value) for arg, value in args.items() if len(value) < 20])
        special_char = {'/', '\\', '\"', ':', '*', '?', '<', '>', '|', '\t', '\n', '\r', '\v', ' '}
        params_id = [c if c not in special_char else '_' for c in params_id]
        params_id = ''.join(params_id)
        return params_id

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Index must be a str.")
        for sec_name, sec_args in self._sections.items():
            if item in sec_args:
                param = sec_args[item]
                break
        else:
            if item in self._cmd_args:
                param = self._cmd_args[item]
            else:
                raise KeyError("There are not the argument named '%s'" % item)

        # convert param from str to value, i.e. int, float or list etc.
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, bool, None.__class__)):
                value = param
        except (NameError, SyntaxError):
            if param.lower() == "true":
                value = True
            elif param.lower() == "false":
                value = False
            else:
                value = param

        return value

    def __getattr__(self, item):
        return self[item]

    def __contains__(self, o):
        for sec_name, sec_args in self._sections.items():
            if o in sec_args:
                flag = True
                break
        else:
            if o in self._cmd_args:
                flag = True
            else:
                flag = False

        return flag

    def __str__(self):
        sec_str = []

        # ini files
        for sec_name, sec_args in self._sections.items():
            arg_info = '\n'.join(["{}={}".format(arg, value) for arg, value in sec_args.items()])
            arg_info = "%s:\n%s" % (sec_name, arg_info)
            sec_str.append(arg_info)

        # cmd
        if self._cmd_args:
            cmd_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self._cmd_args.items()])
            cmd_info = "Command line:\n%s" % cmd_info
            sec_str.append(cmd_info)

        info = '\n\n'.join(sec_str)
        return info

    def __repr__(self):
        return self.__str__()
