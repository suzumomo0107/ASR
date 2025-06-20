# -*- coding: utf-8 -*-
import re
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from typing import Union

class HParams:
    def __init__(self, path: Union[str, Path]=None):
        if path is None:
            self._configured = False
        else:
            self.configure(path)

    def __getattr__(self, item):
        #if not self.is_configured():
        #    raise AttributeError("HParams not configured yet. Call self.configure()")
        #else:
        if not hasattr(super(), item):
            raise AttributeError(f'HParams does not have "{item}"')
        return super().__getattr__(item)
        #return super().__getattr__(item)

    def configure(self, path: Union[str, Path]):
        if self.is_configured():
            #raise RuntimeError("Cannot reconfigure hparams!")
            print("HParams already configured. Ignoring reconfiguration request.")
            return

        if not isinstance(path, Path):
            path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError("Could not find hparams file {}".format(path))
        elif path.suffix != ".py":
            raise ValueError("`path` must be a python file")

        m = _import_from_file("hparams", path)

        reg = re.compile(r"^__.+__$")
        for name, value in m.__dict__.items():
            if reg.match(name):
                continue
            if name in self.__dict__:
                raise AttributeError(f"module at `path` cannot contain attribute {name} as it "
                    "overwrites an attribute of the same name in utils.hparams")
            self.__setattr__(name, value)

        self._configured = True

    def is_configured(self):
        return self._configured

hparams = HParams()

def _import_from_file(name, path: Path):
    """Programmatically returns a module object from a filepath"""
    if not Path(path).exists():
        raise FileNotFoundError('"%s" doesn\'t exist!' % path)
    spec = spec_from_file_location(name, path)
    if spec is None:
        raise ValueError('could not load module from "%s"' % path)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

