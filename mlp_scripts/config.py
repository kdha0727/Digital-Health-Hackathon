# ==============================================================================
# Freezes configs to make itself cannot be changed in other modules, by chance.
# By naming each variable with ** CAMEL CASE **,
# just write your configs down in this block:
#

#
# IMPORT
#
import sys
import pathlib

#
# PATH CONFIGS
#

ROOT = pathlib.Path(__file__).resolve().parent.parent

OUTPUT_FILE_PATH = ROOT / 'output.txt'

CHECKPOINT_PATH = ROOT / 'checkpoint'

DATA_PATH = ROOT / 'data'

#
# HYPER PARAMETERS : convertible as json
#

CHANNELS = [2048, 512]  # type: list[int]

BATCH_SIZE = 10  # type: int

NUM_K_FOLD = 20  # type: int

EPOCH_PER_K_FOLD = 100  # type: int

OPTIMIZER = 'Adam'  # type: str

OPTIMIZER_OPTIONS = {

    'lr': 1e-6

}  # type: dict[str, float]

LR_SCHEDULER = 'CosineAnnealingWarmUpRestarts'  # type: str

LR_SCHEDULER_OPTIONS = {

    'T_0': 10, 'T_mult': 2, 'eta_max': 1e-3, 'T_up': 3, 'gamma': 0.5

}  # type: dict[str, [int, float]]

#
# ==============================================================================


# ==============================================================================
# If a particular config needs to be mutable,
# add the name of the particular setting in the following tuple:
#
__mutable_attributes__ = ('ACTIVATION', )
#
# ==============================================================================


# ==============================================================================
# Do not change scripts below:
#
class FrozenConfig(type(sys)):
    __doc__ = __doc__

    __INITIALIZED = False
    __MUTABLE_ATTRIBUTES = __mutable_attributes__[:]

    def __init__(self, name):
        super().__init__(name)
        self.__package__ = __package__
        for k, v in globals().items():
            if k.isupper():
                super().__setattr__(k, v)  # Avoid redundant checking by super()
        self.__INITIALIZED = True

    def __setattr__(self, k, v):
        if self.__INITIALIZED and not k.startswith('_') and k not in self.__MUTABLE_ATTRIBUTES:
            raise TypeError("can't set {!r} attribute".format(k))
        return super().__setattr__(k, v)

    def __delattr__(self, k):
        if self.__INITIALIZED and not k.startswith('_') and k not in self.__MUTABLE_ATTRIBUTES:
            raise TypeError("can't delete {!r} attribute".format(k))
        return super().__delattr__(k)


sys.modules[__name__] = FrozenConfig(__name__)
#
# ==============================================================================
