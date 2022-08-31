from .llff import LLFFDataset
from .blender import BlenderDataset
from .blender import BlenderMVSDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .tankstempleBG import TanksTempleDatasetBG
from .your_own_data import YourOwnDataset
from .scannet import ScannetDataset



dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'TanksAndTempleBG':TanksTempleDatasetBG,
               'nsvf':NSVF,
               'scannet':ScannetDataset,
                'own_data':YourOwnDataset}

mvs_dataset_dict = {'blender': BlenderMVSDataset}