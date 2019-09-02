import os
from glob import glob
from megnet.models import MEGNetModel


MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../mvl_models')

MODEL_MAPPING = {'Eform_MP_2019': 'mp-2019.4.1/formation_energy.hdf5',
                 'Eform_MP_2018': 'mp-2018.6.1/formation_energy.hdf5',
                 'Efermi_MP_2019': 'mp-2019.4.1/efermi.hdf5',
                 'Bandgap_classifier_MP_2018': 'mp-2018.6.1/band_classification.hdf5',
                 'Bandgap_MP_2018': 'mp-2018.6.1/band_gap_regression.hdf5',
                 'logK_MP_2018': 'mp-2018.6.1/log10K.hdf5',
                 'logG_MP_2018': 'mp-2018.6.1/log10G.hdf5'}

qm9_models = glob(os.path.join(MODEL_PATH, 'qm9-2018.6.1/*.hdf5'))

MODEL_MAPPING.update({'QM9_%s_2018' % i: 'qm9-2018.6.1/%s.hdf5' % i for i in
                      [j.split('/')[-1].split('.')[0] for j in qm9_models]})

for i, j in MODEL_MAPPING.items():
    MODEL_MAPPING[i] = os.path.join(MODEL_PATH, j)

AVAILABLE_MODELS = list(MODEL_MAPPING.keys())


def load_model(model_name):
    """
    load the model by user friendly name as in megnet.utils.models.AVAILABEL_MODELS

    Args:
        model_name: str model name string

    Returns: GraphModel

    """

    if model_name in AVAILABLE_MODELS:
        return MEGNetModel.from_file(MODEL_MAPPING[model_name])
    else:
        raise ValueError('model name %s not in available model list %s' % (model_name, AVAILABLE_MODELS))
