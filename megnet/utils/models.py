"""
Model utilities, mainly for model loading and download
"""
import logging
import os
from glob import glob
from zipfile import ZipFile

from megnet.models import MEGNetModel, GraphModel

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CWD = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(CWD, "./mvl_models.zip")
LOCAL_MODEL_PATH = os.path.join(CWD, "./mvl_models")

MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                          '../../mvl_models')

MODEL_MAPPING = {'Eform_MP_2019': 'mp-2019.4.1/formation_energy.hdf5',
                 'Eform_MP_2018': 'mp-2018.6.1/formation_energy.hdf5',
                 'Efermi_MP_2019': 'mp-2019.4.1/efermi.hdf5',
                 'Bandgap_classifier_MP_2018':
                     'mp-2018.6.1/band_classification.hdf5',
                 'Bandgap_MP_2018':
                     'mp-2018.6.1/band_gap_regression.hdf5',
                 'logK_MP_2018': 'mp-2018.6.1/log10K.hdf5',
                 'logG_MP_2018': 'mp-2018.6.1/log10G.hdf5',
                 'logK_MP_2019': 'mp-2019.4.1/log10K.hdf5',
                 'logG_MP_2019': 'mp-2019.4.1/log10G.hdf5'}

qm9_models = glob(os.path.join(MODEL_PATH, 'qm9-2018.6.1/*.hdf5'))

MODEL_MAPPING.update({'QM9_%s_2018' % i: 'qm9-2018.6.1/%s.hdf5' % i for i in
                      [j.split('/')[-1].split('.')[0] for j in qm9_models]})


AVAILABLE_MODELS = list(MODEL_MAPPING.keys())


def load_model(model_name: str) -> GraphModel:
    """
    load the model by user friendly name as in megnet.utils.models.AVAILABEL_MODELS

    Args:
        model_name: str model name string

    Returns: GraphModel

    """

    if model_name in AVAILABLE_MODELS:
        mvl_path = os.path.join(MODEL_PATH, MODEL_MAPPING[model_name])
        if os.path.isfile(mvl_path):
            return MEGNetModel.from_file(mvl_path)

        logger.info("Package-level mvl_models not included, trying "
                    "temperary mvl_models downloads..")
        local_mvl_path = os.path.join(LOCAL_MODEL_PATH, MODEL_MAPPING[model_name])
        if os.path.isfile(local_mvl_path):
            logger.info("Model found in local mvl_models path")
            return MEGNetModel.from_file(local_mvl_path)
        _download_models()
        return load_model(model_name)
    raise ValueError('model name %s not in available model list %s' %
                     (model_name, AVAILABLE_MODELS))


def _download_models(url: str = "https://ndownloader.figshare.com/files/22291785",
                     file_path: str = TEMP_PATH):
    """
    Download machine learning model files

    Args:
        url: (str) url link for the models
    """

    logger.info("Fetching {} from {} to {}".format(
        os.path.basename(file_path), url, file_path))

    import urllib.request

    urllib.request.urlretrieve(url, file_path)

    logger.info("Start extracting models...")
    with ZipFile(file_path, 'r') as zip_obj:
        zip_obj.extractall(os.path.dirname(file_path))
