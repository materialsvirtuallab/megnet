import glob
import os
import re
from functools import lru_cache

import tensorflow as tf
from flask import Flask, make_response, render_template, request
from flask.json import jsonify
from pymatgen import MPRester, Structure

from megnet.models import MEGNetModel

app = Flask(__name__)
mpr = MPRester(os.environ.get("MAPI_KEY"))
models = None
graph = None


def html_formula(f):
    return re.sub(r"([\d.]+)", r"<sub>\1</sub>", f)


def init():
    global models, graph
    graph = tf.get_default_graph()
    models = {}
    # load the pre-trained Keras model
    for k in glob.glob("models/*.hdf5"):
        if "efermi" not in k:
            name = k.split("/")[1]
            name = name.split(".")[0]
            models[name] = MEGNetModel.from_file(k)


def predict(model_name, structure):
    try:
        model = models[model_name]
        with graph.as_default():
            return model.predict_structure(structure).ravel()
    except Exception:
        return float("nan")


def get_results(structure):
    return {
        "formation_energy": predict("formation_energy", structure),
        "band_gap": predict("band_gap", structure),
        "K": 10 ** predict("log10K", structure),
        "G": 10 ** predict("log10G", structure),
        "efermi": predict("efermi", structure),
    }


@lru_cache(maxsize=64)
def get_mp_results(mp_id):
    data = mpr.query(
        {"task_id": mp_id},
        properties=[
            "structure",
            "formation_energy_per_atom",
            "band_gap",
            "efermi",
            "elasticity.K_VRH",
            "elasticity.G_VRH",
        ],
        mp_decode=False,
    )
    data = data[0]
    s = Structure.from_dict(data["structure"])
    formula = s.composition.reduced_formula
    results = get_results(s)
    results["mp_formation_energy"] = data["formation_energy_per_atom"]
    results["mp_band_gap"] = data["band_gap"]
    results["mp_K"] = data.get("elasticity.K_VRH", "--")
    results["mp_G"] = data.get("elasticity.G_VRH", "--")
    results["mp_efermi"] = data.get("efermi", "--")
    return formula, results


@app.route("/")
def index():
    return make_response(render_template("index.html"))


@app.route("/models")
def get_models():
    return jsonify(list(models.keys()))


@app.route("/query", methods=["GET"])
def query():
    message = ""
    try:
        mp_id = request.args.get("mp_id")
        formula, results = get_mp_results(mp_id)
    except Exception:
        message = "Please check your Materials Project Id."
        formula = ""
        results = []
    return make_response(
        render_template("index.html", mp_id=mp_id, formula=html_formula(formula), message=message, mp_results=results)
    )


@app.route("/query_structure", methods=["POST"])
def query_structure():
    formula = ""
    message = ""
    results = []
    try:
        structure_string = request.form.get("structure_string")
        fmt = "POSCAR"
        if "data_" in structure_string:
            fmt = "CIF"
        try:
            s = Structure.from_str(structure_string, fmt=fmt)
        except:
            s = Structure.from_str(structure_string, fmt="POSCAR")

        formula = s.composition.reduced_formula
        results = get_results(s)
    except Exception as ex:
        message = "Error reading structure! %s" % (str(ex))
    return make_response(
        render_template(
            "index.html",
            structure_string=structure_string,
            formula=html_formula(formula),
            structure_results=results,
            message=message,
        )
    )


@app.route("/rest/predict_structure/<string:model_name>", methods=["POST"])
def predict_structure_rest(model_name):
    try:
        structure = Structure.from_str(request.form["structure"], fmt=request.form["fmt"])
        val = predict(model_name, structure)
        d = {"model": model_name, "val": float(val), "formula": structure.composition.reduced_formula}
        return jsonify(d)
    except Exception as ex:
        return jsonify(str(ex))


@app.route("/rest/predict_mp/<string:model_name>/<string:mp_id>")
def predict_mp_rest(model_name, mp_id):
    try:
        structure = mpr.get_structure_by_material_id(mp_id)
        val = predict(model_name, structure)
        d = {"model": model_name, "val": float(val), "formula": structure.composition.reduced_formula}
        return jsonify(d)
    except Exception as ex:
        return jsonify(str(ex))


@app.route("/predict_structure/<string:model_name>", methods=["POST"])
def predict_structure(model_name):
    try:
        structure = Structure.from_str(request.form["structure"], fmt=request.form["fmt"])
        val = predict(model_name, structure)
        return jsonify(float(val))
    except Exception as ex:
        return jsonify(str(ex))


@app.route("/predict_mp/<string:model_name>/<string:mp_id>")
def predict_mp(model_name, mp_id):
    try:
        structure = mpr.get_structure_by_material_id(mp_id)
        val = predict(model_name, structure)
        return jsonify(float(val))
    except Exception as ex:
        return jsonify(str(ex))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""Basic web app for MEGNet.""", epilog="Authors: Chi Chen, Shyue Ping Ong"
    )

    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="Whether to run in debug mode.")
    parser.add_argument(
        "-hh",
        "--host",
        dest="host",
        type=str,
        nargs="?",
        default="0.0.0.0",
        help="Host in which to run the server. Defaults to 0.0.0.0.",
    )
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        type=int,
        nargs="?",
        default=5000,
        help="Port in which to run the server. Defaults to 5000.",
    )

    args = parser.parse_args()
    init()
    print("Loading models... please wait until server has fully started")
    app.run(threaded=True, debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
