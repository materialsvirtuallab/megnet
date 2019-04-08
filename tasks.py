"""
Pyinvoke tasks.py file for automating releases and admin stuff.

Author: Shyue Ping Ong
"""


from invoke import task
import glob
import os
import json
import webbrowser
import requests
import re
import subprocess
import datetime

from monty.os import cd
from pymatgen import __version__ as CURRENT_VER

NEW_VER = datetime.datetime.today().strftime("%Y.%-m.%-d")


@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def release_github(ctx):
    with open("CHANGES.md") as f:
        contents = f.read()
    toks = re.split(r"\#+", contents)
    desc = toks[1].strip()
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "master",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/megnet/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def update_changelog(ctx):

    output = subprocess.check_output(["git", "log", "--pretty=format:%s",
                                      "v%s..HEAD" % CURRENT_VER])
    lines = ["* " + l for l in output.decode("utf-8").strip().split("\n")]
    with open("CHANGES.rst") as f:
        contents = f.read()
    l = "=========="
    toks = contents.split(l)
    head = "\n\nv%s\n" % NEW_VER + "-" * (len(NEW_VER) + 1) + "\n"
    toks.insert(-1, head + "\n".join(lines))
    with open("CHANGES.rst", "w") as f:
        f.write(toks[0] + l + "".join(toks[1:]))
    ctx.run("open CHANGES.rst")


@task
def release(ctx, notest=False):
    ctx.run("rm -r dist build megnet.egg-info", warn=True)
    if not notest:
        ctx.run("nosetests")
    publish(ctx)
    release_github(ctx)

