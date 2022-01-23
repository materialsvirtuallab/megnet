# vim:fileencoding=utf-8

# Copyright (c) 2006-2019  Andrey Golovigin
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from __future__ import unicode_literals

import os
import posixpath
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp

import pytest

from pybtex import errors, io
from .utils import diff, get_data


@contextmanager
def cd_tempdir():
    current_workdir = os.getcwd()
    tempdir = mkdtemp(prefix='pybtex_test_')
    os.chdir(tempdir)
    try:
        yield tempdir
    finally:
        os.chdir(current_workdir)
        rmtree(tempdir)


def copy_file(filename):
    data = get_data(filename)
    with io.open_unicode(filename, 'w') as data_file:
        data_file.write(data)


def copy_files(filenames):
    for filename in filenames:
        copy_file(filename)


def write_aux(aux_name, bib_name, bst_name):
    with io.open_unicode(aux_name, 'w') as aux_file:
        aux_file.write(u'\\citation{*}\n')
        aux_file.write(u'\\bibstyle{{{0}}}\n'.format(bst_name))
        aux_file.write(u'\\bibdata{{{0}}}\n'.format(bib_name))


def group_by_suffix(filenames):
    filenames_by_suffix = dict(
        (posixpath.splitext(filename)[1], filename) for filename in filenames
    )
    return filenames_by_suffix


def check_format_from_string(engine, filenames):
    filenames_by_suffix = group_by_suffix(filenames)
    engine_name = engine.__name__.rsplit('.', 1)[-1]

    if '.aux' in filenames_by_suffix:
        from io import StringIO
        from pybtex import auxfile
        aux_contents = StringIO(get_data(filenames_by_suffix['.aux']))
        auxdata = auxfile.parse_file(aux_contents)
        citations = auxdata.citations
        style = auxdata.style
    else:
        citations = '*'
        style = posixpath.splitext(filenames_by_suffix['.bst'])[0]

    with cd_tempdir():
        copy_file(filenames_by_suffix['.bst'])
        bib_name = posixpath.splitext(filenames_by_suffix['.bib'])[0]
        bib_string = get_data(filenames_by_suffix['.bib'])
        with errors.capture():  # FIXME check error messages
            result = engine.format_from_string(bib_string, style=style, citations=citations)
        correct_result_name = '{0}_{1}.{2}.bbl'.format(bib_name, style, engine_name)
        correct_result = get_data(correct_result_name)
        assert result == correct_result, diff(correct_result, result)


def check_make_bibliography(engine, filenames):
    allowed_exts = {'.bst', '.bib', '.aux'}
    filenames_by_ext = dict(
        (posixpath.splitext(filename)[1], filename) for filename in filenames
    )
    engine_name = engine.__name__.rsplit('.', 1)[-1]

    for ext in filenames_by_ext:
        if ext not in allowed_exts:
            raise ValueError(ext)

    with cd_tempdir():
        copy_files(filenames)
        bib_name = posixpath.splitext(filenames_by_ext['.bib'])[0]
        bst_name = posixpath.splitext(filenames_by_ext['.bst'])[0]
        if '.aux' not in filenames_by_ext:
            write_aux('test.aux', bib_name, bst_name)
            filenames_by_ext['.aux'] = 'test.aux'
        with errors.capture():  # FIXME check error messages
            engine.make_bibliography(filenames_by_ext['.aux'])
        result_name = posixpath.splitext(filenames_by_ext['.aux'])[0] + '.bbl'
        with io.open_unicode(result_name) as result_file:
            result = result_file.read()
        correct_result_name = '{0}_{1}.{2}.bbl'.format(bib_name, bst_name, engine_name)
        correct_result = get_data(correct_result_name)
        assert result == correct_result, diff(correct_result, result)


@pytest.mark.parametrize(
    ["filenames"],
    [
        (['xampl.bib', 'unsrt.bst'],),
        (['xampl.bib', 'plain.bst'],),
        (['xampl.bib', 'alpha.bst'],),
        (['xampl.bib', 'jurabib.bst'],),
        (['cyrillic.bib', 'unsrt.bst'],),
        (['cyrillic.bib', 'alpha.bst'],),
        (['xampl_mixed.bib', 'unsrt_mixed.bst', 'xampl_mixed_unsrt_mixed.aux'],),
        (['IEEEtran.bib', 'IEEEtran.bst', 'IEEEtran.aux'],),
    ]
)
@pytest.mark.parametrize(
    ["check"], [(check_make_bibliography,), (check_format_from_string,)]
)
def test_bibtex_engine(check, filenames):
    from pybtex import bibtex
    check(bibtex, filenames)


@pytest.mark.parametrize(
    ["filenames"],
    [
        (['cyrillic.bib', 'unsrt.bst'],),
        (['cyrillic.bib', 'plain.bst'],),
        (['cyrillic.bib', 'alpha.bst'],),
        (['extrafields.bib', 'unsrt.bst'],),
    ]
)
@pytest.mark.parametrize(
    ["check"], [(check_make_bibliography,), (check_format_from_string,)]
)
def test_pybtex_engine(check, filenames):
    import pybtex
    check(pybtex, filenames)
