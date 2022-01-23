# Copyright (c) 2014  Matthias C. M. Troffaes
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

import re

import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain


def test_plugin_loader():
    """Check that all enumerated plugins can be imported."""
    for group in pybtex.plugin._DEFAULT_PLUGINS:
        for name in pybtex.plugin.enumerate_plugin_names(group):
            pybtex.plugin.find_plugin(group, name)


class TestPlugin1(pybtex.plugin.Plugin):
    pass


class TestPlugin2(pybtex.plugin.Plugin):
    pass


class TestPlugin3(pybtex.plugin.Plugin):
    pass


class TestPlugin4(pybtex.plugin.Plugin):
    pass


def test_register_plugin_1():
    assert pybtex.plugin.register_plugin(
        'pybtex.style.formatting', 'yippikayee', TestPlugin1
    )
    assert pybtex.plugin.find_plugin(
        'pybtex.style.formatting', 'yippikayee'
    ) is TestPlugin1
    assert not pybtex.plugin.register_plugin(
        'pybtex.style.formatting', 'yippikayee', TestPlugin2
    )
    assert pybtex.plugin.find_plugin(
        'pybtex.style.formatting', 'yippikayee'
    ) is TestPlugin1
    assert pybtex.plugin.register_plugin(
        'pybtex.style.formatting', 'yippikayee', TestPlugin2, force=True
    )
    assert pybtex.plugin.find_plugin(
        'pybtex.style.formatting', 'yippikayee'
    ), TestPlugin2


def test_register_plugin_2():
    assert not pybtex.plugin.register_plugin(
        'pybtex.style.formatting', 'plain', TestPlugin2
    )
    plugin = pybtex.plugin.find_plugin('pybtex.style.formatting', 'plain')
    assert plugin is not TestPlugin2
    assert plugin is pybtex.style.formatting.plain.Style


def test_register_plugin_3():
    assert pybtex.plugin.register_plugin(
        'pybtex.style.formatting.suffixes', '.woo', TestPlugin3
    )
    plugin = pybtex.plugin.find_plugin(
        'pybtex.style.formatting', filename='test.woo')
    assert plugin is TestPlugin3


def test_bad_find_plugin():
    with pytest.raises(pybtex.plugin.PluginGroupNotFound):
        pybtex.plugin.find_plugin("pybtex.invalid.group", "__oops")

    with pytest.raises(pybtex.plugin.PluginNotFound) as excinfo:
        pybtex.plugin.find_plugin("pybtex.style.formatting", "__oops")
    assert 'plugin pybtex.style.formatting.__oops not found' in str(excinfo.value)

    with pytest.raises(pybtex.plugin.PluginNotFound):
        pybtex.plugin.find_plugin("pybtex.style.formatting", filename="oh.__oops")


def test_bad_register_plugin():
    with pytest.raises(pybtex.plugin.PluginGroupNotFound):
        pybtex.plugin.register_plugin( "pybtex.invalid.group", "__oops", TestPlugin1)

    with pytest.raises(pybtex.plugin.PluginGroupNotFound):
        pybtex.plugin.register_plugin( "pybtex.invalid.group.suffixes", ".__oops", TestPlugin1)

    # suffixes must start with a dot
    with pytest.raises(ValueError):
        pybtex.plugin.register_plugin( "pybtex.style.formatting.suffixes", "notasuffix", TestPlugin1)


def test_plugin_suffix():
    plugin = pybtex.plugin.find_plugin(
        "pybtex.database.input", filename="test.bib")
    assert plugin is pybtex.database.input.bibtex.Parser


def test_plugin_alias():
    pybtex.plugin._DEFAULT_PLUGINS['pybtex.legacy.input'] = 'punchcard'
    assert pybtex.plugin.register_plugin('pybtex.legacy.input', 'punchcard', TestPlugin4)
    assert pybtex.plugin.register_plugin(
        'pybtex.legacy.input.aliases', 'punchedcard', TestPlugin4
    )
    assert list(pybtex.plugin.enumerate_plugin_names('pybtex.legacy.input')) == ['punchcard']
    plugin = pybtex.plugin.find_plugin("pybtex.legacy.input", 'punchedcard')
    assert plugin is TestPlugin4
    del pybtex.plugin._DEFAULT_PLUGINS['pybtex.legacy.input']


def test_plugin_class():
    """If a plugin class is passed to find_plugin(), it shoud be returned back."""
    plugin = pybtex.plugin.find_plugin("pybtex.database.input", 'bibtex')
    plugin2 = pybtex.plugin.find_plugin("pybtex.database.input", plugin)
    assert plugin == plugin2
