# Copyright (c) 2006-2021  Andrey Golovizin
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


import sys

PY3 = sys.version_info[0] >= 3


if PY3:
    def fix_unicode_literals_in_doctest(obj):
        obj.__doc__ = obj.__doc__.replace("u'", "'")
        return obj

    def python_2_unicode_compatible(obj):
        return obj

    def __str__(obj):
        return obj.__str__()
else:
    def fix_unicode_literals_in_doctest(obj):
        return obj

    def python_2_unicode_compatible(obj):
        obj.__unicode__ = obj.__str__
        del obj.__str__
        return obj

    def __str__(obj):
        return obj.__unicode__()
