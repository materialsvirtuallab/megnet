# -*- coding: utf-8 -*-
"""
    LaTeX Codec
    ~~~~~~~~~~~

    The :mod:`latexcodec.codec` module
    contains all classes and functions for LaTeX code
    translation. For practical use,
    you should only ever need to import the :mod:`latexcodec` module,
    which will automatically register the codec
    so it can be used by :meth:`str.encode`, :meth:`str.decode`,
    and any of the functions defined in the :mod:`codecs` module
    such as :func:`codecs.open` and so on.
    The other functions and classes
    are exposed in case someone would want to extend them.

    .. autofunction:: register

    .. autofunction:: find_latex

    .. autoclass:: LatexIncrementalEncoder
        :show-inheritance:
        :members:

    .. autoclass:: LatexIncrementalDecoder
        :show-inheritance:
        :members:

    .. autoclass:: LatexCodec
        :show-inheritance:
        :members:

    .. autoclass:: LatexUnicodeTable
        :members:
"""

# Copyright (c) 2003, 2008 David Eppstein
# Copyright (c) 2011-2020 Matthias C. M. Troffaes
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

from __future__ import print_function

import codecs
from six import string_types, text_type
from six.moves import range

from latexcodec import lexer


def register():
    """Register the :func:`find_latex` codec search function.

    .. seealso:: :func:`codecs.register`
    """
    codecs.register(find_latex)

# returns the codec search function
# this is used if latex_codec.py were to be placed in stdlib


def getregentry():
    """Encodings module API."""
    return find_latex('latex')


class LatexUnicodeTable:

    """Tabulates a translation between LaTeX and unicode."""

    def __init__(self, lexer):
        self.lexer = lexer
        self.unicode_map = {}
        self.max_length = 0
        self.latex_map = {}
        self.register_all()

    def register_all(self):
        """Register all symbols and their LaTeX equivalents
        (called by constructor).
        """
        # TODO complete this list
        # register special symbols
        self.register(u'\n\n', u' \\par', encode=False)
        self.register(u'\n\n', u'\\par', encode=False)
        self.register(u' ', u'\\ ', encode=False)
        self.register(u'\N{EM SPACE}', u'\\quad')
        self.register(u'\N{THIN SPACE}', u' ', decode=False)
        self.register(u'%', u'\\%')
        self.register(u'\N{EN DASH}', u'--')
        self.register(u'\N{EN DASH}', u'\\textendash')
        self.register(u'\N{EM DASH}', u'---')
        self.register(u'\N{EM DASH}', u'\\textemdash')
        self.register(u'\N{REPLACEMENT CHARACTER}', u"????", decode=False)
        self.register(u'\N{LEFT SINGLE QUOTATION MARK}', u'`', decode=False)
        self.register(u'\N{RIGHT SINGLE QUOTATION MARK}', u"'", decode=False)
        self.register(u'\N{LEFT DOUBLE QUOTATION MARK}', u'``')
        self.register(u'\N{RIGHT DOUBLE QUOTATION MARK}', u"''")
        self.register(u'\N{DOUBLE LOW-9 QUOTATION MARK}', u",,")
        self.register(u'\N{DOUBLE LOW-9 QUOTATION MARK}', u'\\glqq',
                      encode=False)
        self.register(u'\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}',
                      u'\\guillemotleft')
        self.register(u'\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}',
                      u'\\guillemotright')
        self.register(u'\N{MODIFIER LETTER PRIME}', u"'", decode=False)
        self.register(u'\N{MODIFIER LETTER DOUBLE PRIME}', u"''", decode=False)
        self.register(u'\N{MODIFIER LETTER TURNED COMMA}', u'`', decode=False)
        self.register(u'\N{MODIFIER LETTER APOSTROPHE}', u"'", decode=False)
        self.register(u'\N{MODIFIER LETTER REVERSED COMMA}', u'`',
                      decode=False)
        self.register(u'\N{DAGGER}', u'\\dag')
        self.register(u'\N{DOUBLE DAGGER}', u'\\ddag')

        self.register(u'\\', u'\\textbackslash', encode=False)
        self.register(u'\\', u'\\backslash', mode='math', encode=False)

        self.register(u'\N{TILDE OPERATOR}', u'\\sim', mode='math')
        self.register(u'\N{MODIFIER LETTER LOW TILDE}',
                      u'\\texttildelow', package='textcomp')
        self.register(u'\N{SMALL TILDE}', u'\\~{}')
        self.register(u'~', u'\\textasciitilde')

        self.register(u'\N{BULLET}', u'\\bullet', mode='math')
        self.register(u'\N{BULLET}', u'\\textbullet', package='textcomp')
        self.register(u'\N{ASTERISK OPERATOR}', u'\\ast', mode='math')

        self.register(u'\N{NUMBER SIGN}', u'\\#')
        self.register(u'\N{LOW LINE}', u'\\_')
        self.register(u'\N{AMPERSAND}', u'\\&')
        self.register(u'\N{NO-BREAK SPACE}', u'~')
        self.register(u'\N{INVERTED EXCLAMATION MARK}', u'!`')
        self.register(u'\N{CENT SIGN}', u'\\not{c}')

        self.register(u'\N{POUND SIGN}', u'\\pounds')
        self.register(u'\N{POUND SIGN}', u'\\textsterling', package='textcomp')
        self.register(u'\N{YEN SIGN}', u'\\yen')
        self.register(u'\N{YEN SIGN}', u'\\textyen', package='textcomp')

        self.register(u'\N{SECTION SIGN}', u'\\S')
        self.register(u'\N{DIAERESIS}', u'\\"{}')
        self.register(u'\N{NOT SIGN}', u'\\neg')
        self.register(u'\N{HYPHEN}', u'-', decode=False)
        self.register(u'\N{SOFT HYPHEN}', u'\\-')
        self.register(u'\N{MACRON}', u'\\={}')

        self.register(u'\N{DEGREE SIGN}', u'^\\circ', mode='math')
        self.register(u'\N{DEGREE SIGN}', u'\\textdegree', package='textcomp')

        self.register(u'\N{MINUS SIGN}', u'-', mode='math')
        self.register(u'\N{PLUS-MINUS SIGN}', u'\\pm', mode='math')
        self.register(u'\N{PLUS-MINUS SIGN}', u'\\textpm', package='textcomp')

        self.register(u'\N{SUPERSCRIPT TWO}', u'^2', mode='math')
        self.register(
            u'\N{SUPERSCRIPT TWO}',
            u'\\texttwosuperior',
            package='textcomp')

        self.register(u'\N{SUPERSCRIPT THREE}', u'^3', mode='math')
        self.register(
            u'\N{SUPERSCRIPT THREE}',
            u'\\textthreesuperior',
            package='textcomp')

        self.register(u'\N{ACUTE ACCENT}', u"\\'{}")

        self.register(u'\N{MICRO SIGN}', u'\\mu', mode='math')
        self.register(u'\N{MICRO SIGN}', u'\\micro', package='gensymu')

        self.register(u'\N{PILCROW SIGN}', u'\\P')

        self.register(u'\N{MIDDLE DOT}', u'\\cdot', mode='math')
        self.register(
            u'\N{MIDDLE DOT}',
            u'\\textperiodcentered',
            package='textcomp')

        self.register(u'\N{CEDILLA}', u'\\c{}')

        self.register(u'\N{SUPERSCRIPT ONE}', u'^1', mode='math')
        self.register(
            u'\N{SUPERSCRIPT ONE}',
            u'\\textonesuperior',
            package='textcomp')

        self.register(u'\N{INVERTED QUESTION MARK}', u'?`')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH GRAVE}', u'\\`A')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH CIRCUMFLEX}', u'\\^A')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH TILDE}', u'\\~A')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH DIAERESIS}', u'\\"A')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}', u'\\AA')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH RING ABOVE}', u'\\r A',
                      encode=False)
        self.register(u'\N{LATIN CAPITAL LETTER AE}', u'\\AE')
        self.register(u'\N{LATIN CAPITAL LETTER C WITH CEDILLA}', u'\\c C')
        self.register(u'\N{LATIN CAPITAL LETTER E WITH GRAVE}', u'\\`E')
        self.register(u'\N{LATIN CAPITAL LETTER E WITH ACUTE}', u"\\'E")
        self.register(u'\N{LATIN CAPITAL LETTER E WITH CIRCUMFLEX}', u'\\^E')
        self.register(u'\N{LATIN CAPITAL LETTER E WITH DIAERESIS}', u'\\"E')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH GRAVE}', u'\\`I')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH CIRCUMFLEX}', u'\\^I')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH DIAERESIS}', u'\\"I')
        self.register(u'\N{LATIN CAPITAL LETTER N WITH TILDE}', u'\\~N')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH GRAVE}', u'\\`O')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH ACUTE}', u"\\'O")
        self.register(u'\N{LATIN CAPITAL LETTER O WITH CIRCUMFLEX}', u'\\^O')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH TILDE}', u'\\~O')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH DIAERESIS}', u'\\"O')
        self.register(u'\N{MULTIPLICATION SIGN}', u'\\times', mode='math')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH STROKE}', u'\\O')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH GRAVE}', u'\\`U')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH ACUTE}', u"\\'U")
        self.register(u'\N{LATIN CAPITAL LETTER U WITH CIRCUMFLEX}', u'\\^U')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH DIAERESIS}', u'\\"U')
        self.register(u'\N{LATIN CAPITAL LETTER Y WITH ACUTE}', u"\\'Y")
        self.register(u'\N{LATIN SMALL LETTER SHARP S}', u'\\ss')
        self.register(u'\N{LATIN SMALL LETTER A WITH GRAVE}', u'\\`a')
        self.register(u'\N{LATIN SMALL LETTER A WITH ACUTE}', u"\\'a")
        self.register(u'\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}', u'\\^a')
        self.register(u'\N{LATIN SMALL LETTER A WITH TILDE}', u'\\~a')
        self.register(u'\N{LATIN SMALL LETTER A WITH DIAERESIS}', u'\\"a')
        self.register(u'\N{LATIN SMALL LETTER A WITH RING ABOVE}', u'\\aa')
        self.register(u'\N{LATIN SMALL LETTER A WITH RING ABOVE}', u'\\r a',
                      encode=False)
        self.register(u'\N{LATIN SMALL LETTER AE}', u'\\ae')
        self.register(u'\N{LATIN SMALL LETTER C WITH CEDILLA}', u'\\c c')
        self.register(u'\N{LATIN SMALL LETTER E WITH GRAVE}', u'\\`e')
        self.register(u'\N{LATIN SMALL LETTER E WITH ACUTE}', u"\\'e")
        self.register(u'\N{LATIN SMALL LETTER E WITH CIRCUMFLEX}', u'\\^e')
        self.register(u'\N{LATIN SMALL LETTER E WITH DIAERESIS}', u'\\"e')
        self.register(u'\N{LATIN SMALL LETTER I WITH GRAVE}', u'\\`\\i')
        self.register(u'\N{LATIN SMALL LETTER I WITH GRAVE}', u'\\`i')
        self.register(u'\N{LATIN SMALL LETTER I WITH ACUTE}', u"\\'\\i")
        self.register(u'\N{LATIN SMALL LETTER I WITH ACUTE}', u"\\'i")
        self.register(u'\N{LATIN SMALL LETTER I WITH CIRCUMFLEX}', u'\\^\\i')
        self.register(u'\N{LATIN SMALL LETTER I WITH CIRCUMFLEX}', u'\\^i')
        self.register(u'\N{LATIN SMALL LETTER I WITH DIAERESIS}', u'\\"\\i')
        self.register(u'\N{LATIN SMALL LETTER I WITH DIAERESIS}', u'\\"i')
        self.register(u'\N{LATIN SMALL LETTER N WITH TILDE}', u'\\~n')
        self.register(u'\N{LATIN SMALL LETTER O WITH GRAVE}', u'\\`o')
        self.register(u'\N{LATIN SMALL LETTER O WITH ACUTE}', u"\\'o")
        self.register(u'\N{LATIN SMALL LETTER O WITH CIRCUMFLEX}', u'\\^o')
        self.register(u'\N{LATIN SMALL LETTER O WITH TILDE}', u'\\~o')
        self.register(u'\N{LATIN SMALL LETTER O WITH DIAERESIS}', u'\\"o')
        self.register(u'\N{DIVISION SIGN}', u'\\div', mode='math')
        self.register(u'\N{LATIN SMALL LETTER O WITH STROKE}', u'\\o')
        self.register(u'\N{LATIN SMALL LETTER U WITH GRAVE}', u'\\`u')
        self.register(u'\N{LATIN SMALL LETTER U WITH ACUTE}', u"\\'u")
        self.register(u'\N{LATIN SMALL LETTER U WITH CIRCUMFLEX}', u'\\^u')
        self.register(u'\N{LATIN SMALL LETTER U WITH DIAERESIS}', u'\\"u')
        self.register(u'\N{LATIN SMALL LETTER Y WITH ACUTE}', u"\\'y")
        self.register(u'\N{LATIN SMALL LETTER Y WITH DIAERESIS}', u'\\"y')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH MACRON}', u'\\=A')
        self.register(u'\N{LATIN SMALL LETTER A WITH MACRON}', u'\\=a')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH BREVE}', u'\\u A')
        self.register(u'\N{LATIN SMALL LETTER A WITH BREVE}', u'\\u a')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH OGONEK}', u'\\k A')
        self.register(u'\N{LATIN SMALL LETTER A WITH OGONEK}', u'\\k a')
        self.register(u'\N{LATIN CAPITAL LETTER C WITH ACUTE}', u"\\'C")
        self.register(u'\N{LATIN SMALL LETTER C WITH ACUTE}', u"\\'c")
        self.register(u'\N{LATIN CAPITAL LETTER C WITH CIRCUMFLEX}', u'\\^C')
        self.register(u'\N{LATIN SMALL LETTER C WITH CIRCUMFLEX}', u'\\^c')
        self.register(u'\N{LATIN CAPITAL LETTER C WITH DOT ABOVE}', u'\\.C')
        self.register(u'\N{LATIN SMALL LETTER C WITH DOT ABOVE}', u'\\.c')
        self.register(u'\N{LATIN CAPITAL LETTER C WITH CARON}', u'\\v C')
        self.register(u'\N{LATIN SMALL LETTER C WITH CARON}', u'\\v c')
        self.register(u'\N{LATIN CAPITAL LETTER D WITH CARON}', u'\\v D')
        self.register(u'\N{LATIN SMALL LETTER D WITH CARON}', u'\\v d')
        self.register(u'\N{LATIN CAPITAL LETTER E WITH MACRON}', u'\\=E')
        self.register(u'\N{LATIN SMALL LETTER E WITH MACRON}', u'\\=e')
        self.register(u'\N{LATIN CAPITAL LETTER E WITH BREVE}', u'\\u E')
        self.register(u'\N{LATIN SMALL LETTER E WITH BREVE}', u'\\u e')
        self.register(u'\N{LATIN CAPITAL LETTER E WITH DOT ABOVE}', u'\\.E')
        self.register(u'\N{LATIN SMALL LETTER E WITH DOT ABOVE}', u'\\.e')
        self.register(u'\N{LATIN CAPITAL LETTER E WITH OGONEK}', u'\\k E')
        self.register(u'\N{LATIN SMALL LETTER E WITH OGONEK}', u'\\k e')
        self.register(u'\N{LATIN CAPITAL LETTER E WITH CARON}', u'\\v E')
        self.register(u'\N{LATIN SMALL LETTER E WITH CARON}', u'\\v e')
        self.register(u'\N{LATIN CAPITAL LETTER G WITH CIRCUMFLEX}', u'\\^G')
        self.register(u'\N{LATIN SMALL LETTER G WITH CIRCUMFLEX}', u'\\^g')
        self.register(u'\N{LATIN CAPITAL LETTER G WITH BREVE}', u'\\u G')
        self.register(u'\N{LATIN SMALL LETTER G WITH BREVE}', u'\\u g')
        self.register(u'\N{LATIN CAPITAL LETTER G WITH DOT ABOVE}', u'\\.G')
        self.register(u'\N{LATIN SMALL LETTER G WITH DOT ABOVE}', u'\\.g')
        self.register(u'\N{LATIN CAPITAL LETTER G WITH CEDILLA}', u'\\c G')
        self.register(u'\N{LATIN SMALL LETTER G WITH CEDILLA}', u'\\c g')
        self.register(u'\N{LATIN CAPITAL LETTER H WITH CIRCUMFLEX}', u'\\^H')
        self.register(u'\N{LATIN SMALL LETTER H WITH CIRCUMFLEX}', u'\\^h')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH TILDE}', u'\\~I')
        self.register(u'\N{LATIN SMALL LETTER I WITH TILDE}', u'\\~\\i')
        self.register(u'\N{LATIN SMALL LETTER I WITH TILDE}', u'\\~i')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH MACRON}', u'\\=I')
        self.register(u'\N{LATIN SMALL LETTER I WITH MACRON}', u'\\=\\i')
        self.register(u'\N{LATIN SMALL LETTER I WITH MACRON}', u'\\=i')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH BREVE}', u'\\u I')
        self.register(u'\N{LATIN SMALL LETTER I WITH BREVE}', u'\\u\\i')
        self.register(u'\N{LATIN SMALL LETTER I WITH BREVE}', u'\\u i')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH OGONEK}', u'\\k I')
        self.register(u'\N{LATIN SMALL LETTER I WITH OGONEK}', u'\\k i')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}', u'\\.I')
        self.register(u'\N{LATIN SMALL LETTER DOTLESS I}', u'\\i')
        self.register(u'\N{LATIN CAPITAL LIGATURE IJ}', u'IJ', decode=False)
        self.register(u'\N{LATIN SMALL LIGATURE IJ}', u'ij', decode=False)
        self.register(u'\N{LATIN CAPITAL LETTER J WITH CIRCUMFLEX}', u'\\^J')
        self.register(u'\N{LATIN SMALL LETTER J WITH CIRCUMFLEX}', u'\\^\\j')
        self.register(u'\N{LATIN SMALL LETTER J WITH CIRCUMFLEX}', u'\\^j')
        self.register(u'\N{LATIN CAPITAL LETTER K WITH CEDILLA}', u'\\c K')
        self.register(u'\N{LATIN SMALL LETTER K WITH CEDILLA}', u'\\c k')
        self.register(u'\N{LATIN CAPITAL LETTER L WITH ACUTE}', u"\\'L")
        self.register(u'\N{LATIN SMALL LETTER L WITH ACUTE}', u"\\'l")
        self.register(u'\N{LATIN CAPITAL LETTER L WITH CEDILLA}', u'\\c L')
        self.register(u'\N{LATIN SMALL LETTER L WITH CEDILLA}', u'\\c l')
        self.register(u'\N{LATIN CAPITAL LETTER L WITH CARON}', u'\\v L')
        self.register(u'\N{LATIN SMALL LETTER L WITH CARON}', u'\\v l')
        self.register(u'\N{LATIN CAPITAL LETTER L WITH STROKE}', u'\\L')
        self.register(u'\N{LATIN SMALL LETTER L WITH STROKE}', u'\\l')
        self.register(u'\N{LATIN CAPITAL LETTER N WITH ACUTE}', u"\\'N")
        self.register(u'\N{LATIN SMALL LETTER N WITH ACUTE}', u"\\'n")
        self.register(u'\N{LATIN CAPITAL LETTER N WITH CEDILLA}', u'\\c N')
        self.register(u'\N{LATIN SMALL LETTER N WITH CEDILLA}', u'\\c n')
        self.register(u'\N{LATIN CAPITAL LETTER N WITH CARON}', u'\\v N')
        self.register(u'\N{LATIN SMALL LETTER N WITH CARON}', u'\\v n')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH MACRON}', u'\\=O')
        self.register(u'\N{LATIN SMALL LETTER O WITH MACRON}', u'\\=o')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH BREVE}', u'\\u O')
        self.register(u'\N{LATIN SMALL LETTER O WITH BREVE}', u'\\u o')
        self.register(
            u'\N{LATIN CAPITAL LETTER O WITH DOUBLE ACUTE}',
            u'\\H O')
        self.register(u'\N{LATIN SMALL LETTER O WITH DOUBLE ACUTE}', u'\\H o')
        self.register(u'\N{LATIN CAPITAL LIGATURE OE}', u'\\OE')
        self.register(u'\N{LATIN SMALL LIGATURE OE}', u'\\oe')
        self.register(u'\N{LATIN CAPITAL LETTER R WITH ACUTE}', u"\\'R")
        self.register(u'\N{LATIN SMALL LETTER R WITH ACUTE}', u"\\'r")
        self.register(u'\N{LATIN CAPITAL LETTER R WITH CEDILLA}', u'\\c R')
        self.register(u'\N{LATIN SMALL LETTER R WITH CEDILLA}', u'\\c r')
        self.register(u'\N{LATIN CAPITAL LETTER R WITH CARON}', u'\\v R')
        self.register(u'\N{LATIN SMALL LETTER R WITH CARON}', u'\\v r')
        self.register(u'\N{LATIN CAPITAL LETTER S WITH ACUTE}', u"\\'S")
        self.register(u'\N{LATIN SMALL LETTER S WITH ACUTE}', u"\\'s")
        self.register(u'\N{LATIN CAPITAL LETTER S WITH CIRCUMFLEX}', u'\\^S')
        self.register(u'\N{LATIN SMALL LETTER S WITH CIRCUMFLEX}', u'\\^s')
        self.register(u'\N{LATIN CAPITAL LETTER S WITH CEDILLA}', u'\\c S')
        self.register(u'\N{LATIN SMALL LETTER S WITH CEDILLA}', u'\\c s')
        self.register(u'\N{LATIN CAPITAL LETTER S WITH CARON}', u'\\v S')
        self.register(u'\N{LATIN SMALL LETTER S WITH CARON}', u'\\v s')
        self.register(u'\N{LATIN CAPITAL LETTER T WITH CEDILLA}', u'\\c T')
        self.register(u'\N{LATIN SMALL LETTER T WITH CEDILLA}', u'\\c t')
        self.register(u'\N{LATIN CAPITAL LETTER T WITH CARON}', u'\\v T')
        self.register(u'\N{LATIN SMALL LETTER T WITH CARON}', u'\\v t')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH TILDE}', u'\\~U')
        self.register(u'\N{LATIN SMALL LETTER U WITH TILDE}', u'\\~u')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH MACRON}', u'\\=U')
        self.register(u'\N{LATIN SMALL LETTER U WITH MACRON}', u'\\=u')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH BREVE}', u'\\u U')
        self.register(u'\N{LATIN SMALL LETTER U WITH BREVE}', u'\\u u')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH RING ABOVE}', u'\\r U')
        self.register(u'\N{LATIN SMALL LETTER U WITH RING ABOVE}', u'\\r u')
        self.register(
            u'\N{LATIN CAPITAL LETTER U WITH DOUBLE ACUTE}',
            u'\\H U')
        self.register(u'\N{LATIN SMALL LETTER U WITH DOUBLE ACUTE}', u'\\H u')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH OGONEK}', u'\\k U')
        self.register(u'\N{LATIN SMALL LETTER U WITH OGONEK}', u'\\k u')
        self.register(u'\N{LATIN CAPITAL LETTER W WITH CIRCUMFLEX}', u'\\^W')
        self.register(u'\N{LATIN SMALL LETTER W WITH CIRCUMFLEX}', u'\\^w')
        self.register(u'\N{LATIN CAPITAL LETTER Y WITH CIRCUMFLEX}', u'\\^Y')
        self.register(u'\N{LATIN SMALL LETTER Y WITH CIRCUMFLEX}', u'\\^y')
        self.register(u'\N{LATIN CAPITAL LETTER Y WITH DIAERESIS}', u'\\"Y')
        self.register(u'\N{LATIN CAPITAL LETTER Z WITH ACUTE}', u"\\'Z")
        self.register(u'\N{LATIN SMALL LETTER Z WITH ACUTE}', u"\\'z")
        self.register(u'\N{LATIN CAPITAL LETTER Z WITH DOT ABOVE}', u'\\.Z')
        self.register(u'\N{LATIN SMALL LETTER Z WITH DOT ABOVE}', u'\\.z')
        self.register(u'\N{LATIN CAPITAL LETTER Z WITH CARON}', u'\\v Z')
        self.register(u'\N{LATIN SMALL LETTER Z WITH CARON}', u'\\v z')
        self.register(u'\N{LATIN CAPITAL LETTER DZ WITH CARON}', u'D\\v Z')
        self.register(
            u'\N{LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON}',
            u'D\\v z')
        self.register(u'\N{LATIN SMALL LETTER DZ WITH CARON}', u'd\\v z')
        self.register(u'\N{LATIN CAPITAL LETTER LJ}', u'LJ', decode=False)
        self.register(
            u'\N{LATIN CAPITAL LETTER L WITH SMALL LETTER J}',
            u'Lj',
            decode=False)
        self.register(u'\N{LATIN SMALL LETTER LJ}', u'lj', decode=False)
        self.register(u'\N{LATIN CAPITAL LETTER NJ}', u'NJ', decode=False)
        self.register(
            u'\N{LATIN CAPITAL LETTER N WITH SMALL LETTER J}',
            u'Nj',
            decode=False)
        self.register(u'\N{LATIN SMALL LETTER NJ}', u'nj', decode=False)
        self.register(u'\N{LATIN CAPITAL LETTER A WITH CARON}', u'\\v A')
        self.register(u'\N{LATIN SMALL LETTER A WITH CARON}', u'\\v a')
        self.register(u'\N{LATIN CAPITAL LETTER I WITH CARON}', u'\\v I')
        self.register(u'\N{LATIN SMALL LETTER I WITH CARON}', u'\\v\\i')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH CARON}', u'\\v O')
        self.register(u'\N{LATIN SMALL LETTER O WITH CARON}', u'\\v o')
        self.register(u'\N{LATIN CAPITAL LETTER U WITH CARON}', u'\\v U')
        self.register(u'\N{LATIN SMALL LETTER U WITH CARON}', u'\\v u')
        self.register(u'\N{LATIN CAPITAL LETTER G WITH CARON}', u'\\v G')
        self.register(u'\N{LATIN SMALL LETTER G WITH CARON}', u'\\v g')
        self.register(u'\N{LATIN CAPITAL LETTER K WITH CARON}', u'\\v K')
        self.register(u'\N{LATIN SMALL LETTER K WITH CARON}', u'\\v k')
        self.register(u'\N{LATIN CAPITAL LETTER O WITH OGONEK}', u'\\k O')
        self.register(u'\N{LATIN SMALL LETTER O WITH OGONEK}', u'\\k o')
        self.register(u'\N{LATIN SMALL LETTER J WITH CARON}', u'\\v\\j')
        self.register(u'\N{LATIN CAPITAL LETTER DZ}', u'DZ', decode=False)
        self.register(
            u'\N{LATIN CAPITAL LETTER D WITH SMALL LETTER Z}',
            u'Dz',
            decode=False)
        self.register(u'\N{LATIN SMALL LETTER DZ}', u'dz', decode=False)
        self.register(u'\N{LATIN CAPITAL LETTER G WITH ACUTE}', u"\\'G")
        self.register(u'\N{LATIN SMALL LETTER G WITH ACUTE}', u"\\'g")
        self.register(u'\N{LATIN CAPITAL LETTER AE WITH ACUTE}', u"\\'\\AE")
        self.register(u'\N{LATIN SMALL LETTER AE WITH ACUTE}', u"\\'\\ae")
        self.register(
            u'\N{LATIN CAPITAL LETTER O WITH STROKE AND ACUTE}',
            u"\\'\\O")
        self.register(
            u'\N{LATIN SMALL LETTER O WITH STROKE AND ACUTE}',
            u"\\'\\o")
        self.register(u'\N{LATIN CAPITAL LETTER ETH}', u'\\DH')
        self.register(u'\N{LATIN SMALL LETTER ETH}', u'\\dh')
        self.register(u'\N{LATIN CAPITAL LETTER THORN}', u'\\TH')
        self.register(u'\N{LATIN SMALL LETTER THORN}', u'\\th')
        self.register(u'\N{LATIN CAPITAL LETTER D WITH STROKE}', u'\\DJ')
        self.register(u'\N{LATIN SMALL LETTER D WITH STROKE}', u'\\dj')
        self.register(u'\N{LATIN CAPITAL LETTER D WITH DOT BELOW}', u'\\d D')
        self.register(u'\N{LATIN SMALL LETTER D WITH DOT BELOW}', u'\\d d')
        self.register(u'\N{LATIN CAPITAL LETTER L WITH DOT BELOW}', u'\\d L')
        self.register(u'\N{LATIN SMALL LETTER L WITH DOT BELOW}', u'\\d l')
        self.register(u'\N{LATIN CAPITAL LETTER M WITH DOT BELOW}', u'\\d M')
        self.register(u'\N{LATIN SMALL LETTER M WITH DOT BELOW}', u'\\d m')
        self.register(u'\N{LATIN CAPITAL LETTER N WITH DOT BELOW}', u'\\d N')
        self.register(u'\N{LATIN SMALL LETTER N WITH DOT BELOW}', u'\\d n')
        self.register(u'\N{LATIN CAPITAL LETTER R WITH DOT BELOW}', u'\\d R')
        self.register(u'\N{LATIN SMALL LETTER R WITH DOT BELOW}', u'\\d r')
        self.register(u'\N{LATIN CAPITAL LETTER S WITH DOT BELOW}', u'\\d S')
        self.register(u'\N{LATIN SMALL LETTER S WITH DOT BELOW}', u'\\d s')
        self.register(u'\N{LATIN CAPITAL LETTER T WITH DOT BELOW}', u'\\d T')
        self.register(u'\N{LATIN SMALL LETTER T WITH DOT BELOW}', u'\\d t')
        self.register(u'\N{LATIN CAPITAL LETTER S WITH COMMA BELOW}',
                      u'\\textcommabelow S')
        self.register(u'\N{LATIN SMALL LETTER S WITH COMMA BELOW}',
                      u'\\textcommabelow s')
        self.register(u'\N{LATIN CAPITAL LETTER T WITH COMMA BELOW}',
                      u'\\textcommabelow T')
        self.register(u'\N{LATIN SMALL LETTER T WITH COMMA BELOW}',
                      u'\\textcommabelow t')
        self.register(u'\N{PARTIAL DIFFERENTIAL}', u'\\partial', mode='math')
        self.register(u'\N{N-ARY PRODUCT}', u'\\prod', mode='math')
        self.register(u'\N{N-ARY SUMMATION}', u'\\sum', mode='math')
        self.register(u'\N{SQUARE ROOT}', u'\\surd', mode='math')
        self.register(u'\N{INFINITY}', u'\\infty', mode='math')
        self.register(u'\N{INTEGRAL}', u'\\int', mode='math')
        self.register(u'\N{INTERSECTION}', u'\\cap', mode='math')
        self.register(u'\N{UNION}', u'\\cup', mode='math')
        self.register(u'\N{RIGHTWARDS ARROW}', u'\\rightarrow', mode='math')
        self.register(
            u'\N{RIGHTWARDS DOUBLE ARROW}',
            u'\\Rightarrow',
            mode='math')
        self.register(u'\N{LEFTWARDS ARROW}', u'\\leftarrow', mode='math')
        self.register(
            u'\N{LEFTWARDS DOUBLE ARROW}',
            u'\\Leftarrow',
            mode='math')
        self.register(u'\N{LOGICAL OR}', u'\\vee', mode='math')
        self.register(u'\N{LOGICAL AND}', u'\\wedge', mode='math')
        self.register(u'\N{ALMOST EQUAL TO}', u'\\approx', mode='math')
        self.register(u'\N{NOT EQUAL TO}', u'\\neq', mode='math')
        self.register(u'\N{LESS-THAN OR EQUAL TO}', u'\\leq', mode='math')
        self.register(u'\N{GREATER-THAN OR EQUAL TO}', u'\\geq', mode='math')
        self.register(u'\N{MODIFIER LETTER CIRCUMFLEX ACCENT}', u'\\^{}')
        self.register(u'\N{CARON}', u'\\v{}')
        self.register(u'\N{BREVE}', u'\\u{}')
        self.register(u'\N{DOT ABOVE}', u'\\.{}')
        self.register(u'\N{RING ABOVE}', u'\\r{}')
        self.register(u'\N{OGONEK}', u'\\k{}')
        self.register(u'\N{DOUBLE ACUTE ACCENT}', u'\\H{}')
        self.register(u'\N{LATIN SMALL LIGATURE FI}', u'fi', decode=False)
        self.register(u'\N{LATIN SMALL LIGATURE FL}', u'fl', decode=False)
        self.register(u'\N{LATIN SMALL LIGATURE FF}', u'ff', decode=False)

        self.register(u'\N{GREEK SMALL LETTER ALPHA}', u'\\alpha', mode='math')
        self.register(u'\N{GREEK SMALL LETTER BETA}', u'\\beta', mode='math')
        self.register(u'\N{GREEK SMALL LETTER GAMMA}', u'\\gamma', mode='math')
        self.register(u'\N{GREEK SMALL LETTER DELTA}', u'\\delta', mode='math')
        self.register(
            u'\N{GREEK SMALL LETTER EPSILON}',
            u'\\epsilon',
            mode='math')
        self.register(u'\N{GREEK SMALL LETTER ZETA}', u'\\zeta', mode='math')
        self.register(u'\N{GREEK SMALL LETTER ETA}', u'\\eta', mode='math')
        self.register(u'\N{GREEK SMALL LETTER THETA}', u'\\theta', mode='math')
        self.register(u'\N{GREEK SMALL LETTER THETA}', u'\\texttheta',
                      package='textgreek', encode=False)
        self.register(u'\N{GREEK SMALL LETTER IOTA}', u'\\iota', mode='math')
        self.register(u'\N{GREEK SMALL LETTER KAPPA}', u'\\kappa', mode='math')
        self.register(
            u'\N{GREEK SMALL LETTER LAMDA}',
            u'\\lambda',
            mode='math')  # LAMDA not LAMBDA
        self.register(u'\N{GREEK SMALL LETTER MU}', u'\\mu', mode='math')
        self.register(u'\N{GREEK SMALL LETTER NU}', u'\\nu', mode='math')
        self.register(u'\N{GREEK SMALL LETTER XI}', u'\\xi', mode='math')
        self.register(
            u'\N{GREEK SMALL LETTER OMICRON}',
            u'\\omicron',
            mode='math')
        self.register(u'\N{GREEK SMALL LETTER PI}', u'\\pi', mode='math')
        self.register(u'\N{GREEK SMALL LETTER RHO}', u'\\rho', mode='math')
        self.register(u'\N{GREEK SMALL LETTER SIGMA}', u'\\sigma', mode='math')
        self.register(u'\N{GREEK SMALL LETTER TAU}', u'\\tau', mode='math')
        self.register(
            u'\N{GREEK SMALL LETTER UPSILON}',
            u'\\upsilon',
            mode='math')
        self.register(u'\N{GREEK SMALL LETTER PHI}', u'\\phi', mode='math')
        self.register(u'\N{GREEK PHI SYMBOL}', u'\\varphi', mode='math')
        self.register(u'\N{GREEK SMALL LETTER CHI}', u'\\chi', mode='math')
        self.register(u'\N{GREEK SMALL LETTER PSI}', u'\\psi', mode='math')
        self.register(u'\N{GREEK SMALL LETTER OMEGA}', u'\\omega', mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER ALPHA}',
            u'\\Alpha',
            mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER BETA}', u'\\Beta', mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER GAMMA}',
            u'\\Gamma',
            mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER DELTA}',
            u'\\Delta',
            mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER EPSILON}',
            u'\\Epsilon',
            mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER ZETA}', u'\\Zeta', mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER ETA}', u'\\Eta', mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER THETA}',
            u'\\Theta',
            mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER IOTA}', u'\\Iota', mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER KAPPA}',
            u'\\Kappa',
            mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER LAMDA}',
            u'\\Lambda',
            mode='math')  # LAMDA not LAMBDA
        self.register(u'\N{GREEK CAPITAL LETTER MU}', u'\\Mu', mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER NU}', u'\\Nu', mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER XI}', u'\\Xi', mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER OMICRON}',
            u'\\Omicron',
            mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER PI}', u'\\Pi', mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER RHO}', u'\\Rho', mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER SIGMA}',
            u'\\Sigma',
            mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER TAU}', u'\\Tau', mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER UPSILON}',
            u'\\Upsilon',
            mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER PHI}', u'\\Phi', mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER CHI}', u'\\Chi', mode='math')
        self.register(u'\N{GREEK CAPITAL LETTER PSI}', u'\\Psi', mode='math')
        self.register(
            u'\N{GREEK CAPITAL LETTER OMEGA}',
            u'\\Omega',
            mode='math')
        self.register(u'\N{COPYRIGHT SIGN}', u'\\copyright')
        self.register(u'\N{COPYRIGHT SIGN}', u'\\textcopyright')
        self.register(u'\N{LATIN CAPITAL LETTER A WITH ACUTE}', u"\\'A")
        self.register(u'\N{LATIN CAPITAL LETTER I WITH ACUTE}', u"\\'I")
        self.register(u'\N{HORIZONTAL ELLIPSIS}', u'\\ldots')
        self.register(u'\N{TRADE MARK SIGN}', u'^{TM}', mode='math')
        self.register(
            u'\N{TRADE MARK SIGN}',
            u'\\texttrademark',
            package='textcomp')
        self.register(
            u'\N{REGISTERED SIGN}',
            u'\\textregistered',
            package='textcomp')
        # \=O and \=o will be translated into Ō and ō before we can
        # match the full latex string... so decoding disabled for now
        self.register(u'Ǭ', text_type(r'\textogonekcentered{\=O}'),
                      decode=False)
        self.register(u'ǭ', text_type(r'\textogonekcentered{\=o}'),
                      decode=False)
        self.register(u'ℕ', text_type(r'\mathbb{N}'), mode='math')
        self.register(u'ℕ', text_type(r'\mathbb N'), mode='math', decode=False)
        self.register(u'ℤ', text_type(r'\mathbb{Z}'), mode='math')
        self.register(u'ℤ', text_type(r'\mathbb Z'), mode='math', decode=False)
        self.register(u'ℚ', text_type(r'\mathbb{Q}'), mode='math')
        self.register(u'ℚ', text_type(r'\mathbb Q'), mode='math', decode=False)
        self.register(u'ℝ', text_type(r'\mathbb{R}'), mode='math')
        self.register(u'ℝ', text_type(r'\mathbb R'), mode='math', decode=False)
        self.register(u'ℂ', text_type(r'\mathbb{C}'), mode='math')
        self.register(u'ℂ', text_type(r'\mathbb C'), mode='math', decode=False)

    def register(self, unicode_text, latex_text, mode='text', package=None,
                 decode=True, encode=True):
        """Register a correspondence between *unicode_text* and *latex_text*.

        :param str unicode_text: A unicode character.
        :param str latex_text: Its corresponding LaTeX translation.
        :param str mode: LaTeX mode in which the translation applies
            (``'text'`` or ``'math'``).
        :param str package: LaTeX package requirements (currently ignored).
        :param bool decode: Whether this translation applies to decoding
            (default: ``True``).
        :param bool encode: Whether this translation applies to encoding
            (default: ``True``).
        """
        if mode == 'math':
            # also register text version
            self.register(unicode_text, u'$' + latex_text + u'$', mode='text',
                          package=package, decode=decode, encode=encode)
            self.register(unicode_text,
                          text_type(r'\(') + latex_text + text_type(r'\)'),
                          mode='text', package=package,
                          decode=decode, encode=encode)
            # XXX for the time being, we do not perform in-math substitutions
            return
        if package is not None:
            # TODO implement packages
            pass
        # tokenize, and register unicode translation
        self.lexer.reset()
        self.lexer.state = 'M'
        tokens = tuple(self.lexer.get_tokens(latex_text, final=True))
        if decode:
            if tokens not in self.unicode_map:
                self.max_length = max(self.max_length, len(tokens))
                self.unicode_map[tokens] = unicode_text
            # also register token variant with brackets, if appropriate
            # for instance, "\'{e}" for "\'e", "\c{c}" for "\c c", etc.
            # note: we do not remove brackets (they sometimes matter,
            # e.g. bibtex uses them to prevent lower case transformation)
            if (len(tokens) == 2 and
                tokens[0].name.startswith(u'control') and
                    tokens[1].name == u'chars'):
                alt_tokens = (tokens[0], self.lexer.curlylefttoken, tokens[1],
                              self.lexer.curlyrighttoken)
                if alt_tokens not in self.unicode_map:
                    self.max_length = max(self.max_length, len(alt_tokens))
                    self.unicode_map[alt_tokens] = u"{" + unicode_text + u"}"
        if encode and unicode_text not in self.latex_map:
            assert len(unicode_text) == 1
            self.latex_map[unicode_text] = (latex_text, tokens)


_LATEX_UNICODE_TABLE = LatexUnicodeTable(lexer.LatexIncrementalDecoder())
_ULATEX_UNICODE_TABLE = LatexUnicodeTable(
    lexer.UnicodeLatexIncrementalDecoder())

# incremental encoder does not need a buffer
# but decoder does


class LatexIncrementalEncoder(lexer.LatexIncrementalEncoder):

    """Translating incremental encoder for latex. Maintains a state to
    determine whether control spaces etc. need to be inserted.
    """

    emptytoken = lexer.Token(u"unknown", u"")
    """The empty token."""

    table = _LATEX_UNICODE_TABLE
    """Translation table."""

    def __init__(self, errors='strict'):
        super(LatexIncrementalEncoder, self).__init__(errors=errors)
        self.reset()

    def reset(self):
        super(LatexIncrementalEncoder, self).reset()
        self.state = 'M'

    def get_space_bytes(self, bytes_):
        """Inserts space bytes in space eating mode."""
        if self.state == 'S':
            # in space eating mode
            # control space needed?
            if bytes_.startswith(u' '):
                # replace by control space
                return u'\\ ', bytes_[1:]
            else:
                # insert space (it is eaten, but needed for separation)
                return u' ', bytes_
        else:
            return u'', bytes_

    def _get_latex_chars_tokens_from_char(self, c):
        # if ascii, try latex equivalents
        # (this covers \, #, &, and other special LaTeX characters)
        if ord(c) < 128:
            try:
                return self.table.latex_map[c]
            except KeyError:
                pass
        # next, try input encoding
        try:
            bytes_ = c.encode(self.inputenc, 'strict')
        except UnicodeEncodeError:
            pass
        else:
            return c, (lexer.Token(name=u'chars', text=c),)
        # next, try latex equivalents of common unicode characters
        try:
            return self.table.latex_map[c]
        except KeyError:
            # translation failed
            if self.errors == 'strict':
                raise UnicodeEncodeError(
                    "latex",  # codec
                    c,  # problematic input
                    0, 1,  # location of problematic character
                    "don't know how to translate {0} into latex"
                    .format(repr(c)))
            elif self.errors == 'ignore':
                return u'', (self.emptytoken,)
            elif self.errors == 'replace':
                # use the \\char command
                # this assumes
                # \usepackage[T1]{fontenc}
                # \usepackage[utf8]{inputenc}
                bytes_ = u'{\\char' + str(ord(c)) + u'}'
                return bytes_, (lexer.Token(name=u'chars', text=bytes_),)
            elif self.errors == 'keep' and not self.binary_mode:
                return c,  (lexer.Token(name=u'chars', text=c),)
            else:
                raise ValueError(
                    "latex codec does not support {0} errors"
                    .format(self.errors))

    def get_latex_chars(self, unicode_, final=False):
        if not isinstance(unicode_, string_types):
            raise TypeError(
                "expected unicode for encode input, but got {0} instead"
                .format(unicode_.__class__.__name__))
        # convert character by character
        for pos, c in enumerate(unicode_):
            bytes_, tokens = self._get_latex_chars_tokens_from_char(c)
            space, bytes_ = self.get_space_bytes(bytes_)
            # update state
            if tokens[-1].name == u'control_word':
                # we're eating spaces
                self.state = 'S'
            else:
                self.state = 'M'
            if space:
                yield space
            yield bytes_


class LatexIncrementalDecoder(lexer.LatexIncrementalDecoder):

    """Translating incremental decoder for LaTeX."""

    table = _LATEX_UNICODE_TABLE
    """Translation table."""

    def __init__(self, errors='strict'):
        lexer.LatexIncrementalDecoder.__init__(self, errors=errors)

    def reset(self):
        lexer.LatexIncrementalDecoder.reset(self)
        self.token_buffer = []

    # python codecs API does not support multibuffer incremental decoders

    def getstate(self):
        raise NotImplementedError

    def setstate(self, state):
        raise NotImplementedError

    def get_unicode_tokens(self, chars, final=False):
        for token in self.get_tokens(chars, final=final):
            # at this point, token_buffer does not match anything
            self.token_buffer.append(token)
            # new token appended at the end, see if we have a match now
            # note: match is only possible at the *end* of the buffer
            # because all other positions have already been checked in
            # earlier iterations
            for i in range(len(self.token_buffer), 0, -1):
                last_tokens = tuple(self.token_buffer[-i:])  # last i tokens
                try:
                    unicode_text = self.table.unicode_map[last_tokens]
                except KeyError:
                    # no match: continue
                    continue
                else:
                    # match!! flush buffer, and translate last bit
                    # exclude last i tokens
                    for token in self.token_buffer[:-i]:
                        yield self.decode_token(token)
                    yield unicode_text
                    self.token_buffer = []
                    break
            # flush tokens that can no longer match
            while len(self.token_buffer) >= self.table.max_length:
                yield self.decode_token(self.token_buffer.pop(0))
        # also flush the buffer at the end
        if final:
            for token in self.token_buffer:
                yield self.decode_token(token)
            self.token_buffer = []


class LatexCodec(codecs.Codec):
    IncrementalEncoder = None
    IncrementalDecoder = None

    def encode(self, unicode_, errors='strict'):
        """Convert unicode string to LaTeX bytes."""
        encoder = self.IncrementalEncoder(errors=errors)
        return (
            encoder.encode(unicode_, final=True),
            len(unicode_),
        )

    def decode(self, bytes_, errors='strict'):
        """Convert LaTeX bytes to unicode string."""
        decoder = self.IncrementalDecoder(errors=errors)
        return (
            decoder.decode(bytes_, final=True),
            len(bytes_),
        )


class UnicodeLatexIncrementalDecoder(LatexIncrementalDecoder):
    table = _ULATEX_UNICODE_TABLE
    binary_mode = False


class UnicodeLatexIncrementalEncoder(LatexIncrementalEncoder):
    table = _ULATEX_UNICODE_TABLE
    binary_mode = False


def find_latex(encoding):
    """Return a :class:`codecs.CodecInfo` instance for the requested
    LaTeX *encoding*, which must be equal to ``latex``,
    or to ``latex+<encoding>``
    where ``<encoding>`` describes another encoding.
    """
    if u'_' in encoding:
        # Python 3.9 now normalizes "latex+latin1" to "latex_latin1"
        # https://bugs.python.org/issue37751
        encoding, _, inputenc_ = encoding.partition(u"_")
    else:
        encoding, _, inputenc_ = encoding.partition(u"+")
    if not inputenc_:
        inputenc_ = "ascii"
    if encoding == "latex":
        IncEnc = LatexIncrementalEncoder
        DecEnc = LatexIncrementalDecoder
    elif encoding == "ulatex":
        IncEnc = UnicodeLatexIncrementalEncoder
        DecEnc = UnicodeLatexIncrementalDecoder
    else:
        return None

    class IncrementalEncoder_(IncEnc):
        inputenc = inputenc_

    class IncrementalDecoder_(DecEnc):
        inputenc = inputenc_

    class Codec(LatexCodec):
        IncrementalEncoder = IncrementalEncoder_
        IncrementalDecoder = IncrementalDecoder_

    class StreamWriter(Codec, codecs.StreamWriter):
        pass

    class StreamReader(Codec, codecs.StreamReader):
        pass

    return codecs.CodecInfo(
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=Codec.IncrementalEncoder,
        incrementaldecoder=Codec.IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
