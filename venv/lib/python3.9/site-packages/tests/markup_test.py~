import pytest

from pybtex.markup import LaTeXParser
from pybtex.scanner import PybtexSyntaxError


@pytest.mark.parametrize(["bad_input"], [("abc{def}}",), ("abc{def}{",)])
def test_syntax_error(bad_input):
    with pytest.raises(PybtexSyntaxError):
        LaTeXParser(bad_input).parse()
