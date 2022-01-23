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


import pytest

from pybtex.database import parse_string
from pybtex.backends.html import Backend as HtmlBackend
from pybtex.style.formatting.unsrt import Style as UnsrtStyle

article_bib = """
@article{article,
  author  = {Peter Adams}, 
  title   = {The title of the work},
  journal = {The name of the journal},
  year    = 1993,
  number  = 2,
  pages   = {201-213},
  month   = 7,
  note    = {An optional note}, 
  volume  = 4
}
"""

article_html = """
Peter Adams.
The title of the work.
<em>The name of the journal</em>, 4(2):201&ndash;213, 7 1993.
An optional note.
"""

book_bib = """@book{book,
  author    = {Peter Babington}, 
  title     = {The title of the work},
  publisher = {The name of the publisher},
  year      = 1993,
  volume    = 4,
  series    = 10,
  address   = {The address},
  edition   = {3rd},
  month     = 7,
  note      = {An optional note},
  isbn      = {3257227892}
}
"""

book_html = """
Peter Babington.
<em>The title of the work</em>.
Volume&nbsp;4 of 10.
The name of the publisher, The address, 3rd edition, 7 1993.
ISBN 3257227892.
An optional note.
"""

booklet_bib = """
@booklet{booklet,
  title        = {The title of the work},
  author       = {Peter Caxton}, 
  howpublished = {How it was published},
  address      = {The address of the publisher},
  month        = jul,
  year         = 1993,
  note         = {An optional note}
}
"""

booklet_html = """
Peter Caxton.
The title of the work.
How it was published, The address of the publisher, July 1993, An optional note.
"""

inbook_bib = """
@inbook{inbook,
  author       = {Peter Eston}, 
  title        = {The title of the work},
  chapter      = 8,
  pages        = {201-213},
  publisher    = {The name of the publisher},
  year         = 1993,
  volume       = 4,
  series       = {The name of the series},
  address      = {The address of the publisher},
  edition      = {3rd},
  month        = 7,
  note         = {An optional note}
}
"""

inbook_html = """
Peter Eston.
<em>The title of the work</em>, chapter&nbsp;8, pages 201&ndash;213.
Volume&nbsp;4 of The name of the series.
The name of the publisher, The address of the publisher, 3rd edition, 7 1993, An optional note.
"""

incollection_bib = """
@incollection{incollection,
  author       = {Peter Farindon}, 
  title        = {The title of the work},
  booktitle    = {The title of the book},
  publisher    = {The name of the publisher},
  year         = 1993,
  editor       = {The editor},
  volume       = 4,
  series       = {The name of the series},
  chapter      = 8,
  pages        = {201-213},
  address      = {The address of the publisher},
  edition      = {3rd},
  month        = aug,
  note         = {An optional note}
}
"""

incollection_html = """
Peter Farindon.
The title of the work.
In The editor, editor, <em>The title of the book</em>, volume&nbsp;4 of The name of the series, chapter&nbsp;8, pages 201&ndash;213.
The name of the publisher, The address of the publisher, 3rd edition, August 1993.
"""

manual_bib = """
@manual{manual,
  title        = {The title of the work},
  author       = {Peter Gainsford}, 
  organization = {The organization},
  address      = {The address of the publisher},
  edition      = {3rd},
  month        = jan,
  year         = 1993,
  note         = {An optional note}
}
"""

manual_html = """
Peter Gainsford.
<em>The title of the work</em>.
The organization, The address of the publisher, 3rd edition, January 1993.
An optional note.
"""

masterthesis_bib = """
@mastersthesis{mastersthesis,
  author       = {Peter Harwood}, 
  title        = {The title of the work},
  school       = {The school of the thesis},
  year         = 1993,
  address      = {The address of the publisher},
  month        = feb,
  note         = {An optional note}
}
"""

masterthesis_html = """
Peter Harwood.
The title of the work.
Master's thesis, The school of the thesis, The address of the publisher, February 1993.
An optional note.
"""

misc_bib = """
@misc{misc,
  author       = {Peter Isley}, 
  title        = {The title of the work},
  howpublished = {How it was published},
  month        = oct,
  year         = 1993,
  note         = {An optional note}
}
"""

misc_html = """
Peter Isley.
The title of the work.
How it was published, October 1993.
An optional note.
"""

phdthesis_bib = """
@phdthesis{phdthesis,
  author       = {Peter Joslin}, 
  title        = {The title of the work},
  school       = {The school of the thesis},
  year         = 1993,
  address      = {The address of the publisher},
  month        = dec,
  note         = {An optional note}
}
"""

phdthesis_html = """
Peter Joslin.
<em>The title of the work</em>.
PhD thesis, The school of the thesis, The address of the publisher, December 1993.
An optional note.
"""

proceedings_bib = """
@proceedings{proceedings,
  title        = {The title of the work},
  year         = 1993,
  editor       = {Peter Kidwelly},
  volume       = 4,
  series       = 5,
  address      = {The address of the publisher},
  month        = oct,
  organization = {The organization},
  publisher    = {The name of the publisher},
  note         = {An optional note}
}
"""

proceedings_html = """
Peter Kidwelly, editor.
<em>The title of the work</em>, volume&nbsp;4 of 5, The address of the publisher, October 1993. The organization, The name of the publisher.
An optional note.
"""

techreport_bib = """
@techreport{techreport,
  author       = {Peter Lambert}, 
  title        = {The title of the work},
  institution  = {The institution that published},
  year         = 1993,
  number       = 2,
  address      = {The address of the publisher},
  month        = nov,
  note         = {An optional note}
}
"""

techreport_html = """
Peter Lambert.
The title of the work.
Technical Report 2, The institution that published, The address of the publisher, November 1993.
An optional note.
"""

unpublished_bib = """
@unpublished{unpublished,
  author       = {Peter Marcheford}, 
  title        = {The title of the work},
  note         = {An optional note},
  month        = mar,
  year         = 1993
}
"""

unpublished_html = """
Peter Marcheford.
The title of the work.
An optional note, March 1993.
"""

online_bib = """
@online{online,
    author = {Peter Nash},
    title = {The title of the work},
    year = 1993,
    url = {http://www.google.com/},
    urldate = {1993-03-07},
}
"""

online_html = """
Peter Nash.
The title of the work.
1993.
URL: <a href="http://www.google.com/">http://www.google.com/</a> (visited on 1993-03-07).
"""


@pytest.mark.parametrize("bib,html", [
    (article_bib, article_html),
    (book_bib, book_html),
    (booklet_bib, booklet_html),
    (inbook_bib, inbook_html),
    (incollection_bib, incollection_html),
    (manual_bib, manual_html),
    (masterthesis_bib, masterthesis_html),
    (misc_bib, misc_html),
    (online_bib, online_html),
    (phdthesis_bib, phdthesis_html),
    (proceedings_bib, proceedings_html),
    (techreport_bib, techreport_html),
    (unpublished_bib, unpublished_html),
])
def test_backend_html(bib, html):
    style = UnsrtStyle()
    backend = HtmlBackend()
    bib_data = parse_string(bib, 'bibtex')
    for formatted_entry in style.format_entries(bib_data.entries.values()):
        render = formatted_entry.text.render(backend)
        print(render)
        assert render.strip() == html.strip()
