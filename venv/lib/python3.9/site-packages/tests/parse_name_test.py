# vim:fileencoding=utf-8

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


from __future__ import unicode_literals

import pytest

from pybtex import errors
from pybtex.database import InvalidNameString, Person

# name, (bibtex_first, prelast, last, lineage
# as parsed by the bibtex program itself
sample_names = [
    ('A. E.                   Siegman', (['A.', 'E.'], [], ['Siegman'], []), None),
    ('A. G. W. Cameron', (['A.', 'G.', 'W.'], [], ['Cameron'], []), None),
    ('A. Hoenig', (['A.'], [], ['Hoenig'], []), None),
    ('A. J. Van Haagen', (['A.', 'J.', 'Van'], [], ['Haagen'], []), None),
    ('A. S. Berdnikov', (['A.', 'S.'], [], ['Berdnikov'], []), None),
    ('A. Trevorrow', (['A.'], [], ['Trevorrow'], []), None),
    ('Adam H. Lewenberg', (['Adam', 'H.'], [], ['Lewenberg'], []), None),
    ('Addison-Wesley Publishing Company',
    (['Addison-Wesley', 'Publishing'], [], ['Company'], []), None),
    ('Advogato (Raph Levien)', (['Advogato', '(Raph'], [], ['Levien)'], []), None),
    ('Andrea de Leeuw van Weenen',
    (['Andrea'], ['de', 'Leeuw', 'van'], ['Weenen'], []), None),
    ('Andreas Geyer-Schulz', (['Andreas'], [], ['Geyer-Schulz'], []), None),
    ("Andr{\\'e} Heck", (["Andr{\\'e}"], [], ['Heck'], []), None),
    ('Anne Br{\\"u}ggemann-Klein', (['Anne'], [], ['Br{\\"u}ggemann-Klein'], []), None),
    ('Anonymous', ([], [], ['Anonymous'], []), None),
    ('B. Beeton', (['B.'], [], ['Beeton'], []), None),
    ('B. Hamilton Kelly', (['B.', 'Hamilton'], [], ['Kelly'], []), None),
    ('B. V. Venkata Krishna Sastry',
    (['B.', 'V.', 'Venkata', 'Krishna'], [], ['Sastry'], []), None),
    ('Benedict L{\\o}fstedt', (['Benedict'], [], ['L{\\o}fstedt'], []), None),
    ('Bogus{\\l}aw Jackowski', (['Bogus{\\l}aw'], [], ['Jackowski'], []), None),
    ('Christina A. L.\\ Thiele',
    # (['Christina', 'A.', 'L.\\'], [], ['Thiele'], []), None),
    (['Christina', 'A.', 'L.'], [], ['Thiele'], []), None),  # BibTeX incompatible: treat "\ " as a space
    ("D. Men'shikov", (['D.'], [], ["Men'shikov"], []), None),
    ("Darko \\v{Z}ubrini{\\'c}", (['Darko'], [], ["\\v{Z}ubrini{\\'c}"], []), None),
    ("Dunja Mladeni{\\'c}", (['Dunja'], [], ["Mladeni{\\'c}"], []), None),
    ('Edwin V. {Bell, II}', (['Edwin', 'V.'], [], ['{Bell, II}'], []), None),
    ('Frank G. {Bennett, Jr.}', (['Frank', 'G.'], [], ['{Bennett, Jr.}'], []), None),
    ("Fr{\\'e}d{\\'e}ric Boulanger",
    (["Fr{\\'e}d{\\'e}ric"], [], ['Boulanger'], []), None),
    ('Ford, Jr., Henry', (['Henry'], [], ['Ford'], ['Jr.']), None),
    ('mr Ford, Jr., Henry', (['Henry'], ['mr'], ['Ford'], ['Jr.']), None),
    ('Fukui Rei', (['Fukui'], [], ['Rei'], []), None),
    ('G. Gr{\\"a}tzer', (['G.'], [], ['Gr{\\"a}tzer'], []), None),
    ('George Gr{\\"a}tzer', (['George'], [], ['Gr{\\"a}tzer'], []), None),
    ('Georgia K. M. Tobin', (['Georgia', 'K.', 'M.'], [], ['Tobin'], []), None),
    ('Gilbert van den Dobbelsteen',
    (['Gilbert'], ['van', 'den'], ['Dobbelsteen'], []), None),
    ('Gy{\\"o}ngyi Bujdos{\\\'o}', (['Gy{\\"o}ngyi'], [], ["Bujdos{\\'o}"], []), None),
    ('Helmut J{\\"u}rgensen', (['Helmut'], [], ['J{\\"u}rgensen'], []), None),
    ('Herbert Vo{\\ss}', (['Herbert'], [], ['Vo{\\ss}'], []), None),
    ("H{\\'a}n Th{\\^e}\\llap{\\raise 0.5ex\\hbox{\\'{\\relax}}}                  Th{\\'a}nh",
    (["H{\\'a}n", "Th{\\^e}\\llap{\\raise 0.5ex\\hbox{\\'{\\relax}}}"],
    [],
    ["Th{\\'a}nh"],
    []), None),
    ("H{\\`a}n Th\\^e\\llap{\\raise0.5ex\\hbox{\\'{\\relax}}}                  Th{\\`a}nh",
    (['H{\\`a}n', "Th\\^e\\llap{\\raise0.5ex\\hbox{\\'{\\relax}}}"],
    [],
    ['Th{\\`a}nh'],
    []), None),
    ("J. Vesel{\\'y}", (['J.'], [], ["Vesel{\\'y}"], []), None),
    ("Javier Rodr\\'{\\i}guez Laguna",
    (['Javier', "Rodr\\'{\\i}guez"], [], ['Laguna'], []), None),
    ("Ji\\v{r}\\'{\\i} Vesel{\\'y}",
    (["Ji\\v{r}\\'{\\i}"], [], ["Vesel{\\'y}"], []), None),
    ("Ji\\v{r}\\'{\\i} Zlatu{\\v{s}}ka",
    (["Ji\\v{r}\\'{\\i}"], [], ['Zlatu{\\v{s}}ka'], []), None),
    ("Ji\\v{r}{\\'\\i} Vesel{\\'y}",
    (["Ji\\v{r}{\\'\\i}"], [], ["Vesel{\\'y}"], []), None),
    ("Ji\\v{r}{\\'{\\i}}Zlatu{\\v{s}}ka",
    ([], [], ["Ji\\v{r}{\\'{\\i}}Zlatu{\\v{s}}ka"], []), None),
    ('Jim Hef{}feron', (['Jim'], [], ['Hef{}feron'], []), None),
    ('J{\\"o}rg Knappen', (['J{\\"o}rg'], [], ['Knappen'], []), None),
    ('J{\\"o}rgen L. Pind', (['J{\\"o}rgen', 'L.'], [], ['Pind'], []), None),
    ("J{\\'e}r\\^ome Laurens", (["J{\\'e}r\\^ome"], [], ['Laurens'], []), None),
    ('J{{\\"o}}rg Knappen', (['J{{\\"o}}rg'], [], ['Knappen'], []), None),
    ('K. Anil Kumar', (['K.', 'Anil'], [], ['Kumar'], []), None),
    ("Karel Hor{\\'a}k", (['Karel'], [], ["Hor{\\'a}k"], []), None),
    ("Karel P\\'{\\i}{\\v{s}}ka", (['Karel'], [], ["P\\'{\\i}{\\v{s}}ka"], []), None),
    ("Karel P{\\'\\i}{\\v{s}}ka", (['Karel'], [], ["P{\\'\\i}{\\v{s}}ka"], []), None),
    ("Karel Skoup\\'{y}", (['Karel'], [], ["Skoup\\'{y}"], []), None),
    ("Karel Skoup{\\'y}", (['Karel'], [], ["Skoup{\\'y}"], []), None),
    ('Kent McPherson', (['Kent'], [], ['McPherson'], []), None),
    ('Klaus H{\\"o}ppner', (['Klaus'], [], ['H{\\"o}ppner'], []), None),
    ('Lars Hellstr{\\"o}m', (['Lars'], [], ['Hellstr{\\"o}m'], []), None),
    ('Laura Elizabeth Jackson',
    (['Laura', 'Elizabeth'], [], ['Jackson'], []), None),
    ("M. D{\\'{\\i}}az", (['M.'], [], ["D{\\'{\\i}}az"], []), None),
    ('M/iche/al /O Searc/oid', (['M/iche/al', '/O'], [], ['Searc/oid'], []), None),
    ("Marek Ry{\\'c}ko", (['Marek'], [], ["Ry{\\'c}ko"], []), None),
    ('Marina Yu. Nikulina', (['Marina', 'Yu.'], [], ['Nikulina'], []), None),
    ("Max D{\\'{\\i}}az", (['Max'], [], ["D{\\'{\\i}}az"], []), None),
    ('Merry Obrecht Sawdey', (['Merry', 'Obrecht'], [], ['Sawdey'], []), None),
    ("Miroslava Mis{\\'a}kov{\\'a}",
    (['Miroslava'], [], ["Mis{\\'a}kov{\\'a}"], []), None),
    ('N. A. F. M. Poppelier', (['N.', 'A.', 'F.', 'M.'], [], ['Poppelier'], []), None),
    ('Nico A. F. M. Poppelier',
    (['Nico', 'A.', 'F.', 'M.'], [], ['Poppelier'], []), None),
    ('Onofrio de Bari', (['Onofrio'], ['de'], ['Bari'], []), None),
    ("Pablo Rosell-Gonz{\\'a}lez", (['Pablo'], [], ["Rosell-Gonz{\\'a}lez"], []), None),
    ('Paco La                  Bruna', (['Paco', 'La'], [], ['Bruna'], []), None),
    ('Paul                  Franchi-Zannettacci',
    (['Paul'], [], ['Franchi-Zannettacci'], []), None),
    ('Pavel \\v{S}eve\\v{c}ek', (['Pavel'], [], ['\\v{S}eve\\v{c}ek'], []), None),
    ('Petr Ol{\\v{s}}ak', (['Petr'], [], ['Ol{\\v{s}}ak'], []), None),
    ("Petr Ol{\\v{s}}{\\'a}k", (['Petr'], [], ["Ol{\\v{s}}{\\'a}k"], []), None),
    ('Primo\\v{z} Peterlin', (['Primo\\v{z}'], [], ['Peterlin'], []), None),
    ('Prof. Alban Grimm', (['Prof.', 'Alban'], [], ['Grimm'], []), None),
    ("P{\\'e}ter Husz{\\'a}r", (["P{\\'e}ter"], [], ["Husz{\\'a}r"], []), None),
    ("P{\\'e}ter Szab{\\'o}", (["P{\\'e}ter"], [], ["Szab{\\'o}"], []), None),
    ('Rafa{\\l}\\.Zbikowski', ([], [], ['Rafa{\\l}\\.Zbikowski'], []), None),
    ('Rainer Sch{\\"o}pf', (['Rainer'], [], ['Sch{\\"o}pf'], []), None),
    ('T. L. (Frank) Pappas', (['T.', 'L.', '(Frank)'], [], ['Pappas'], []), None),
    ('TUG 2004 conference', (['TUG', '2004'], [], ['conference'], []), None),

    # von part with BibTeX special characters
    ('TUG {\\sltt DVI} Driver Standards Committee',
    (['TUG', '{\\sltt DVI}', 'Driver', 'Standards'], [], ['Committee'], []), None),
    ('TUG {\\sltt xDVIx} Driver Standards Committee',
    (['TUG'], ['{\\sltt xDVIx}'], ['Driver', 'Standards', 'Committee'], []), None),

    ('University of M{\\"u}nster',
    (['University'], ['of'], ['M{\\"u}nster'], []), None),
    ('Walter van der Laan', (['Walter'], ['van', 'der'], ['Laan'], []), None),
    ('Wendy G.                  McKay', (['Wendy', 'G.'], [], ['McKay'], []), None),
    ('Wendy McKay', (['Wendy'], [], ['McKay'], []), None),
    ('W{\\l}odek Bzyl', (['W{\\l}odek'], [], ['Bzyl'], []), None),
    ('\\LaTeX Project Team', (['\\LaTeX', 'Project'], [], ['Team'], []), None),
    ('\\rlap{Lutz Birkhahn}', ([], [], ['\\rlap{Lutz Birkhahn}'], []), None),
    ('{Jim Hef{}feron}', ([], [], ['{Jim Hef{}feron}'], []), None),
    ('{Kristoffer H\\o{}gsbro Rose}',
    ([], [], ['{Kristoffer H\\o{}gsbro Rose}'], []), None),
    ('{TUG} {Working} {Group} on a {\\TeX} {Directory}                  {Structure}',
    (['{TUG}', '{Working}', '{Group}'],
    ['on', 'a'],
    ['{\\TeX}', '{Directory}', '{Structure}'],
    []), None),
    ('{The \\TUB{} Team}', ([], [], ['{The \\TUB{} Team}'], []), None),
    ('{\\LaTeX} project team', (['{\\LaTeX}'], ['project'], ['team'], []), None),
    ('{\\NTG{} \\TeX{} future working group}',
    ([], [], ['{\\NTG{} \\TeX{} future working group}'], []), None),
    ('{{\\LaTeX\\,3} Project Team}',
    ([], [], ['{{\\LaTeX\\,3} Project Team}'], []), None),
    ('Johansen Kyle, Derik Mamania M.',
    (['Derik', 'Mamania', 'M.'], [], ['Johansen', 'Kyle'], []), None),
    ("Johannes Adam Ferdinand Alois Josef Maria Marko d'Aviano "
    'Pius von und zu Liechtenstein',
    (['Johannes', 'Adam', 'Ferdinand', 'Alois', 'Josef', 'Maria', 'Marko'],
    ["d'Aviano", 'Pius', 'von', 'und', 'zu'], ['Liechtenstein'],[]), None),

    (r'Brand\~{a}o, F', (['F'], [], [r'Brand\~{a}o'], []), None),
    # but BibTeX parses it like this:
    # (r'Brand\~{a}o, F', (['F'], [], ['Brand\\', '{a}o'], []), None),

    # incorrectly formatted name strings below

    # too many commas
    ('Chong, B. M., Specia, L., & Mitkov, R.',
    (['Specia', 'L.', '&', 'Mitkov', 'R.'], [], ['Chong'], ['B.', 'M.']),
    [InvalidNameString('Chong, B. M., Specia, L., & Mitkov, R.')]
    ),
    # too many commas, sloppy whitespace
    ('LeCun, Y. ,      Bottou,   L . , Bengio, Y. ,  Haffner ,  P',
    (['Bottou', 'L', '.', 'Bengio', 'Y.', 'Haffner', 'P'], [], ['LeCun'], ['Y.']),
    [InvalidNameString('LeCun, Y. ,      Bottou,   L . , Bengio, Y. ,  Haffner ,  P')]),
]


@pytest.mark.parametrize(
    ["name", "correct_result", "expected_errors"],
    sample_names
)
def test_parse_name(name, correct_result, expected_errors):
    if expected_errors is None:
        expected_errors = []

    with errors.capture() as captured_errors:
        person = Person(name)

    result = (person.bibtex_first_names, person.prelast_names, person.last_names, person.lineage_names)
    assert result == correct_result
    assert captured_errors == expected_errors
