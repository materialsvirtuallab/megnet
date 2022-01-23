# vim:fileencoding=utf-8

from __future__ import unicode_literals

from pybtex.database import BibliographyData, Entry, Person

reference_data = BibliographyData(
    entries=[
        ('viktorov-metodoj', Entry('book',
            fields=[
                ('publisher', u'Л.: <<Химия>>'),
                ('language', u'russian'),
                ('year', u'1977'),
                ('title', u'Методы вычисления физико-химических величин и прикладные расчёты'),
            ],
            persons={'author': [Person(u'Викторов, Михаил Маркович')]}
        )),
        ('test-booklet', Entry('booklet',
            fields=[
                ('year', u'2006'),
                ('month', u'January'),
                ('title', u'Just a booklet'),
                ('howpublished', u'Published by Foo'),
                ('language', u'english'),
                ('address', u'Moscow'),
            ],
            persons={'author': [Person(u'de Last, Jr., First Middle')]}
        )),
        ('ruckenstein-diffusion', Entry('article',
            fields=[
                ('language', u'english'),
                ('title', u'Predicting the Diffusion Coefficient in Supercritical Fluids'),
                ('journal', u'Ind. Eng. Chem. Res.'),
                ('volume', u'36'),
                ('year', u'1997'),
                ('pages', u'888-895'),
            ],
            persons={'author': [Person(u'Liu, Hongquin'), Person(u'Ruckenstein, Eli')]},
        )),
        ('test-inbook', Entry('inbook',
            fields=[
                ('title', u'Some Title'),
                ('booktitle', u'Some Good Book'),
                ('series', u'Some series'),
                ('number', u'3'),
                ('publisher', u'Some Publisher'),
                ('edition', u'Second'),
                ('language', u'english'),
                ('year', u'1933'),
                ('pages', u'44--59'),
            ],
            persons={'author': [Person(u'Jackson, Peter')]}
        )),
    ],
    preamble=['%%% pybtex example file']
)
