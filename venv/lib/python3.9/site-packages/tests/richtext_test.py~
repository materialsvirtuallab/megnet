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


from __future__ import absolute_import, unicode_literals

import re
from abc import ABCMeta, abstractmethod
from unittest import TestCase

import pytest
import six

from pybtex import textutils
from pybtex.richtext import HRef, Protected, String, Symbol, Tag, Text, nbsp


class TextTestMixin(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def test__init__(self):
        raise NotImplementedError

    @abstractmethod
    def test__str__(self):
        raise NotImplementedError

    @abstractmethod
    def test__eq__(self):
        raise NotImplementedError

    @abstractmethod
    def test__len__(self):
        raise NotImplementedError

    @abstractmethod
    def test__contains__(self):
        raise NotImplementedError

    @abstractmethod
    def test__getitem__(self):
        raise NotImplementedError

    @abstractmethod
    def test__add__(self):
        raise NotImplementedError

    @abstractmethod
    def test_append(self):
        raise NotImplementedError

    @abstractmethod
    def test_join(self):
        raise NotImplementedError

    @abstractmethod
    def test_split(self):
        raise NotImplementedError

    @abstractmethod
    def test_startswith(self):
        raise NotImplementedError

    @abstractmethod
    def test_endswith(self):
        raise NotImplementedError

    @abstractmethod
    def test_isalpha(self):
        raise NotImplementedError

    @abstractmethod
    def test_capfirst(self):
        raise NotImplementedError

    @abstractmethod
    def test_capitalize(self):
        raise NotImplementedError

    @abstractmethod
    def test_add_period(self):
        raise NotImplementedError

    @abstractmethod
    def test_lower(self):
        raise NotImplementedError

    @abstractmethod
    def test_upper(self):
        raise NotImplementedError

    @abstractmethod
    def test_render_as(self):
        raise NotImplementedError


class TestText(TextTestMixin, TestCase):
    def test__init__(self):
        assert six.text_type(Text('a', '', 'c')) == 'ac'
        assert six.text_type(Text('a', Text(), 'c')) == 'ac'

        text = Text(Text(), Text('mary ', 'had ', 'a little lamb'))
        assert six.text_type(text) == 'mary had a little lamb'

        text = six.text_type(Text('a', Text('b', 'c'), Tag('em', 'x'), Symbol('nbsp'), 'd'))
        assert text == 'abcx<nbsp>d'

        with pytest.raises(ValueError):
            Text({})

        with pytest.raises(ValueError):
            Text(0, 0)

    def test__eq__(self):
        assert Text() == Text()
        assert not (Text() != Text())

        assert Text('Cat') == Text('Cat')
        assert not (Text('Cat') != Text('Cat'))
        assert Text('Cat', ' tail') == Text('Cat tail')
        assert not (Text('Cat', ' tail') != Text('Cat tail'))

        assert Text('Cat') != Text('Dog')
        assert not (Text('Cat') == Text('Dog'))

    def test__len__(self):
        assert len(Text()) == 0
        assert len(Text('Never', ' ', 'Knows', ' ', 'Best')) == len('Never Knows Best')
        assert len(Text('Never', ' ', Tag('em', 'Knows', ' '), 'Best')) == len('Never Knows Best')
        assert len(Text('Never', ' ', Tag('em', HRef('/', 'Knows'), ' '), 'Best')) == len('Never Knows Best')

    def test__str__(self):
        assert six.text_type(Text()) == ''
        assert six.text_type(Text(u'Чудаки украшают мир')) == u'Чудаки украшают мир'

    def test__contains__(self):
        text = Text('mary ', 'had ', 'a little lamb')
        assert 'mary' in text
        assert not 'Mary' in text
        assert 'had a little' in text

        text = Text('a', 'b', 'c')
        assert 'abc' in text

    def test_capfirst(self):
        text = Text('dear ', 'Alice')
        assert six.text_type(text.capfirst()) == 'Dear Alice'

    def test_capitalize(self):
        text = Text('mary ', 'had ', 'a Little Lamb')
        assert six.text_type(text.capitalize()) == 'Mary had a little lamb'

    def test__add__(self):
        t = Text('a')
        assert six.text_type(t + 'b') == 'ab'
        assert six.text_type(t + t) == 'aa'
        assert six.text_type(t) == 'a'

    def test__getitem__(self):
        t = Text('123', Text('456', Text('78'), '9'), '0')

        with pytest.raises(TypeError):
            1 in t

        assert t == Text('1234567890')
        assert t[:0] == Text('')
        assert t[:1] == Text('1')
        assert t[:3] == Text('123')
        assert t[:5] == Text('12345')
        assert t[:7] == Text('1234567')
        assert t[:10] == Text('1234567890')
        assert t[:100] == Text('1234567890')
        assert t[:-100] == Text('')
        assert t[:-10] == Text('')
        assert t[:-9] == Text('1')
        assert t[:-7] == Text('123')
        assert t[:-5] == Text('12345')
        assert t[:-3] == Text('1234567')
        assert t[-100:] == Text('1234567890')
        assert t[-10:] == Text('1234567890')
        assert t[-9:] == Text('234567890')
        assert t[-7:] == Text('4567890')
        assert t[-5:] == Text('67890')
        assert t[-3:] == Text('890')
        assert t[1:] == Text('234567890')
        assert t[3:] == Text('4567890')
        assert t[5:] == Text('67890')
        assert t[7:] == Text('890')
        assert t[10:] == Text('')
        assert t[100:] == Text('')
        assert t[0:10] == Text('1234567890')
        assert t[0:100] == Text('1234567890')
        assert t[2:3] == Text('3')
        assert t[2:4] == Text('34')
        assert t[3:7] == Text('4567')
        assert t[4:7] == Text('567')
        assert t[4:7] == Text('567')
        assert t[7:9] == Text('89')
        assert t[100:200] == Text('')

        t = Text('123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:3] == Text('123')
        assert t[:5] == Text('123', Tag('em', '45'))
        assert t[:7] == Text('123', Tag('em', '456', HRef('/', '7')))
        assert t[:10] == Text('123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:100] == Text('123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:-7] == Text('123')
        assert t[:-5] == Text('123', Tag('em', '45'))
        assert t[:-3] == Text('123', Tag('em', '456', HRef('/', '7')))

    def test_append(self):
        text = Tag('strong', 'Chuck Norris')
        assert (text +  ' wins!').render_as('html') == '<strong>Chuck Norris</strong> wins!'
        assert text.append(' wins!').render_as('html') == '<strong>Chuck Norris wins!</strong>'
        text = HRef('/', 'Chuck Norris')
        assert (text +  ' wins!').render_as('html') == '<a href="/">Chuck Norris</a> wins!'
        assert text.append(' wins!').render_as('html') == '<a href="/">Chuck Norris wins!</a>'

    def test_upper(self):
        text = Text('mary ', 'had ', 'a little lamb')
        assert six.text_type(text.upper()) == 'MARY HAD A LITTLE LAMB'

    def test_lower(self):
        text = Text('mary ', 'had ', 'a little lamb')
        assert six.text_type(text.lower()) == 'mary had a little lamb'

    def test_startswith(self):
        assert not Text().startswith('.')
        assert not Text().startswith(('.', '!'))

        text = Text('mary ', 'had ', 'a little lamb')
        assert not text.startswith('M')
        assert text.startswith('m')

        text = Text('a', 'b', 'c')
        assert text.startswith('ab')

        assert Text('This is good').startswith(('This', 'That'))
        assert not Text('This is good').startswith(('That', 'Those'))

    def test_endswith(self):
        assert not Text().endswith('.')
        assert not Text().endswith(('.', '!'))

        text = Text('mary ', 'had ', 'a little lamb')
        assert not text.endswith('B')
        assert text.endswith('b')

        text = Text('a', 'b', 'c')
        assert text.endswith('bc')

        assert Text('This is good').endswith(('good', 'wonderful'))
        assert not Text('This is good').endswith(('bad', 'awful'))

    def test_isalpha(self):
        assert not Text().isalpha()
        assert not Text('a b c').isalpha()
        assert Text('abc').isalpha()
        assert Text(u'文字').isalpha()
        assert Text('ab', Tag('em', 'cd'), 'ef').isalpha()
        assert not Text('ab', Tag('em', '12'), 'ef').isalpha()

    def test_join(self):
        assert six.text_type(Text(' ').join(['a', Text('b c')])) == 'a b c'
        assert six.text_type(Text(nbsp).join(['a', 'b', 'c'])) == 'a<nbsp>b<nbsp>c'
        assert six.text_type(nbsp.join(['a', 'b', 'c'])) == 'a<nbsp>b<nbsp>c'
        assert six.text_type(String('-').join(['a', 'b', 'c'])) == 'a-b-c'
        result = Tag('em', ' and ').join(['a', 'b', 'c']).render_as('html')
        assert result == 'a<em> and </em>b<em> and </em>c'
        result = HRef('/', ' and ').join(['a', 'b', 'c']).render_as('html')
        assert result == 'a<a href="/"> and </a>b<a href="/"> and </a>c'

    def test_split(self):
        assert Text().split() == []
        assert Text().split('abc') == [Text()]
        assert Text('a').split() == [Text('a')]
        assert Text('a ').split() == [Text('a')]
        assert Text('   a   ').split() == [Text('a')]
        assert Text('a + b').split() == [Text('a'), Text('+'), Text('b')]
        assert Text('a + b').split(' + ') == [Text('a'), Text('b')]
        assert Text('a + b').split(re.compile(r'\s')) == [Text('a'), Text('+'), Text('b')]
        assert Text('abc').split('xyz') == [Text('abc')]
        assert Text('---').split('--') == [Text(), Text('-')]
        assert Text('---').split('-') == [Text(), Text(), Text(), Text()]

    def test_add_period(self):
        assert Text().endswith(('.', '!', '?')) == False
        assert textutils.is_terminated(Text()) == False

        assert six.text_type(Text().add_period()) == ''

        text = Text("That's all, folks")
        assert six.text_type(text.add_period()) == "That's all, folks."

    def test_render_as(self):
        string = Text(u'Detektivbyrån & friends')
        assert string.render_as('text') == u'Detektivbyrån & friends'
        assert string.render_as('html') == u'Detektivbyrån &amp; friends'


class TestString(TextTestMixin, TestCase):
    def test__init__(self):
        assert String().value == ''
        assert String('').value == ''
        assert String('Wa', '', 'ke', ' ', 'up!').value == 'Wake up!'

    def test__eq__(self):
        assert String() != ''
        assert String('') != ''
        assert String() == String()
        assert String('') == String()
        assert String('', '', '') == String()
        assert String('Wa', '', 'ke', ' ', 'up!') == String('Wake up!')

    def test__len__(self):
        assert len(String()) == len(String('')) == 0

        val = 'test string'
        assert len(String(val)) == len(val)

    def test__str__(self):
        val = u'Detektivbyrån'
        assert six.text_type(String(val)) == val

    def test__contains__(self):
        assert '' in String()
        assert 'abc' not in String()
        assert '' in String(' ')
        assert ' + ' in String('2 + 2')

    def test__getitem__(self):
        digits = String('0123456789')
        assert digits[0] != '0'
        assert digits[0] == String('0')

    def test__add__(self):
        assert String('Python') + String(' 3') != 'Python 3'
        assert String('Python') + String(' 3') == Text('Python 3')
        assert String('A').lower() == String('a')
        assert six.text_type(String('Python') + String(' ') + String('3')) == 'Python 3'
        assert six.text_type(String('Python') + Text(' ') + String('3')) == 'Python 3'
        assert six.text_type(String('Python') + ' ' + '3') == 'Python 3'
        assert six.text_type(String('Python').append(' 3')) == 'Python 3'

    def test_startswith(self):
        assert not String().startswith('n')
        assert not String('').startswith('n')
        assert not String().endswith('n')
        assert not String('').endswith('n')
        assert not String('November.').startswith('n')
        assert String('November.').startswith('N')

    def test_endswith(self):
        assert not String().endswith('.')
        assert not String().endswith(('.', '!'))
        assert not String('November.').endswith('r')
        assert String('November.').endswith('.')
        assert String('November.').endswith(('.', '!'))
        assert not String('November.').endswith(('?', '!'))

    def test_isalpha(self):
        assert not String().isalpha()
        assert not String('a b c').isalpha()
        assert String('abc').isalpha()
        assert String(u'文字').isalpha()

    def test_append(self):
        assert String().append('') == Text()
        text = String('The').append(' Adventures of ').append('Tom Sawyer')
        assert text == Text('The Adventures of Tom Sawyer')

    def test_lower(self):
        assert String('').lower() == String()
        assert String('A').lower() == String('a')
        assert String('November').lower() == String('november')

    def test_upper(self):
        assert String('').upper() == String()
        assert String('a').upper() == String('A')
        assert String('November').upper() == String('NOVEMBER')

    def test_split(self):
        assert String().split() == []
        assert String().split('abc') == [String('')]
        assert String('a').split() == [String('a')]
        assert String('a ').split() == [String('a')]
        assert String('a + b').split() == [String('a'), String('+'), String('b')]
        assert String('a + b').split(' + ') == [String('a'), String('b')]
        assert String('a + b').split(re.compile(r'\s')) == [String('a'), String('+'), String('b')]

    def test_join(self):
        assert String().join([]) == Text()
        assert String('nothing to see here').join([]) == Text()
        assert String().join(['a', 'b', 'c']) == Text('abc')
        assert String(', ').join(['tomatoes']) == Text('tomatoes')
        assert String(', ').join(['tomatoes', 'cucumbers']) == Text('tomatoes, cucumbers')
        assert String(', ').join(['tomatoes', 'cucumbers', 'lemons']) == Text('tomatoes, cucumbers, lemons')

    def test_capfirst(self):
        assert six.text_type(String('').capitalize()) == ''
        assert six.text_type(String('november december').capitalize()) == 'November december'
        assert six.text_type(String('November December').capitalize()) == 'November december'
        assert six.text_type(String('NOVEMBER DECEMBER').capitalize()) == 'November december'

    def test_capitalize(self):
        assert six.text_type(String('').capfirst()) == ''
        assert six.text_type(String('november').capfirst()) == 'November'
        assert six.text_type(String('November').capfirst()) == 'November'
        assert six.text_type(String('november december').capfirst()) == 'November december'
        assert six.text_type(String('November December').capfirst()) == 'November December'
        assert six.text_type(String('NOVEMBER DECEMBER').capfirst()) == 'NOVEMBER DECEMBER'

    def test_add_period(self):
        assert six.text_type(String('').add_period()) == ''
        assert six.text_type(String('').add_period('!')) == ''
        assert six.text_type(String('').add_period().add_period()) == ''
        assert six.text_type(String('').add_period().add_period('!')) == ''
        assert six.text_type(String('').add_period('!').add_period()) == ''
        six.text_type(String('November').add_period()) == 'November.'
        result = six.text_type(String('November').add_period().add_period())
        assert result == 'November.'

    def test_render_as(self):
        string = String(u'Detektivbyrån & friends')
        assert string.render_as('text') == u'Detektivbyrån & friends'
        assert string.render_as('html') == u'Detektivbyrån &amp; friends'


class TestTag(TextTestMixin, TestCase):
    def test__init__(self):
        empty = Tag('em')
        assert six.text_type(empty) == ''

        text = Text('This ', Tag('em', 'is'), ' good')
        assert 'This is' in six.text_type(text)
        assert six.text_type(text).startswith('This is')
        assert six.text_type(text).endswith('is good')

    def test__eq__(self):
        assert Tag('em', '') != ''
        assert Tag('em', '') != Text()
        assert Tag('em', '') != Tag('strong', '')
        assert Tag('em', '') == Tag('em', '')

        assert Tag('em', 'good') != Tag('em', 'bad')
        assert Tag('em', 'good') != Text('good')
        assert Tag('em', 'good') == Tag('em', 'good')
        assert not (Tag('em', 'good') != Tag('em', 'good'))

    def test__len__(self):
        val = 'Tomato apple!'
        assert len(Tag('em', val)) == len(val)

    def test__str__(self):
        empty = Tag('em')
        assert six.text_type(empty.lower()) == ''
        assert six.text_type(empty.capitalize()) == ''
        assert six.text_type(empty.add_period()) == ''

        assert six.text_type(Tag('strong', u'ねここねこ')) == u'ねここねこ'

    def test__contains__(self):
        tag = Tag('em', Text(), Text('mary ', 'had ', 'a little lamb'))
        assert 'mary' in tag
        assert 'Mary' not in tag
        assert 'had a little' in tag

        text = Text('This ', Tag('em', 'is'), ' good')
        assert not 'This is' in text

    def test__getitem__(self):
        t = Tag('em', '1234567890')

        with pytest.raises(TypeError):
            1 in t

        assert t == Tag('em', '1234567890')
        assert t[:] == t
        assert t[:0] == Tag('em', '')
        assert t[:1] == Tag('em', '1')
        assert t[:3] == Tag('em', '123')
        assert t[:5] == Tag('em', '12345')
        assert t[:7] == Tag('em', '1234567')
        assert t[:10] == Tag('em', '1234567890')
        assert t[:100] == Tag('em', '1234567890')
        assert t[:-100] == Tag('em', '')
        assert t[:-10] == Tag('em', '')
        assert t[:-9] == Tag('em', '1')
        assert t[:-7] == Tag('em', '123')
        assert t[:-5] == Tag('em', '12345')
        assert t[:-3] == Tag('em', '1234567')
        assert t[-100:] == Tag('em', '1234567890')
        assert t[-10:] == Tag('em', '1234567890')
        assert t[-9:] == Tag('em', '234567890')
        assert t[-7:] == Tag('em', '4567890')
        assert t[-5:] == Tag('em', '67890')
        assert t[-3:] == Tag('em', '890')
        assert t[1:] == Tag('em', '234567890')
        assert t[3:] == Tag('em', '4567890')
        assert t[5:] == Tag('em', '67890')
        assert t[7:] == Tag('em', '890')
        assert t[10:] == Tag('em', '')
        assert t[100:] == Tag('em', '')
        assert t[0:10] == Tag('em', '1234567890')
        assert t[0:100] == Tag('em', '1234567890')
        assert t[2:3] == Tag('em', '3')
        assert t[2:4] == Tag('em', '34')
        assert t[3:7] == Tag('em', '4567')
        assert t[4:7] == Tag('em', '567')
        assert t[4:7] == Tag('em', '567')
        assert t[7:9] == Tag('em', '89')
        assert t[100:200] == Tag('em', '')

        t = Tag('strong', '123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:3] == Tag('strong', '123')
        assert t[:5] == Tag('strong', '123', Tag('em', '45'))
        assert t[:7] == Tag('strong', '123', Tag('em', '456', HRef('/', '7')))
        assert t[:10] == Tag('strong', '123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:100] == Tag('strong', '123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:-7] == Tag('strong', '123')
        assert t[:-5] == Tag('strong', '123', Tag('em', '45'))
        assert t[:-3] == Tag('strong', '123', Tag('em', '456', HRef('/', '7')))

    def test__add__(self):
        assert Tag('em', '') + Tag('em', '') == Text(Tag('em', ''))
        assert Tag('em', '') + Tag('strong', '') == Text(Tag('em', ''), Tag('strong', ''))
        assert Tag('em', 'Good') + Tag('em', '') == Text(Tag('em', 'Good'))
        assert Tag('em', 'Good') + Tag('em', ' job!') == Text(Tag('em', 'Good job!'))
        assert Tag('em', 'Good') + Tag('strong', ' job!') == Text(Tag('em', 'Good'), Tag('strong', ' job!'))
        assert Tag('em', 'Good') + Text(' job!') == Text(Tag('em', 'Good'), ' job!')
        assert Text('Good') + Tag('em', ' job!') == Text('Good', Tag('em', ' job!'))

    def test_upper(self):
        tag = Tag('em', Text(), Text('mary ', 'had ', 'a little lamb'))
        assert tag.upper().render_as('html') == '<em>MARY HAD A LITTLE LAMB</em>'

    def test_lower(self):
        tag = Tag('em', Text(), Text('mary ', 'had ', 'a little lamb'))
        assert tag.lower().render_as('html') == '<em>mary had a little lamb</em>'

    def test_capfirst(self):
        tag = Tag('em', Text(), Text('mary ', 'had ', 'a Little Lamb'))
        assert tag.capfirst().render_as('html') == '<em>Mary had a Little Lamb</em>'

    def test_capitalize(self):
        tag = Tag('em', Text(), Text('mary ', 'had ', 'a Little Lamb'))
        assert tag.capitalize().render_as('html') == '<em>Mary had a little lamb</em>'

    def test_startswith(self):
        tag = Tag('em', Text(), Text('mary ', 'had ', 'a little lamb'))
        assert not tag.startswith('M')
        assert tag.startswith('m')

        tag = Tag('em', 'a', 'b', 'c')
        assert tag.startswith('ab')

        tag = Tag('em', 'This is good')
        assert tag.startswith(('This', 'That'))
        assert not tag.startswith(('That', 'Those'))

        text = Text('This ', Tag('em', 'is'), ' good')
        assert not text.startswith('This is')

    def test_isalpha(self):
        assert not Tag('em').isalpha()
        assert not Tag('em', 'a b c').isalpha()
        assert Tag('em', 'abc').isalpha()
        assert Tag('em', u'文字').isalpha()

    def test_endswith(self):
        tag = Tag('em', Text(), Text('mary ', 'had ', 'a little lamb'))
        assert not tag.endswith('B')
        assert tag.endswith('b')

        tag = Tag('em', 'a', 'b', 'c')
        assert tag.endswith('bc')

        tag = Tag('em', 'This is good')
        assert tag.endswith(('good', 'wonderful'))
        assert not tag.endswith(('bad', 'awful'))

        text = Text('This ', Tag('em', 'is'), ' good')
        assert not text.endswith('is good')

    def test_split(self):
        empty = Tag('em')
        assert empty.split() == []
        assert empty.split('abc') == [Tag('em')]

        em = Tag('em', 'Emphasized text')
        assert em.split() == [Tag('em', 'Emphasized'), Tag('em', 'text')]
        assert em.split(' ') == [Tag('em', 'Emphasized'), Tag('em', 'text')]
        assert em.split('no such text') == [em]

        text = Text('Bonnie ', Tag('em', 'and'), ' Clyde')
        assert text.split('and') == [Text('Bonnie '), Text(' Clyde')]
        assert text.split(' and ') == [text]

        text = Text('Bonnie', Tag('em', ' and '), 'Clyde')
        assert text.split('and') == [Text('Bonnie', Tag('em', ' ')), Text(Tag('em', ' '), 'Clyde')]
        assert text.split(' and ') == [Text('Bonnie'), Text('Clyde')]

        text = Text('From ', Tag('em', 'the very beginning'), ' of things')
        assert text.split() == [
            Text('From'), Text(Tag('em', 'the')), Text(Tag('em', 'very')),
            Text(Tag('em', 'beginning')), Text('of'), Text('things'),
        ]

        parts = text.split()
        assert parts == [
            Text('From'),
            Text(Tag('em', 'the')),
            Text(Tag('em', 'very')),
            Text(Tag('em', 'beginning')),
            Text('of'), Text('things'),
        ]

    def test_join(self):
        text = Text('From ', Tag('em', 'the very beginning'), ' of things')
        dashified = String('-').join(text.split())
        assert dashified == Text('From-', Tag('em', 'the'), '-', Tag('em', 'very'), '-', Tag('em', 'beginning'), '-of-things')
        dashified = Tag('em', '-').join(text.split())
        assert dashified == Text('From', Tag('em', '-the-very-beginning-'), 'of', Tag('em', '-'), 'things')

    def test_append(self):
        text = Tag('strong', 'Chuck Norris')
        assert (text +  ' wins!').render_as('html') == '<strong>Chuck Norris</strong> wins!'
        assert text.append(' wins!').render_as('html') == '<strong>Chuck Norris wins!</strong>'

        text = Tag('em', 'Look here')
        assert (text +  '!').render_as('html') == '<em>Look here</em>!'
        assert text.append('!').render_as('html') == '<em>Look here!</em>'

    def test_add_period(self):
        text = Tag('em', Text("That's all, folks"))
        assert text.add_period().render_as('html') == "<em>That's all, folks.</em>"
        assert text.add_period().add_period().render_as('html') == "<em>That's all, folks.</em>"

        text = Text("That's all, ", Tag('em', 'folks'))
        assert text.add_period().render_as('html') == "That's all, <em>folks</em>."
        assert text.add_period().add_period().render_as('html') == "That's all, <em>folks</em>."

        text = Text("That's all, ", Tag('em', 'folks.'))
        assert text.add_period().render_as('html') == "That's all, <em>folks.</em>"

        text = Text("That's all, ", Tag('em', 'folks'))
        assert text.add_period('!').render_as('html') == "That's all, <em>folks</em>!"

        text = text.add_period('!').add_period('.').render_as('html')
        assert text == "That's all, <em>folks</em>!"

        tag = Tag('em', Text(), Text('mary ', 'had ', 'a little lamb'))
        assert tag.add_period().render_as('html') == '<em>mary had a little lamb.</em>'
        assert tag.add_period().add_period().render_as('html') == '<em>mary had a little lamb.</em>'

    def test_render_as(self):
        empty = Tag('em')
        assert empty.render_as('html') == ''
        assert empty.render_as('latex') == ''

        tag = Tag('em', 'a', 'b')
        assert tag.render_as('html') == '<em>ab</em>'
        assert tag.render_as('latex') == '\\emph{ab}'

        em = Tag('em', 'Emphasized text')
        assert em.render_as('latex') == '\\emph{Emphasized text}'
        assert em.upper().render_as('latex') == '\\emph{EMPHASIZED TEXT}'
        assert em.lower().render_as('latex') == '\\emph{emphasized text}'
        assert em.render_as('html') == '<em>Emphasized text</em>'

        t = Tag(u'em', u'123', Tag(u'em', u'456', Text(u'78'), u'9'), u'0')
        assert t[:2].render_as('html') == '<em>12</em>'
        assert t[2:4].render_as('html') == '<em>3<em>4</em></em>'

        tag = Tag('em', Text(), Text('mary ', 'had ', 'a little lamb'))
        assert tag.render_as('html') == '<em>mary had a little lamb</em>'


class TestHRef(TextTestMixin, TestCase):
    def test__init__(self):
        empty = HRef('/')
        assert empty.url == '/'
        assert empty.parts == []

    def test__str__(self):
        empty = HRef('/')
        assert six.text_type(empty) == ''

        text = Text('This ', HRef('/', 'is'), ' good')
        six.text_type(text) == 'This is good'

    def test__eq__(self):
        assert HRef('/', '') != ''
        assert HRef('/', '') != Text()
        assert HRef('/', '') != HRef('', '')
        assert HRef('/', '') == HRef('/', '')

        assert HRef('/', 'good') != HRef('', 'bad')
        assert HRef('/', 'good') != Text('good')
        assert HRef('/', 'good') == HRef('/', 'good')
        assert not (HRef('/', 'good') != HRef('/', 'good'))

        assert HRef('strong', '') != Tag('strong', '')

    def test__len__(self):
        val = 'Tomato apple!'
        assert len(HRef('index', val)) == len(val)

    def test__contains__(self):
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert not 'mary' in tag
        assert 'Mary' in tag
        assert 'had a little' in tag

        text = Text('This ', HRef('/', 'is'), ' good')
        assert not 'This is' in text

    def test__getitem__(self):
        t = HRef('/', '1234567890')

        with pytest.raises(TypeError):
            1 in t

        assert t == HRef('/', '1234567890')
        assert t[:] == t
        assert t[:0] == HRef('/', '')
        assert t[:1] == HRef('/', '1')
        assert t[:3] == HRef('/', '123')
        assert t[:5] == HRef('/', '12345')
        assert t[:7] == HRef('/', '1234567')
        assert t[:10] == HRef('/', '1234567890')
        assert t[:100] == HRef('/', '1234567890')
        assert t[:-100] == HRef('/', '')
        assert t[:-10] == HRef('/', '')
        assert t[:-9] == HRef('/', '1')
        assert t[:-7] == HRef('/', '123')
        assert t[:-5] == HRef('/', '12345')
        assert t[:-3] == HRef('/', '1234567')
        assert t[-100:] == HRef('/', '1234567890')
        assert t[-10:] == HRef('/', '1234567890')
        assert t[-9:] == HRef('/', '234567890')
        assert t[-7:] == HRef('/', '4567890')
        assert t[-5:] == HRef('/', '67890')
        assert t[-3:] == HRef('/', '890')
        assert t[1:] == HRef('/', '234567890')
        assert t[3:] == HRef('/', '4567890')
        assert t[5:] == HRef('/', '67890')
        assert t[7:] == HRef('/', '890')
        assert t[10:] == HRef('/', '')
        assert t[100:] == HRef('/', '')
        assert t[0:10] == HRef('/', '1234567890')
        assert t[0:100] == HRef('/', '1234567890')
        assert t[2:3] == HRef('/', '3')
        assert t[2:4] == HRef('/', '34')
        assert t[3:7] == HRef('/', '4567')
        assert t[4:7] == HRef('/', '567')
        assert t[4:7] == HRef('/', '567')
        assert t[7:9] == HRef('/', '89')
        assert t[100:200] == HRef('/', '')

        t = HRef('', '123', HRef('/', '456', HRef('/', '789')), '0')
        assert t[:3] == HRef('', '123')
        assert t[:5] == HRef('', '123', HRef('/', '45'))
        assert t[:7] == HRef('', '123', HRef('/', '456', HRef('/', '7')))
        assert t[:10] == HRef('', '123', HRef('/', '456', HRef('/', '789')), '0')
        assert t[:100] == HRef('', '123', HRef('/', '456', HRef('/', '789')), '0')
        assert t[:-7] == HRef('', '123')
        assert t[:-5] == HRef('', '123', HRef('/', '45'))
        assert t[:-3] == HRef('', '123', HRef('/', '456', HRef('/', '7')))

    def test__add__(self):
        assert HRef('/', '') + HRef('/', '') == Text(HRef('/', ''))
        assert HRef('/', '') + HRef('strong', '') == Text(HRef('/', ''), HRef('strong', ''))
        assert HRef('/', 'Good') + HRef('/', '') == Text(HRef('/', 'Good'))
        assert HRef('/', 'Good') + HRef('/', ' job!') == Text(HRef('/', 'Good job!'))
        assert HRef('/', 'Good') + HRef('strong', ' job!') == Text(HRef('/', 'Good'), HRef('strong', ' job!'))
        assert HRef('/', 'Good') + Text(' job!') == Text(HRef('/', 'Good'), ' job!')
        assert Text('Good') + HRef('/', ' job!') == Text('Good', HRef('/', ' job!'))

    def test_append(self):
        text = HRef('/', 'Chuck Norris')
        assert (text +  ' wins!').render_as('html') == '<a href="/">Chuck Norris</a> wins!'
        assert text.append(' wins!').render_as('html') == '<a href="/">Chuck Norris wins!</a>'

    def test_lower(self):
        assert HRef('/').lower() == HRef('/')

        href = HRef('http://www.example.com', 'Hyperlinked text.')
        assert href.lower().render_as('latex') == '\\href{http://www.example.com}{hyperlinked text.}'

        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.lower().render_as('html') == '<a href="info.html">mary had a little lamb</a>'

    def test_upper(self):
        assert HRef('/').upper() == HRef('/')

        href = HRef('http://www.example.com', 'Hyperlinked text.')
        assert href.upper().render_as('latex') == '\\href{http://www.example.com}{HYPERLINKED TEXT.}'

        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.upper().render_as('html') == '<a href="info.html">MARY HAD A LITTLE LAMB</a>'

    def test_capfirst(self):
        assert HRef('/').capfirst() == Text(HRef('/'))

        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a Little Lamb'))
        assert tag.capfirst().render_as('html') == '<a href="info.html">Mary had a Little Lamb</a>'
        assert tag.lower().capfirst().render_as('html') == '<a href="info.html">Mary had a little lamb</a>'

    def test_capitalize(self):
        assert HRef('/').capitalize() == Text(HRef('/'))

        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a Little Lamb'))
        assert tag.capitalize().render_as('html') == '<a href="info.html">Mary had a little lamb</a>'
        assert tag.lower().capitalize().render_as('html') == '<a href="info.html">Mary had a little lamb</a>'

    def test_add_period(self):
        assert HRef('/').add_period() == HRef('/')

        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.add_period().render_as('html') == '<a href="info.html">Mary had a little lamb.</a>'
        assert tag.add_period().add_period().render_as('html') == '<a href="info.html">Mary had a little lamb.</a>'

    def test_split(self):
        empty = HRef('/')
        assert empty.split() == []
        assert empty.split('abc') == [empty]

        href = HRef('/', 'World Wide Web')
        assert href.split() == [HRef('/', 'World'), HRef('/', 'Wide'), HRef('/', 'Web')]
        result = Text('Estimated size of the ', href).split()
        assert result == [
            Text('Estimated'),
            Text('size'),
            Text('of'),
            Text('the'),
            Text(HRef('/', 'World')),
            Text(HRef('/', 'Wide')),
            Text(HRef('/', 'Web')),
        ]

        text = Text(Tag('em', Text(Tag('strong', HRef('/', '  Very, very'), ' bad'), ' guys')), '! ')
        assert text.render_as('html') == '<em><strong><a href="/">  Very, very</a> bad</strong> guys</em>! '
        assert text.split(', ') == [
            Text(Tag('em', Tag('strong', HRef('/', '  Very')))),
            Text(Tag('em', Tag('strong', HRef('/', 'very'), ' bad'), ' guys'), '! '),
        ]
        assert text.split(' ') == [
            Text(),
            Text(),
            Text(Tag('em', Tag('strong', HRef('/', 'Very,')))),
            Text(Tag('em', Tag('strong', HRef('/', 'very')))),
            Text(Tag('em', Tag('strong', 'bad'))),
            Text(Tag('em', 'guys'), '!'),
            Text(),
        ]
        assert text.split(' ', keep_empty_parts=False) == [
            Text(Tag('em', Tag('strong', HRef('/', 'Very,')))),
            Text(Tag('em', Tag('strong', HRef('/', 'very')))),
            Text(Tag('em', Tag('strong', 'bad'))),
            Text(Tag('em', 'guys'), '!'),
        ]
        assert text.split() == [
            Text(Tag('em', Tag('strong', HRef('/', 'Very,')))),
            Text(Tag('em', Tag('strong', HRef('/', 'very')))),
            Text(Tag('em', Tag('strong', 'bad'))),
            Text(Tag('em', 'guys'), '!'),
        ]
        assert text.split(keep_empty_parts=True) == [
            Text(),
            Text(Tag('em', Tag('strong', HRef('/', 'Very,')))),
            Text(Tag('em', Tag('strong', HRef('/', 'very')))),
            Text(Tag('em', Tag('strong', 'bad'))),
            Text(Tag('em', 'guys'), '!'),
            Text(),
        ]

        text = Text(' A', Tag('em', ' big', HRef('/', ' ', Tag('strong', 'no-no'), '!  ')))
        assert text.render_as('html') == ' A<em> big<a href="/"> <strong>no-no</strong>!  </a></em>'
        assert text.split('-') == [
            Text(' A', Tag('em', ' big', HRef('/', ' ', Tag('strong', 'no')))),
            Text(Tag('em', HRef('/', Tag('strong', 'no'), '!  '))),
        ]
        assert text.split(' ') == [
            Text(),
            Text('A'),
            Text(Tag('em', 'big')),
            Text(Tag('em', HRef('/', Tag('strong', 'no-no'), '!'))),
            Text(),
            Text(),
        ]
        assert text.split(' ', keep_empty_parts=False) == [
            Text('A'),
            Text(Tag('em', 'big')),
            Text(Tag('em', HRef('/', Tag('strong', 'no-no'), '!'))),
        ]
        assert text.split() == [
            Text('A'),
            Text(Tag('em', 'big')),
            Text(Tag('em', HRef('/', Tag('strong', 'no-no'), '!'))),
        ]
        assert text.split(keep_empty_parts=True) == [
            Text(),
            Text('A'),
            Text(Tag('em', 'big')),
            Text(Tag('em', HRef('/', Tag('strong', 'no-no'), '!'))),
            Text(),
        ]

    def test_join(self):
        href = HRef('/', 'World Wide Web')
        result = Text('-').join(Text('Estimated size of the ', href).split())
        assert result == Text('Estimated-size-of-the-', HRef('/', 'World'), '-', HRef('/', 'Wide'), '-', HRef('/', 'Web'))

    def test_startswith(self):
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.startswith('')
        assert tag.startswith('M')
        assert tag.startswith('Mary')
        assert not tag.startswith('m')
        assert not tag.startswith('mary')

        tag = HRef('/', 'a', 'b', 'c')
        assert tag.startswith('ab')

        tag = HRef('/', 'This is good')
        assert tag.startswith(('This', 'That'))
        assert not tag.startswith(('That', 'Those'))

        text = Text('This ', HRef('/', 'is'), ' good')
        assert not text.startswith('This is')

    def test_endswith(self):
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.endswith('')
        assert not tag.endswith('B')
        assert tag.endswith('b')
        assert tag.endswith('lamb')

        tag = HRef('/', 'a', 'b', 'c')
        assert tag.endswith('bc')

        tag = HRef('/', 'This is good')
        assert tag.endswith(('good', 'wonderful'))
        assert not tag.endswith(('bad', 'awful'))

        text = Text('This ', HRef('/', 'is'), ' good')
        assert not text.endswith('is good')

    def test_isalpha(self):
        assert not HRef('/').isalpha()
        assert not HRef('/', 'a b c').isalpha()
        assert HRef('/', 'abc').isalpha()
        assert HRef('/', u'文字').isalpha()

    def test_render_as(self):
        href = HRef('http://www.example.com', 'Hyperlinked text.')
        assert href.render_as('latex') == '\\href{http://www.example.com}{Hyperlinked text.}'
        assert href.render_as('html') == '<a href="http://www.example.com">Hyperlinked text.</a>'
        assert href.render_as('plaintext') == 'Hyperlinked text.'

        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.render_as('html') == '<a href="info.html">Mary had a little lamb</a>'


class TestProtected(TextTestMixin, TestCase):
    def test__init__(self):
        assert six.text_type(Protected('a', '', 'c')) == 'ac'
        assert six.text_type(Protected('a', Text(), 'c')) == 'ac'

        text = Protected(Protected(), Protected('mary ', 'had ', 'a little lamb'))
        assert text == Protected(Protected('mary had a little lamb'))
        assert six.text_type(text) == 'mary had a little lamb'

        text = six.text_type(Protected('a', Protected('b', 'c'), Tag('em', 'x'), Symbol('nbsp'), 'd'))
        assert text == 'abcx<nbsp>d'

        with pytest.raises(ValueError):
            Protected({})

        with pytest.raises(ValueError):
            Protected(0, 0)

    def test__eq__(self):
        assert Protected() == Protected()
        assert not (Protected() != Protected())

        assert Protected('Cat') == Protected('Cat')
        assert not (Protected('Cat') != Protected('Cat'))
        assert Protected('Cat', ' tail') == Protected('Cat tail')
        assert not (Protected('Cat', ' tail') != Protected('Cat tail'))

        assert Protected('Cat') != Protected('Dog')
        assert not (Protected('Cat') == Protected('Dog'))

    def test__len__(self):
        assert len(Protected()) == 0
        assert len(Protected('Never', ' ', 'Knows', ' ', 'Best')) == len('Never Knows Best')
        assert len(Protected('Never', ' ', Tag('em', 'Knows', ' '), 'Best')) == len('Never Knows Best')
        assert len(Protected('Never', ' ', Tag('em', HRef('/', 'Knows'), ' '), 'Best')) == len('Never Knows Best')

    def test__str__(self):
        assert six.text_type(Protected()) == ''
        assert six.text_type(Protected(u'Чудаки украшают мир')) == u'Чудаки украшают мир'

    def test__contains__(self):
        text = Protected('mary ', 'had ', 'a little lamb')
        assert 'mary' in text
        assert 'Mary' not in text
        assert 'had a little' in text

        text = Protected('a', 'b', 'c')
        assert 'abc' in text

    def test_capfirst(self):
        text = Protected('mary ', 'had ', 'a Little Lamb')
        assert six.text_type(text.capitalize()) == 'mary had a Little Lamb'

    def test_capitalize(self):
        text = Protected('mary ', 'had ', 'a little lamb')
        assert six.text_type(text.capitalize()) == 'mary had a little lamb'

    def test__add__(self):
        t = Protected('a')
        assert t + 'b' == Text(Protected('a'), 'b')
        assert t + t == Text(Protected('aa'))

    def test__getitem__(self):
        t = Protected('1234567890')

        with pytest.raises(TypeError):
            1 in t

        assert t == Protected('1234567890')
        assert t[:0] == Protected('')
        assert t[:1] == Protected('1')
        assert t[:3] == Protected('123')
        assert t[:5] == Protected('12345')
        assert t[:7] == Protected('1234567')
        assert t[:10] == Protected('1234567890')
        assert t[:100] == Protected('1234567890')
        assert t[:-100] == Protected('')
        assert t[:-10] == Protected('')
        assert t[:-9] == Protected('1')
        assert t[:-7] == Protected('123')
        assert t[:-5] == Protected('12345')
        assert t[:-3] == Protected('1234567')
        assert t[-100:] == Protected('1234567890')
        assert t[-10:] == Protected('1234567890')
        assert t[-9:] == Protected('234567890')
        assert t[-7:] == Protected('4567890')
        assert t[-5:] == Protected('67890')
        assert t[-3:] == Protected('890')
        assert t[1:] == Protected('234567890')
        assert t[3:] == Protected('4567890')
        assert t[5:] == Protected('67890')
        assert t[7:] == Protected('890')
        assert t[10:] == Protected('')
        assert t[100:] == Protected('')
        assert t[0:10] == Protected('1234567890')
        assert t[0:100] == Protected('1234567890')
        assert t[2:3] == Protected('3')
        assert t[2:4] == Protected('34')
        assert t[3:7] == Protected('4567')
        assert t[4:7] == Protected('567')
        assert t[4:7] == Protected('567')
        assert t[7:9] == Protected('89')
        assert t[100:200] == Protected('')

        t = Protected('123', Protected('456', Protected('789')), '0')
        assert t[:3] == Protected('123')
        assert t[:5] == Protected('123', Protected('45'))
        assert t[:7] == Protected('123', Protected('456', Protected('7')))
        assert t[:10] == Protected('123', Protected('456', Protected('789')), '0')
        assert t[:100] == Protected('123', Protected('456', Protected('789')), '0')
        assert t[:-7] == Protected('123')
        assert t[:-5] == Protected('123', Protected('45'))
        assert t[:-3] == Protected('123', Protected('456', Protected('7')))

    def test_append(self):
        text = Protected('Chuck Norris')
        assert (text + ' wins!').render_as('latex') == '{Chuck Norris} wins!'
        assert text.append(' wins!').render_as('latex') == '{Chuck Norris wins!}'

    def test_upper(self):
        text = Protected('Mary ', 'had ', 'a little lamb')
        assert six.text_type(text.upper()) == 'Mary had a little lamb'
        text = Protected('mary ', 'had ', 'a little lamb')
        assert six.text_type(text.upper()) == 'mary had a little lamb'

    def test_lower(self):
        text = Protected('Mary ', 'had ', 'a little lamb')
        assert six.text_type(text.lower()) == 'Mary had a little lamb'
        text = Protected('MARY ', 'HAD ', 'A LITTLE LAMB')
        assert six.text_type(text.lower()) == 'MARY HAD A LITTLE LAMB'

    def test_startswith(self):
        assert not Protected().startswith('.')
        assert not Protected().startswith(('.', '!'))

        text = Protected('mary ', 'had ', 'a little lamb')
        assert not text.startswith('M')
        assert text.startswith('m')

        text = Protected('a', 'b', 'c')
        assert text.startswith('ab')

        assert Protected('This is good').startswith(('This', 'That'))
        assert not Protected('This is good').startswith(('That', 'Those'))

    def test_endswith(self):
        assert not Protected().endswith('.')
        assert not Protected().endswith(('.', '!'))

        text = Protected('mary ', 'had ', 'a little lamb')
        assert not text.endswith('B')
        assert text.endswith('b')

        text = Protected('a', 'b', 'c')
        assert text.endswith('bc')

        assert Protected('This is good').endswith(('good', 'wonderful'))
        assert not Protected('This is good').endswith(('bad', 'awful'))

    def test_isalpha(self):
        assert not Protected().isalpha()
        assert not Protected('a b c').isalpha()
        assert Protected('abc').isalpha()
        assert Protected(u'文字').isalpha()

    def test_join(self):
        assert Protected(' ').join(['a', Protected('b c')]).render_as('latex') == 'a{ b c}'
        assert Protected(nbsp).join(['a', 'b', 'c']).render_as('latex') == 'a{~}b{~}c'
        assert nbsp.join(['a', Protected('b'), 'c']).render_as('latex') == 'a~{b}~c'
        assert String('-').join([Protected('a'), Protected('b'), Protected('c')]).render_as('latex') == '{a}-{b}-{c}'
        result = Protected(' and ').join(['a', 'b', 'c']).render_as('latex')
        assert result == 'a{ and }b{ and }c'

    def test_split(self):
        assert Protected().split() == [Protected()]
        assert Protected().split('abc') == [Protected()]
        assert Protected('a').split() == [Protected('a')]
        assert Protected('a ').split() == [Protected('a ')]
        assert Protected('   a   ').split() == [Protected('   a   ')]
        assert Protected('a + b').split() == [Protected('a + b')]
        assert Protected('a + b').split(' + ') == [Protected('a + b')]
        assert Protected('abc').split('xyz') == [Protected('abc')]
        assert Protected('---').split('--') == [Protected('---')]
        assert Protected('---').split('-') == [Protected('---')]

    def test_add_period(self):
        assert not Protected().endswith(('.', '!', '?'))
        assert not textutils.is_terminated(Protected())

        assert Protected().add_period().render_as('latex') == '{}'

        text = Protected("That's all, folks")
        assert text.add_period().render_as('latex') == "{That's all, folks.}"

    def test_render_as(self):
        string = Protected('a < b')
        assert string.render_as('latex') == '{a < b}'
        assert string.render_as('html') == '<span class="bibtex-protected">a &lt; b</span>'


class TestSymbol(TextTestMixin, TestCase):
    def test__init__(self):
        assert nbsp.name == 'nbsp'

    def test__eq__(self):
        assert Symbol('nbsp') == Symbol('nbsp')
        assert not Symbol('nbsp') != Symbol('nbsp')

        assert not Symbol('nbsp') == Symbol('ndash')
        assert Symbol('nbsp') != Symbol('ndash')

        assert Text(nbsp, nbsp) == Text(Symbol('nbsp'), Symbol('nbsp'))

    def test__str__(self):
        assert six.text_type(nbsp) == '<nbsp>'

    def test__len__(self):
        assert len(nbsp) == 1

    def test__contains__(self):
        assert not '' in nbsp
        assert not 'abc' in nbsp

    def test__getitem__(self):
        symbol = Symbol('nbsp')
        assert symbol[0] == Symbol('nbsp')
        assert symbol[0:] == Symbol('nbsp')
        assert symbol[0:5] == Symbol('nbsp')
        assert symbol[1:] == String()
        assert symbol[1:5] == String()

        with pytest.raises(IndexError):
            symbol[1]

    def test__add__(self):
        assert (nbsp + '.').render_as('html') == '&nbsp;.'

    def test_split(self):
        assert nbsp.split() == [nbsp]
        text = Text('F.', nbsp, 'Miller')
        assert text.split() == [text]

    def test_join(self):
        assert nbsp.join(['S.', 'Jerusalem']) == Text('S.', nbsp, 'Jerusalem')

    def test_upper(self):
        assert nbsp.upper().render_as('html') == '&nbsp;'

    def test_lower(self):
        assert nbsp.lower().render_as('html') == '&nbsp;'

    def test_capfirst(self):
        assert Text(nbsp, nbsp).capfirst().render_as('html') == '&nbsp;&nbsp;'

    def test_capitalize(self):
        assert Text(nbsp, nbsp).capitalize().render_as('html') == '&nbsp;&nbsp;'

    def test_add_period(self):
        assert nbsp.add_period().render_as('html') == '&nbsp;.'
        assert nbsp.add_period().add_period().render_as('html') == '&nbsp;.'

    def test_append(self):
        assert nbsp.append('.').render_as('html') == '&nbsp;.'

    def test_startswith(self):
        assert not nbsp.startswith('.')
        assert not nbsp.startswith(('.', '?!'))

    def test_endswith(self):
        assert not nbsp.endswith('.')
        assert not nbsp.endswith(('.', '?!'))

    def test_isalpha(self):
        assert not nbsp.isalpha()

    def test_render_as(self):
        assert nbsp.render_as('latex') == '~'
        assert nbsp.render_as('html') == '&nbsp;'
        assert Text(nbsp, nbsp).render_as('html') == '&nbsp;&nbsp;'
