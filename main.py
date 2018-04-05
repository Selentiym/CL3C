import pymorphy2
from xtok import tokenize
import re

def isplit(source, sep=None, regex=False):
    """
    generator version of str.split()

    :param source:
        source string (unicode or bytes)

    :param sep:
        separator to split on.

    :param regex:
        if True, will treat sep as regular expression.

    :returns:
        generator yielding elements of string.
    """
    if sep is None:
        # mimic default python behavior
        source = source.strip()
        sep = "\\s+"
        if isinstance(source, bytes):
            sep = sep.encode("utf-8")
        regex = True
    if regex:
        # version using re.finditer()
        if not hasattr(sep, "finditer"):
            sep = re.compile(sep)
        start = 0
        for m in sep.finditer(source):
            idx = m.start()
            assert idx >= start
            yield source[start:idx]
            start = m.end()
        yield source[start:]
    else:
        # version using str.find(), less overhead than re.finditer()
        sepsize = len(sep)
        start = 0
        while True:
            idx = source.find(sep, start)
            if idx == -1:
                yield source[start:]
                return
            yield source[start:idx]
            start = idx + sepsize


morph = pymorphy2.MorphAnalyzer()

print(morph.parse('эта'))

exit()

links = [
    ({'_POS':'ADJF', 'case': True, 'gender': True, 'number': True}, {'_POS':'NOUN'}),
    ({'_POS':'GRND'}, {'_POS':'NOUN'}),
    ({'_POS':'NOUN', 'case':'nomn','number':True,'gender':True}, {'_POS':'VERB'}),
    ({'_POS':'NOUN', 'case':'nomn'}, {'_POS':'NOUN', 'case':'gent'}),
]

def testForm(form, template, secondTemplate = None):
    """
    Проверяет конкретную форму слова на соответствие конкретному набору тегов.
    form - один из вариантов py
    """
    if secondTemplate is None:
        secondTemplate = dict()
    ok = True
    for cat, val in template.items():
        if type(val) is bool:
            secondTemplate[cat] = getattr(form, cat)
        else:
            if (val != getattr(form, cat)):
                ok = False
                break
    return ok, secondTemplate

def testOnRules(first, second):
    # Пробегаем по всем интерпретациям первого слова (в нашем случае в основном падежи)
    for form in first:
        # Берем Opencorpora Tag из объекта Parse
        form = form[1]
        # Проверяем каждое правило
        for rule in links:
            # Сначала проверяем левую часть правила с первой формой слова
            # Если хотим согласование, то может добавиться новое условие
            # На вторую форму в словаре toTest
            rez, toTest = testForm(form, rule[0], rule[1].copy())
            # Если первая форма не подходит к левой части данного правила, то пропускаем правило.
            if not rez:
                continue
            # Если же форма подходит, то проверяем все формы второго слова на соответствие обновленному правилу
            for form2 in second:
                # Берем Opencorpora Tag из объекта Parse
                form2 = form2[1]
                rez, _ = testForm(form2, toTest)
                # Если какое-то сочетание подошло, то нашли правило и пару форм
                # Если же не подошло, то продолжаем поиск...
                if rez:
                    return True
    # Если все возможные варианты форм первого слова, правил и форм второго слова пробежали, и нигде
    # не получили совпадение, то возвращаем негативный ответ
    return False


pairCounts = dict()
def registerPair(word1, word2):
    # достаем из словаря все уже встреченные словосочетания, где первая словоформа
    # соответсвует word1
    innerDict = pairCounts.get(word1, dict())
    # достаем число сколько раз встретилось словосочетание word1 word2 и инкрементируем его
    number = innerDict.get(word2, 0) + 1
    # сохраняем все обратно
    innerDict[word2] = number
    pairCounts[word1] = innerDict


wordCounts = {'summa': 0}
def registerWord(word):
    # Увеличиваем на 1 общий счет словоформ
    wordCounts['summa'] += 1
    # наращиваем на 1 счетчик для данной словоформы
    number = wordCounts.get(word, 0) + 1
    wordCounts[word] = number


# Немного тестирования.
# белого гриб - подходит, тк среди анализа слова белого есть вариант с тегами sing, accs,
# и при анализе слова гриб тоже есть sing, accs
# print(testOnRules(morph.parse('белого'),morph.parse('гриб')))

findWord = morph.parse('жизнь')[0]

# print(type(findWord))

# Считываем огромный файл
# with open('smallest.txt','r') as file:
# with open('fragment.txt','r') as file:
with open('WarAndPeace.txt','r') as file:
    fulltext = file.read()
    iter = isplit(fulltext)
    i = 0
    # Предыдущий токен сначала не определен, значит и его разбора нет.
    prev = None
    # Здесь будет лежать само слово, не модифицированное, без разбора, просто строка
    prevText = None
    # По очереди проходим по кусочкам между пробелами
    for preToken in iter:
        preToken = preToken.strip()
        # пустые элементы пропускаем
        if len(preToken) > 1:
            # делаем из того, что между пробелами, токены. Может получиться более, чем один токен.
            # Например, 'платье.' превратится в 2 токена: 'платье', '.'
            for el in tokenize(preToken):
                if el['type'] == b'WORD':
                    # Сохраняем слово
                    registerWord(el['data'])
                    cur = morph.parse(el['data'])
                    # cur = []
                    # Из всех возможных разборов текущего слова ищем те, которые имеют искомую основу.
                    # for form in tmp:
                    #     if form[2] == findWord[2]:
                    #         cur.append(form)
                    if prev is not None:
                        formPairs = []
                        for f1 in cur:
                            firstOk = False
                            if f1[2] == findWord[2]:
                                firstOk = True
                            for f2 in prev:
                                if firstOk or f2[2] == findWord[2]:
                                    formPairs.append((f1, f2))
                        if len(formPairs) > 0:
                            if (testOnRules(prev, cur)):
                                registerPair(prevText, el['data'])
                else:
                    cur = None
                prev = cur
                prevText = el['data']
                    # cur = morph.parse(el['data'])[0]
                    # if cur[2] == findWord[2]:
                    #     i += 1
                    #     print(prev.word, cur.word)
                    #     break
                    # if prev is not None and prev[2] == findWord[2]:
                    #     print(prev.word, cur.word)
                    # prev = cur
            # if i > 50:
            #     break

print(wordCounts['summa'])
print(len(pairCounts))

import numpy as np

def rounds(num, max_=2):
    '''с точностью не более n "значащих цифр", после запятой. '''
    left, right = str(num).split('.')
    zero, nums = zero_nums = [], []
    for n in right:
        zero_nums[0 if not nums and n == '0' else 1].append(n)
        if len(nums) == max_:
            break
    return '.'.join([left, ''.join(zero) + ''.join(nums)])

def MImetric(w1, w2):
    return np.log2(pairCounts[w1][w2] * wordCounts['summa'] / (wordCounts[w1] * wordCounts[w2]))

def MI3metric(w1, w2):
    return np.log2((pairCounts[w1][w2])**3 * wordCounts['summa'] / (wordCounts[w1] * wordCounts[w2]))

def tScore(w1, w2):
    pair = pairCounts[w1][w2]
    return (pair - wordCounts[w1] * wordCounts[w2] / wordCounts['summa']) / pairCounts[w1][w2] ** 0.5

def loglogLikelihood(w1,w2):
    return pairCounts[w1][w2] * MImetric(w1,w2)

def dice(w1, w2):
    return 2 * pairCounts[w1][w2] / (wordCounts[w1] + wordCounts[w2])

metrics = {
    'Mutual information': MImetric,
    'Mutual information (cube)': MI3metric,
    't-score': tScore,
    'log-log likelihood': loglogLikelihood,
    'dice': dice
}

allPairs = []
for w1 in pairCounts:
    for w2 in pairCounts[w1]:
        allPairs.append((w1,w2))
n = 20
for name, func in metrics.items():
    print(name + ':')
    for pair in sorted(allPairs,key=lambda x: func(*x), reverse=True)[0:n]:
        print(pair[0] + ' ' + pair[1]+': '+rounds(func(pair[0], pair[1]), 2) + ',',pairCounts[pair[0]][pair[1]],',', wordCounts[pair[0]],',',wordCounts[pair[1]])
# for token in allTokens:
#     pass