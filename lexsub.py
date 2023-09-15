from scipy.spatial.distance import cosine
import numpy as np
import xml.etree.ElementTree as et


class NearestWords():
    def fit(self, embs, target, k):
        sim_vals = {w: cosine(x, target) for w, x in embs.items()}
        sim_vals = sorted(sim_vals.items(), key=lambda item: item[1])
        return sim_vals[1:k]


class LexSub():
    def fit(self, sentence, nwrds, target, embs, subs=5, op='cos'):
        sent_embt = np.stack([embs[e] for e in sentence if e in embs.keys(
        )] + [embs[target]], axis=-1).mean(axis=1)
        sent_embc = [np.stack([embs[e] for e in sentence if e in embs.keys(
        )] + [embs[wrd]], axis=-1).mean(axis=1) for wrd in nwrds]
        res = []
        for embs in sent_embc:
            if op == "dot":
                res.append(embs @ sent_embt)
            if op == "cos":
                res.append(cosine(embs, sent_embt))
        sub_wrds = []
        if op == 'cos':
            for i in range(subs):
                idx = res.index(min(res))
                sub_wrds.append(nwrds[idx])
                del res[idx]
                del nwrds[idx]
        if op == 'dot':
            for i in range(subs):
                idx = res.index(max(res))
                sub_wrds.append(nwrds[idx])
                del res[idx]
                del nwrds[idx]
        return sub_wrds


def getEmb(filename):
    with open(filename) as fp:
        txt = fp.read().split("\n")
    vocab = txt[0].split(' ')[0]
    dim = txt[0].split(' ')[1]
    del txt[0]
    wrdEmbs = {}
    for l in txt:
        l = l.split(" ")
        word = l[0]
        del l[0]
        emb = np.array(l, dtype='float32')
        if emb.shape != (0, ):
            wrdEmbs[word] = emb
    return wrdEmbs


def getWordEmbeddings(wordfile, embfile):
    with open(wordfile) as mp:
        words = mp.read()
        words = words.split('\n')
    embs = np.loadtxt(embfile, delimiter='\t')
    if words[-1] == "":
        del words[-1]
    wrdEmb = {}
    for i in range(len(words)):
        wrdEmb[words[i]] = embs[i]
    return wrdEmb


# def parseCoinco(xmlfile):
#     tree = et.parse(xmlfile)
#     root = tree.getroot()
#     for item in root.findall('./sent'):
#         for child in item:
#             # print(child.tag)
#             if child.tag == 'tokens':
#                 for gchild in child:
#                     print(gchild.tag)
    # print(root.findall('./sent/targetsentence'))


if __name__ == '__main__':
    nw = NearestWords()
    wrdEmb = getWordEmbeddings(
        'gensim10k/metadata.tsv', 'gensim10k/vectors.tsv')
    word = 'mission'
    k = 100
    subs = 10
    near_wrds = nw.fit(wrdEmb, wrdEmb[word], k)
    ls = LexSub()
    sub_wrd = ls.fit('A mission to end a war AUSTIN, Texas -- Tom Karnes was dialing for destiny but not everyone wanted to cooperate'.split(
        ' '), [x[0] for x in near_wrds], word, wrdEmb, subs=subs, op='dot')
    print(f"The nearest {k} words for {word} are: {near_wrds}")
    print(f"The top {subs} substitution words for {word} are: {sub_wrd}")
