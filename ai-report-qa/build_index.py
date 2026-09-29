import array, collections, json, math, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'知识库提取_2026半年报'/'knowledge_base_chunks.jsonl'
OUT=ROOT/'dist'/'data';OUT.mkdir(parents=True,exist_ok=True)
DIM=256

def tokens(text):
    text=text.lower();out=[]
    for run in re.findall(r'[\u4e00-\u9fff]+',text):
        out += list(run)
        for n in (2,3,4):out += [run[i:i+n] for i in range(max(0,len(run)-n+1))]
    out += re.findall(r'[a-z0-9][a-z0-9._%+-]*',text)
    return out

def fnv(s):
    h=2166136261
    for b in s.encode('utf-8'):h=((h^b)*16777619)&0xffffffff
    return h

rows=[json.loads(x) for x in SOURCE.open(encoding='utf-8') if x.strip()]
counts=[];df=collections.Counter();lengths=[]
for r in rows:
    searchable=' '.join(str(r.get(k) or '') for k in ('company','stock_code','chapter','section','text'))
    c=collections.Counter(tokens(searchable));counts.append(c);lengths.append(sum(c.values()));df.update(c)
N=len(rows);keep={t for t,n in df.items() if n<=N*.35}
postings={};vectors=array.array('b')
for doc,c in enumerate(counts):
    vec=[0.0]*DIM
    for t,tf in c.items():
        if t not in keep:continue
        postings.setdefault(t,[]).extend((doc,tf))
        w=(1+math.log(tf))*(math.log((N+1)/(df[t]+1))+1);h=fnv(t)
        vec[h%DIM] += w if ((h>>30)&1)==0 else -w
    norm=math.sqrt(sum(x*x for x in vec)) or 1
    vectors.extend(max(-127,min(127,round(x/norm*127))) for x in vec)

corpus=[]
for r in rows:
    corpus.append({k:r.get(k) for k in ('chunk_id','company','stock_code','content_type','chapter','section','page_start','page_end','text')})
meta={'count':N,'dim':DIM,'avgdl':sum(lengths)/N,'lengths':lengths,'postings':postings,'companies':sorted({r['company'] for r in rows})}
(OUT/'corpus.json').write_text(json.dumps(corpus,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
(OUT/'index.json').write_text(json.dumps(meta,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
with (OUT/'vectors.bin').open('wb') as f:vectors.tofile(f)
print(json.dumps({'documents':N,'terms':len(postings),'vector_bytes':len(vectors),'corpus_mb':round((OUT/'corpus.json').stat().st_size/1048576,2)},ensure_ascii=False))
