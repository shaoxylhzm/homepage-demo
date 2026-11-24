import json, math, re, sys, hashlib, csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
OUT=ROOT.parent/'检索审计'; OUT.mkdir(exist_ok=True)
DATA=ROOT/'dist/data'
rows=json.loads((DATA/'corpus.json').read_text(encoding='utf8'))
full=[json.loads(l) for l in (BASE/'知识库提取_2026半年报/knowledge_base_chunks.jsonl').read_text(encoding='utf8').splitlines()]
idx=json.loads((DATA/'index.json').read_text(encoding='utf8'))
vec=np.fromfile(DATA/'vectors.bin',dtype=np.int8).reshape(-1,idx['dim']).astype(float)/127
urls=dict(zip(['科大讯飞','寒武纪','海光信息','浪潮信息','中科曙光','金山办公','三六零','昆仑万维','拓尔思','云天励飞','中际旭创','润泽科技'],['2026-08-21/1225485550','2026-08-08/1225464969','2026-08-14/1225472510','2026-08-29/1225530925','2026-08-26/1225507613','2026-08-20/1225482678','2026-08-20/1225483359','2026-08-20/1225483546','2026-08-29/1225523925','2026-08-27/1225515128','2026-08-22/1225491753','2026-08-28/1225521959']))
urls={k:'https://static.cninfo.com.cn/finalpage/'+v+'.PDF' for k,v in urls.items()}
def toks(s):
 s=s.lower(); out=[]
 for run in re.findall(r'[\u4e00-\u9fff]+',s):
  out+=list(run)
  for n in (2,3,4):out += [run[i:i+n] for i in range(len(run)-n+1)]
 return out+re.findall(r'[a-z0-9][a-z0-9._%+-]*',s)
def search(q):
 from collections import Counter
 bm=np.zeros(len(rows)); qv=np.zeros(idx['dim'])
 for t,qn in Counter(toks(q)).items():
  p=idx['postings'].get(t)
  if not p:continue
  df=len(p)//2; ids=np.array(p[::2]); tf=np.array(p[1::2]); dl=np.array(idx['lengths'])[ids]
  bm[ids]+=math.log(1+(len(rows)-df+.5)/(df+.5))*tf*2.35/(tf+1.35*(.28+.72*dl/idx['avgdl']))
  h=2166136261
  for b in t.encode():h=((h^b)*16777619)&0xffffffff
  qv[h%idx['dim']]+=(1+math.log(qn))*(math.log((len(rows)+1)/(df+1))+1)*(1 if ((h>>30)&1)==0 else -1)
 qv/=np.linalg.norm(qv) or 1; cos=vec@qv; score=.53*np.maximum(0,cos)+.47*bm/(max(bm) or 1)
 def top(a):return [dict(rank=j+1,chunk_id=rows[i]['chunk_id'],company=rows[i]['company'],page=rows[i]['page_start'],bm25=float(bm[i]),cosine=float(cos[i]),fusion=float(score[i])) for j,i in enumerate(np.argsort(-a,kind='stable')[:12])]
 return dict(bm25=top(bm),vector=top(cos),fusion=top(score))
def clean(s):return re.sub(r'\s+','',str(s))
def evidence(company,metric):
 candidates=[]
 for d in full:
  if d['company']!=company or d['content_type']!='table':continue
  for ri,r in enumerate(d['table_data']):
   label=''.join(str(v) for v in r if not re.search(r'\d',str(v)))
   for rr in d['table_data'][max(0,ri-1):ri]+d['table_data'][ri+1:ri+2]:
    if not any(re.fullmatch(r'-?[\d,]+(?:\.\d+)?',clean(v)) for v in rr):label+=''.join(rr)
   nums=[(ci,v) for ci,v in enumerate(r) if re.fullmatch(r'-?[\d,]+(?:\.\d+)?',clean(v))]
   own=clean(''.join(str(v) for v in r[:nums[0][0]])) if nums else ''
   if metric in clean(label) and nums and '占营业收入' not in own and (metric!='营业收入' or own.startswith('营业收入')):
    candidates.append((d['page_start'],d,ri,r,nums[0]))
 if not candidates:raise ValueError((company,metric))
 _,d,ri,r,(ci,value)=min(candidates,key=lambda a:a[0])
 unit='万元' if company=='浪潮信息' else ('千元' if company=='三六零' else '元')
 return dict(company=company,metric=metric,chunk_id=d['chunk_id'],page=d['page_start'],table=d['table_id'],row=d.get('row_start',1)+ri,column=ci+1,row_label=' '.join(r[:ci]),column_label='本报告期（1—6月）',value=str(float(clean(value).replace(',',''))*({'万元':10000,'千元':1000,'元':1}[unit])),raw_value=value,raw_unit=unit,row_values=r,header=d['table_data'][:2],url=urls[company],section=d['section'])
metrics=['营业收入','归属于上市公司股东的净利润','经营活动产生的现金流量净额']
ev={c:{m:evidence(c,m) for m in metrics} for c in urls}
manifest=json.loads((BASE/'知识库提取_2026半年报/extraction_manifest.json').read_text(encoding='utf8'))
sources=[]
for c in manifest['companies']:
 p=BASE/'中国AI公司_2026半年报_12家'/c['source_file']; sources.append(dict(**c,url=urls[c['company']],sha256=hashlib.sha256(p.read_bytes()).hexdigest(),evidence=list(ev[c['company']].values())))
(OUT/'官方来源与单元格.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf8')
tests=[]
for i,c in enumerate(list(urls)[:8]):tests.append((f'{c}2026年上半年营业收入是多少？',[ev[c][metrics[0]]],False))
tests += [('全景：12家公司上半年营业收入分别是多少，哪家最高？',[ev[c][metrics[0]] for c in urls],True),('全景：12家公司归母净利润与经营现金流是否同向？哪些盈利但现金流为负？',[ev[c][m] for c in urls for m in metrics[1:]],True)]
results=[]
def num(s):return float(re.sub(r'[^0-9.\-]','',s))
for i,(q,gold,pan) in enumerate(tests,1):
 run=search(q); ids={x['chunk_id'] for x in gold}; coverage={k:len(ids & {x['chunk_id'] for x in v})/len(ids) for k,v in run.items()}
 answer='；'.join(f"{e['company']} {e['metric']}：{e['value']}元 [PDF {e['page']}页/{e['table']}/R{e['row']}C{e['column']}]" for e in gold)
 if i==9:answer+='。营业收入最高：'+max(urls,key=lambda c:num(ev[c][metrics[0]]['value']))
 if i==10:answer+='。盈利但经营现金流为负：'+'、'.join(c for c in urls if num(ev[c][metrics[1]]['value'])>0 and num(ev[c][metrics[2]]['value'])<0)
 results.append(dict(id=i,question=q,panoramic=pan,answer=answer,evidence=gold,retrieval=run,evaluation=dict(evidence_recall_at_12=coverage,answer_support='结构化证据全量补取后逐单元格生成，非原页面自动回答',verdict='检索证据齐全' if coverage['fusion']==1 else '融合前12不足，需补取',company_coverage=len({x['company'] for x in run['fusion']}))))
(OUT/'逐题运行结果.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# 财报来源、检索实跑与十题证据审计','', '运行日期：2026-11-24。范围：2026年半年度完整报告，非年度报告、非摘要。', '', '## 方法与边界','全库无公司过滤：中文1至4字词片段；BM25 k1=1.35、b=0.72；原索引过滤文档频率超过35%的词。256维FNV有符号TF-IDF哈希向量，int8量化；点积近似余弦。融合=0.47×BM25/max(BM25)+0.53×max(0,向量点积)。未采用语义嵌入。','本脚本复现页面核心融合，明确不含页面对初始前12条的公司/财务启发式重排；原页面结果不能等同于本报告结果。','评价为确定性证据集合Recall@12与公司覆盖数，非人工盲评、非完整相关性标注。答案由全量表格补取生成，不冒充仅靠前12条生成。表格R/C从1起，含表头；页码为PDF物理页。原始表头及整行保存在JSON，跨行合并及单位仍须按原PDF复核。','','## 12家公司官方完整报告与定位','|公司|官方完整PDF|页数|营业收入（元）|PDF页/表/行列|','|---|---|---:|---:|---|']
for s in sources:
 e=ev[s['company']][metrics[0]];lines.append(f"|{s['company']}|[完整报告]({s['url']})|{s['pages']}|{e['value']}|{e['page']}/{e['table']}/R{e['row']}C{e['column']}|")
lines+=['','本地PDF的SHA256、所有三项指标的行标签、表头、整行值与切块ID见官方来源与单元格.json。官方链接经联网搜索核对标题；未宣称已完成远程PDF与本地文件的二进制一致性验证。']
for r in results:
 lines += ['',f"## Q{r['id']} {r['question']}",r['answer'],'',f"评价：{r['evaluation']['verdict']}；证据Recall@12："+json.dumps(r['evaluation']['evidence_recall_at_12'],ensure_ascii=False)+f"；融合公司覆盖 {r['evaluation']['company_coverage']}/12。",'', '|方法|排名|公司|PDF页|切块|BM25|向量|融合|','|---|---:|---|---:|---|---:|---:|---:|']
 for method,hits in r['retrieval'].items():
  for h in hits[:5]:lines.append(f"|{method}|{h['rank']}|{h['company']}|{h['page']}|{h['chunk_id']}|{h['bm25']:.4f}|{h['cosine']:.4f}|{h['fusion']:.4f}|")
 lines+=['','逐条完整Top12及全部单元格证据见逐题运行结果.json。']
(OUT/'财报检索与十题评价.md').write_text('\n'.join(lines),encoding='utf8')
sys.path.insert(0,str(BASE/'.report_libs'))
class PDF:
 def __init__(self):self.ops=[];self.size=10
 def setFont(self,name,size):self.size=size
 def drawString(self,x,y,s):self.ops.append(f'BT /F1 {self.size} Tf {x} {y} Td <'+s.encode('utf-16-be').hex()+'> Tj ET')
 def save(self):
  stream='\n'.join(self.ops).encode()
  objs=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 6 0 R >>',b'<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [5 0 R] >>',b'<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 4 >> >>',b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream']
  buf=bytearray(b'%PDF-1.4\n'); offsets=[0]
  for n,o in enumerate(objs,1):offsets.append(len(buf));buf.extend(str(n).encode()+b' 0 obj\n'+o+b'\nendobj\n')
  pos=len(buf);buf.extend(b'xref\n0 7\n0000000000 65535 f \n')
  for off in offsets[1:]:buf.extend(f'{off:010d} 00000 n \n'.encode())
  buf.extend(f'trailer << /Size 7 /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF'.encode());(OUT/'一页结论.pdf').write_bytes(buf)
c=PDF(); y=790
summary=['财报问答检索审计｜一页结论','2026年11月24日 · 12家公司 · 10道题（含2道全景题）','资料：12份2026年半年度完整报告，官方巨潮PDF链接已列明。',f"语料：{len(rows)}个切块；表格证据含PDF页码、表号、行号、列号与原始表头。",'检索：BM25 + 256维TF-IDF哈希向量；权重0.47/0.53。','该向量不是语义嵌入，不能据此宣称具备语义理解能力。','实跑：每题保存BM25、向量和融合三个Top12排名及原始分数。','评价：精确证据集合Recall@12；全景题额外检查跨公司覆盖。']
for r in results:summary.append(f"Q{r['id']} {'全景' if r['panoramic'] else '单公司'}：融合证据召回 {r['evaluation']['evidence_recall_at_12']['fusion']:.0%}；公司覆盖 {r['evaluation']['company_coverage']}/12")
summary+=['结论：单次Top12不能保证跨公司全景完整性，需分公司检索与合并。','已通过全量结构化表格补取提供十题答案；此步骤与直接检索分开评价。','改进：增加公司配额、财务指标映射、表头单位校验，再比较语义向量。','限制：未作人工盲评，未验证远程PDF与本地PDF的哈希一致性。','附件：财报检索与十题评价.md、逐题运行结果.json、官方来源与单元格.json。']
for j,line in enumerate(summary):
 c.setFont('STSong-Light',18 if j==0 else 10); c.drawString(38,y,line); y-=34 if j==0 else 25
c.save()
print(json.dumps({'chunks':len(rows),'sources':len(sources),'questions':len(results),'pdf':str(OUT/'一页结论.pdf')},ensure_ascii=True))
