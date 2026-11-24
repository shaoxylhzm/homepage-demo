# homepage-demo

- 个人主页：index.html
- 中国 AI 半年报问答：ai-report-qa/dist/index.html

问答页面使用浏览器端 Vector + BM25 混合检索，语料覆盖 12 家中国 AI 代表性公司的 2026 年半年报。

## 财报检索审计

入口： [检索实跑与证据审计](ai-report-qa/dist/audit/index.html)。包含12家公司官方完整报告来源、表格行列与页码、10题（含2道全景题）的证据和评价，以及BM25、哈希向量与融合各Top12实际分数。

[项目说明一页结论（Word）](ai-report-qa/docs/一页结论.docx) · [检索审计一页结论（PDF）](ai-report-qa/docs/一页结论_检索审计.pdf) · [原始运行结果](ai-report-qa/dist/audit/逐题运行结果.json)

使用现有256维哈希向量，并非语义嵌入；本次离线实跑不含页面启发式重排，补取答案与检索命中分开评价。
