# 中国 AI 半年报问答

纯静态页面，包含 4,371 个半年报切块、BM25 倒排索引和 256 维哈希 TF-IDF 向量索引。

在仓库根目录运行 python -m http.server 8000，然后打开 http://localhost:8000/ai-report-qa/dist/。

- dist/index.html：页面结构
- dist/app.js：混合检索和答案引用
- dist/styles.css：页面样式
- dist/data/：语料、BM25 索引和向量数据
- build_index.py：索引构建脚本

仓库已包含可直接使用的索引。重新构建时，脚本默认读取仓库外的知识库切块文件。
