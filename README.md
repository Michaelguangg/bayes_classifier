数据
	来源于腾讯新闻，共7类，分别是1 财经，2 科技，3 汽车，4 房产，5 体育，6 娱乐，7 其他。文件的开头数字即表示新闻类型。
	数据是youthpasses爬虫爬取的，为了方便直接引用了其数据。链接为：https://github.com/youthpasses/bayes_classifier。
	
分类器
	朴素贝叶斯分类器
	
工具
	python+jieba+sklearn+numpy
	
流程
	中文分词存入中间表
	引入停用词
	TfidfVectorizer进行特征处理（向量化，TF-IDF和标准化三步处理）
	MultinomialNB创建文本分类器，并用测试集进行测试。score=0.827。
	找出每个文件中最大的TF-IDF值及其对应的词语
	