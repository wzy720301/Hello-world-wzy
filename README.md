# Hello-world-wzy
今天是2025年4月24日，天气真不错！开始学习GIT！！！世界真美好！Keep Going! Keep Driving!

### @Time     : 2025-04-24 16:03:50
### @Author   : 青岛未来城（乡）设计研究院有限公司 王智勇
### @File     : temp.py
### @Software : Visual Studio Code
### @WebSite  : https://blog.csdn.net/putidaxian
### @Phone    : 13370892288
### @E-mail   : wzy720301@163.com 729627596@qq.com
### @变量命名  :  https://blog.csdn.net/nikeylee/article/details/109383399#t0



### 2. 小红书种草文
- **指令模板**：以[身份]的口吻，写一篇[产品]的种草笔记，突出3个使用场景和2个痛点解决方案
- **示例**：以“健身博主”身份写一篇“运动耳机”的种草文，强调通勤、跑步、健身房场景，解决漏音和佩戴不适问题
- **输出示例**：  --------------
  “通勤路上听播客？这副耳机的降噪绝了！跑步狂甩不掉，健身房还能防水防汗…”

### 3. 小红书文案
- **指令模板**：以[身份]的口吻，写一篇[主题]的小红书文案，突出3个使用场景和2个痛点解决方案
- **示例**：以“美食博主”身份写一篇“美食探店”的小红书文案，突出“美食探店”场景，解决“美食探店”痛点

# 搭建科技文献机器学习训练数据集的详细步骤：

**一、确定目标和任务**

- 明确你希望通过机器学习解决的具体问题，例如对科技文献进行分类（如将其分为不同学科领域）、进行情感分析（判断文献中的情感倾向）、信息抽取（提取关键信息，如作者、研究方法、结论等）或摘要生成（根据文献内容生成摘要）等。

**二、数据收集**

- **从公共数据库获取**：
  
  - 可以利用学术数据库，如Web of Science、IEEE Xplore、ACM Digital Library等，它们存储了大量的科技文献。部分数据库提供API接口，你可以通过编写程序调用API来获取数据。
  - 开放获取平台，像arXiv，是一个很好的免费论文来源，许多预印本论文可在此找到。

- **网络爬虫**：
  
  - 对于一些没有开放API的网站，你可以编写网络爬虫程序。例如，使用Python的`requests`和`BeautifulSoup`库。
    
    ```python
    import requests
    from bs4 import BeautifulSoup
    
    ```
  
  def fetch_papers(url):
  
      response = requests.get(url)
      if response.status_code == 200:
          soup = BeautifulSoup(response.text, 'html.parser')
          # 这里需要根据目标网站的HTML结构来提取文献信息
          # 例如，假设文献标题在<h1>标签中，摘要在<p class="abstract">标签中
          papers = []
          for title in soup.find_all('h1'):
              paper_title = title.text
              abstract = soup.find('p', class_='abstract').text
              papers.append((paper_title, abstract))
          return papers
      else:
          print(f"Failed to fetch data from {url}, status code: {response.status_code}")
          return []
  
  ```
  这段代码使用`requests`库发送HTTP请求获取网页内容，`BeautifulSoup`库解析HTML。`fetch_papers`函数接收一个URL，发送请求并解析网页。根据HTML结构提取文献标题（假设在`<h1>`标签）和摘要（假设在`<p class="abstract">`标签）。
  ```

- **数据购买**：
  
  - 一些商业数据提供商提供整理好的数据，你可以根据自己的需求购买相应的数据，但成本较高。

**三、数据预处理**

- **清理数据**：
  
  - 去除HTML标签、特殊字符和多余的空格，例如使用Python的`re`库：
    
    ```python
    import re
    
    ```
  
  def clean_text(text):
  
      # 去除HTML标签
      clean = re.compile('<.*?>')
      text = re.sub(clean, '', text)
      # 去除特殊字符
      text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
      # 去除多余空格
      text = re.sub(r'\s+', ' ', text).strip()
      return text
  
  ```
  该函数首先使用正则表达式`<.*?>`去除HTML标签，然后将非字母数字和空格的字符替换为空，最后将多个空格替换为一个空格并去除前后多余空格。
  ```

- **数据标准化**：
  
  - 将文本转换为统一的格式，例如全部转换为小写：
    
    ```python
    def standardize_text(text):
      return text.lower()
    ```
    
    这将文本中的所有字符转换为小写，使后续处理更方便，因为模型通常对大小写不敏感。

**四、数据标注（如果需要）**

- **人工标注**：
  - 对于分类任务，需要人工给数据打标签。例如，将文献标记为“物理学”“化学”“生物学”等类别。
- **自动标注工具**：
  - 对于一些简单的任务，如情感分析，可以使用一些开源的标注工具辅助标注，如Stanford CoreNLP等。

**五、特征提取**

- **词袋模型**：
  
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(cleaned_texts)
  ```
  
  `CountVectorizer`将文本转换为词袋表示，其中每个文档被表示为一个向量，向量的每个元素表示对应单词的出现次数。

- **TF-IDF**：
  
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  
  tfidf_vectorizer = TfidfVectorizer()
  X = tfidf_vectorizer.fit_transform(cleaned_texts)
  ```
  
  `TfidfVectorizer`将文本转换为TF-IDF表示，不仅考虑单词的出现次数，还考虑其在整个语料库中的重要性，避免一些常见词（如“the”“and”）对结果产生过大影响。

**六、数据划分**

- **使用sklearn**：
  
  ```python
  from sklearn.model_selection import train_test_split
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```
  
  `train_test_split`函数将数据`X`（特征）和标签`y`划分为训练集`X_train`、`y_train`和测试集`X_test`、`y_test`，其中测试集占比为20%（`test_size=0.2`），`random_state`确保结果可重复。

**七、数据存储和管理**

- **存储为文件**：
  
  - 可以将数据存储为CSV文件，方便后续读取和使用：
    
    ```python
    import pandas as pd
    
    ```
  
  train_df = pd.DataFrame({'text': X_train, 'label': y_train})
  test_df = pd.DataFrame({'text': X_test, 'label': y_test})
  train_df.to_csv('train_data.csv', index=False)
  test_df.to_csv('test_data.csv', index=False)
  
  ```
  这段代码使用`pandas`将训练集和测试集存储为CSV文件，方便保存和后续操作。
  ```

- **使用数据库**：
  
  - 对于大规模数据集，可以使用数据库（如SQLite、MySQL等）存储数据，以便更高效地管理和查询。

**八、数据增强（可选）**

- **同义词替换**：
  
  - 对于文本数据，可以使用同义词典将一些词替换为其同义词，例如使用`nltk`库的同义词资源：
    
    ```python
    import nltk
    from nltk.corpus import wordnet
    
    ```
  
  def get_synonyms(word):
  
      synonyms = []
      for syn in wordnet.synsets(word):
          for lemma in syn.lemmas():
              synonyms.append(lemma.name())
      return synonyms
  
  def synonym_replacement(text):
  
      words = text.split()
      new_text = []
      for word in words:
          syns = get_synonyms(word)
          if syns:
              new_text.append(syns[0])
          else:
              new_text.append(word)
      return ' '.join(new_text)
  
  ```
  此函数使用`nltk`的`wordnet`查找词的同义词，将文本中的部分词替换为同义词，从而扩充数据。
  
  
  ```

```

 通过以上步骤，你可以搭建一个完整的科技文献机器学习训练数据集。首先根据任务确定目标，然后收集数据，对数据进行预处理和标注，提取特征，划分数据集，存储和管理数据，最后可以考虑数据增强，为后续的机器学习模型训练做好准备。


```

