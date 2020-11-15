# A COVID Fake News War
## COVID Fake News Data Is Always Drifting

Although many people read news online in social media nowaday, many people do not know they are under attack of fake news.

To understand how serious it is, we show the attacking trend of fake news  and find it higly related to current events. That is, this data is always drifting.

![Alt Text](https://media0.giphy.com/media/lTJK46k4l4kL4wBDEg/giphy.gif)

The current fake news is about COVID. We try to shows the changing wordclouds of the keywords in LDA topic models from 2020 Feb. to 2020 Nov from COVID related articles the factcheck website. We ring the words which indeeds show the hot topics at that moment.

Therefore, it is important for us to recognize them.

### The ML models should be always updated to keep up with the drift. 
The main issue here is that the fake news mainly spreads in social media such as Twitter. The tweets on Twitter have three main problems that make fake news detection difficult.

*   The tweets contain many unstructed contexts such as tags, hashtags, emojis, stickers, emoticons, and other website links, which normally harm the detection performance.
*   The number of tweets is large, which requires the tweet collection and labeling process to be fast.
*   The topics of tweets are more messy and changing more swiftly than the factcheck website.
 
![Alt Text](https://media2.giphy.com/media/v1AJSl8f7ZJG14Cz1W/giphy.gif)

We do not ring the words as we cannot recognize any obvious topics in LDA topic models in our labeled tweets data.

### Thus, we offer three solutions to the problems, respectively.

*   Data Cleaning
*   COVID Tweet Collection and Efficient Corroborative Labeling Process
*   The topics of tweets are more messy and changing more swiftly than the factcheck website.

## Let us start to fight against the ultimate evil of COVID Fake News on Twitter!

###   Data Cleaning

The raw tweets are not clear at all.

```
"😜 🤪 🤨 🧐 Coronaviris"
"RT @SKenson: As a gay man who came of age during the AIDS era of the 1980s, I\u2019m intensely curious for you, a Republican Senator, to tell me\u2026"
"@welt @kuku27 Jens Spahn: \"We took the corona virus very seriously from day one\"
"Hans de Boer (VNO-NCW): corona hakt fors in op economie | NOS https://t.co/zBTS0fRDSm"
```

We need to recognize the unstructed contexts and then clean them. 


1. **decode the tweets by HTML**
To see the most raw text features, we decide to decode the tweets by HTML (using python BeautifulSoup'lxml). This is because the tweets contain some punctuations with speicial presentation in collected data such as &quot=". 
2. **recognize tags and links**
The tags and url links will be useless for normal text analysis and machine learning models. Thus, we decide to remove it by regular expression.
3. **readable charaters uft-8 transformation**
The UTF-8 representation of the BOM is hexadecimal byte sequence. We need to convert it into readable charaters. Thus, we decode the text with ‘utf-8-sig’, which achieves this goal.
4. **Remove the hashtag, hashtags, emojis, stickers, and emoticons. Even digits**
The fake news detection should focus on the text. Thus, we use regular expression again to remove tokens other than alphabets and common punctuations such as [,?.].

After we complete all these steps, we convert the text to lowercase and refine the punctuation by removing some abnormal space.

```
"quarantine in united states diamondcruise american passengers. covid coronavirus"
"los alamos experts warn covid almost certainly cannot be contained, project up to. million dead"
"wtf is boosie? is that like the slime left behind a snail? not a fungus but not alive? not coronavirus. you ll get it."
``` 

