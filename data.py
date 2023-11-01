import requests
import re

# for downloading the names of the athletes

url = "https://tiyu.baidu.com/major/delegation/26/tab/%E8%BF%90%E5%8A%A8%E5%91%98"
data=requests.get(url=url).content
with open('./data_china.txt','wb') as fp:
    fp.write(data)
print(re.match(data,r"(?<=<div class="name c-line-clamp1").+?(?=</div>)"))