# request
# for downloading pictures online
# Purpose: to get the data which is formed in pictures
import requests
if __name__ == "__main__":
    url = "https://ss2.baidu.com/-vo3dSag_xI4khGko9WTAnF6hhy/baike/w=268/sign=b97505d9a2c27d1ea5263cc223d4adaf/4afbfbedab64034f15ba757ba9c379310a551d53.jpg"
    img_data=requests.get(url=url).content

    with open('./test.jpg','wb') as fp:
        fp.write(img_data)