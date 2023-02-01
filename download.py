import gdown
import os

des = 'download'
os.mkdir(des)

url = "https://drive.google.com/file/d/1r_pi4oBYTDVnyvEtvj_aOCDqMgQdiF8E/view?usp=share_link"
output =  f"{des}/MoRe_Dataset.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)
