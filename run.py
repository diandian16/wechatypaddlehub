# coding=utf-8

######  欢迎使用脚本任务，首先让我们熟悉脚本任务的一些使用规则  ######
# 脚本任务支持两种运行方式 

# 1.shell 脚本. 在 run.sh 中编写项目运行时所需的命令，并在启动命令框中填写 bash run.sh <参数1> <参数2>使脚本任务正常运行.

# 2.python 指令. 在 run.py 编写运行所需的代码，并在启动命令框中填写 python run.py <参数1> <参数2> 使脚本任务正常运行.

#注：run.sh、run.py 可使用自己的文件替代。

###数据集文件目录
# datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'

# 数据集文件具体路径请在编辑项目状态下通过左侧导航「数据集」中文件路径拷贝按钮获取
# train_datasets =  '通过路径拷贝获取真实数据集文件路径 '

# 输出文件目录. 任务完成后平台会自动把该目录所有文件压缩为tar.gz包，用户可以通过「下载输出」可以将输出信息下载到本地.
# output_dir = "/root/paddlejob/workspace/output"

# 日志记录. 任务会自动记录环境初始化日志、任务执行日志、错误日志、执行脚本中所有标准输出和标准出错流(例如print()),用户可以在「提交」任务后,通过「查看日志」追踪日志信息.

import os
import cv2
import asyncio
import numpy as np
import paddlehub as hub
import requests
import base64

from wechaty import (
    Contact,
    FileBox,
    Message,
    Wechaty,
    ScanStatus,
)

# 定义model
model = hub.Module(name='animegan_v2_shinkai_33', use_gpu=True)

# 在模型定义时，可以通过设置line=4或8指定输出绝句或律诗，设置word=5或7指定输出五言或七言。
# 默认line=4, word=7 即输出七言绝句。
module = hub.Module(name="ernie_gen_acrostic_poetry", line=4, word=7)


stylepro_artistic = hub.Module(name="stylepro_artistic")

def get_token():

# client_id 为官网获取的AK， client_secret 为官网获取的SK
    #url1 = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=22739331&client_secret=【官网获取的SK】'
    #response = requests.get(url1)
    url2='https://aip.baidubce.com/oauth/2.0/token'
    pa={
        'grant_type':'client_credentials',
        'client_id':'txD7je4svLUbcqVloxXnD1So',
        'client_secret':'cF6SKVIDgdTPInh68Oc7T5w5L6XwBqFy'
    }
    response=requests.post(url2,pa)

    if response:
        print(response.json())
    access_token = eval(response.text)['access_token']

    return access_token




def img_transform1(img_path, img_name):
    """
    人物动漫画的风格
    img_path1: 图片的路径
    img_name1: 图片的文件名
    """
    # 图片转换后存放的路径
    img_new_path1 = './image-new1/' + img_name

    # 模型预测
    request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/selfie_anime"  # 二进制方式打开图片文件

    with open(img_path, 'rb') as file:
        img = base64.b64encode(file.read())

    params = {
        'image':img,
        'access_token':get_token()
    }

    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    
    # 将图片保存到指定路径
    if response:
        with open(img_new_path1,'wb') as file:
        #print (response.json())
            donghua=response.json()['image']
            image= base64.b64decode(donghua)
            file.write(image)

    
    
    # 返回新图片的路径
    return img_new_path1






def img_transform(img_path, img_name):
    """
    将图片转换为新海诚《你的名字》、《天气之子》风格的图片
    img_path: 图片的路径
    img_name: 图片的文件名
    """
    # 图片转换后存放的路径
    img_new_path = './image-new/' + img_name

    # 模型预测
    result = model.style_transfer(images=[cv2.imread(img_path)])

    # 将图片保存到指定路径
    cv2.imwrite(img_new_path, result[0])

    # 返回新图片的路径
    return img_new_path

async def on_message(msg: Message):
    if msg.text() == 'ding':
        await msg.say('这是自动回复: dong dong dong')
        
    if msg.text() == '小队长' :
        await msg.say('我好想你的！！快点好起来，上班来看我！！')
        
    if msg.text() == '小仙女' :
        await msg.say('宇宙超级无敌可爱美丽的人，303之光！')
    
    if msg.text() == '小余' :
        await msg.say('一直想要骗我钱的女人！但是我还是很爱她！')

    if msg.text() == 'hi' or msg.text() == '你好':
        await msg.say('这是自动回复: 机器人目前的功能是\n- 收到"ding", 自动回复"dong dong dong"\n- 收到"图片", 自动回复一张图片\n- 还会把图片变成漫画样子，不过这个案例是参考细菌的\n- 收到"藏头诗：我是菜鸟"，自动以"我是菜鸟"作藏头诗一首，当然也可以输入其他的4个字作诗')
        
    if msg.text().startswith('藏头诗'): 
        #await msg.say('请输入4个字作为藏头诗的头')
        test_texts = [msg.text()[-4:]]
        results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
        #for result in results:
        await msg.say(results[0][0])
    
        

    if msg.text() == '图片':
        url = 'http://qrul2d5a1.hn-bkt.clouddn.com/image/street.jpg'
        file_box_12 = FileBox.from_url(url=url, name='xx.jpg')

        await msg.say(file_box_12)
    
    if msg.text() == '周深':
        url = 'https://z3.ax1x.com/2021/04/27/gC0EtA.jpg'
        file_box_11 = FileBox.from_url(url=url, name='xx.jpg')

        await msg.say(file_box_11)
    
    if msg.text() == '费玉清'or msg.text() == '小哥' :
        url = 'https://z3.ax1x.com/2021/04/27/gCBeUJ.jpg'
        
        # 构建一个FileBox
        file_box_1 = FileBox.from_url(url=url, name='xx.jpg')

        await msg.say(file_box_1)
    # 如果收到的message是一张图片
    if msg.type() == Message.Type.MESSAGE_TYPE_IMAGE:

        # 将Message转换为FileBox
        file_box_2 = await msg.to_file_box()

        # 获取图片名
        img_name = file_box_2.name

        # 图片保存的路径
        img_path = './image/' + img_name

        # 将图片保存为本地文件
        await file_box_2.to_file(file_path=img_path)

        # 调用图片风格转换的函数
        img_new_path = img_transform(img_path, img_name)
        img_new_path1 = img_transform1(img_path, img_name)

        # 从新的路径获取图片
        file_box_3 = FileBox.from_file(img_new_path)
        file_box_4 = FileBox.from_file(img_new_path1)
        await msg.say(file_box_4)
        
        


async def on_scan(
        qrcode: str,
        status: ScanStatus,
        _data,
):
    print('Status: ' + str(status))
    print('View QR Code Online: https://wechaty.js.org/qrcode/' + qrcode)


async def on_login(user: Contact):
    print(user)


async def main():
    # 确保我们在环境变量中设置了WECHATY_PUPPET_SERVICE_TOKEN
    if 'WECHATY_PUPPET_SERVICE_TOKEN' not in os.environ:
        print('''
            Error: WECHATY_PUPPET_SERVICE_TOKEN is not found in the environment variables
            You need a TOKEN to run the Python Wechaty. Please goto our README for details
            https://github.com/wechaty/python-wechaty-getting-started/#wechaty_puppet_service_token
        ''')

    bot = Wechaty()

    bot.on('scan',      on_scan)
    bot.on('login',     on_login)
    bot.on('message',   on_message)

    await bot.start()

    print('[Python Wechaty] Ding Dong Bot started.')


asyncio.run(main())
