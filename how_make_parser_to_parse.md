# Описание работы парсера

Мы создадим два скрипта - первый с авторизацией и другой с самим парсером, потом будем работать в JupiterNotebook

```python
# после подключения нового номера телефона заходим на сайт - https://docs.pyrogram.org/intro/quickstart
# устанавливаем Pyrogram через PowerShell Prompt если работаем через Анаконду

pip3 install -U pyrogram
```

```python
# после этого регистрируем номер для Telegram API. Получили api_hash и api_id
# Дальше устанавливаем в PowerShell Prompt
pip3 install pyrogram tgcrypto
pip3 install python-dotenv
```

Далее создадим файл dot.env с нашими  api_hash и api_id

Структура у него примерно такая :

```jsx
API_ID=********
API_HASH=**с***fsdfsg**м***
DEMO=demo
```

Далее нам нужен скрипт с авторизацией 

Код :

```python
# !/usr/bin/python
# -*- coding: utf-8 -*-

from pyrogram import Client
import os
from dotenv import load_dotenv

path = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(path + '/dot.env')

if os.path.exists(dotenv_path):
	load_dotenv(dotenv_path)
	
API_ID = os.environ['API_ID']
API_HASH = os.environ['API_HASH']
with Client("my_account", API_ID, API_HASH) as app:
	app.send_message("me", "successfully authorized")
```

Далее мы запускаем этот скрипт в PowerShell Prompt. Авторизация пройдена (нам приходит об этом уведомление в телеграм)

Отбираем каналы для парсинга и подписываемся на них

Напишем и запустим скрипт для парсера, получим файл с данными - telegram.csv

```python
from pyrogram import Client
import os
from dotenv import load_dotenv
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))

dotenv_path = os.path.join(path + '/dot.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    
API_ID = os.environ['API_ID']
API_HASH = os.environ['API_HASH']

targets = ['datasciencejobs', 'bds_job', 'foranalysts', 'analyst_job', 'datajob' , \
                'datajobschannel', 'biheadhunter', 'data_hr']
all_messages = []

try:
    with Client("my_account", API_ID, API_HASH) as app:
        for target in targets:
            for message in app.get_chat_history(target, limit=3000):
                all_messages.append([message.sender_chat, message.id, message.date, message.text, message.entities])
    
    df = pd.DataFrame(all_messages)
    df.columns = ["chat", "message_id", "date", "text", "entities"]
    df.to_csv(path + '/telegram.csv', index=False)
    print('Success: ', path + '/telegram.csv')
except Exception as e:
    print('Error: ', e)
```

 Перейдём в JupiterNotebook для его обработки …