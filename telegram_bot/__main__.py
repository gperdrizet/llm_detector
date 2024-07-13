'''Main module to run the telegram bot'''

import os
import asyncio
import telegram

async def main():
    '''Main async bot function'''

    bot = telegram.Bot(os.environ['TELEGRAM_TOKEN'])

    async with bot:
        print(await bot.get_me())


if __name__ == '__main__':

    asyncio.run(main())
