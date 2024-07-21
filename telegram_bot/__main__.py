'''Main module to run the telegram bot'''

import os
import logging
from telegram import Update # pylint: disable=import-error
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

import telegram_bot.functions.helper as helper_funcs
import telegram_bot.functions.scoring_api as api_funcs
import telegram_bot.configuration as config

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''Listens for conversation start messages & sends an explanitory gretting'''

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=config.BOT_GREETING)


async def score_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''Sends user provided text to scoring function'''

    # Get the logger
    logger = logging.getLogger(f'telegram_bot.score_text')

    # Get the message text
    text = update.message.text

    # Send the text to be scored
    submission = api_funcs.submit_text(suspect_text = text)
    result_id = await submission

    # Get the result, when ready
    result = api_funcs.retreive_result(result_id = result_id)
    author_call = await result
    reply = f'Author is likley {author_call}'

    logger.info(f'Got user text: {text}')
    logger.info(f'Result ID: {result_id}')
    logger.info(f'Reply: {reply}')

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=reply)


if __name__ == '__main__':

    # Start the logger
    logger = helper_funcs.start_logger(
        logfile_name = 'telegram_bot.log',
        logger_name = 'telegram_bot'
    )

    # Build app and add handlers
    application = ApplicationBuilder().token(os.environ['TELEGRAM_TOKEN']).build()
    start_handler = CommandHandler('start', start)
    score_text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), score_text)
    application.add_handler(start_handler)
    application.add_handler(score_text_handler)

    # Run the bot
    application.run_polling()
