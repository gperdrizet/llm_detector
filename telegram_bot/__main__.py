'''Main module to run the telegram bot'''

import os
import logging
import time
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

import telegram_bot.functions.helper as helper_funcs
import telegram_bot.functions.scoring_api as api_funcs
import telegram_bot.configuration as config

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''Listens for conversation start messages & sends an explanatory greeting.'''

    await context.bot.send_message(
        chat_id = update.effective_chat.id, text = config.BOT_GREETING)

async def set_response_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''Sets user's desired response mode.'''

    # Get the logger
    function_logger = logging.getLogger('telegram_bot.set_response_mode')

    response_mode = update.message.text.partition(' ')[2]
    context.user_data['response_mode'] = response_mode

    function_logger.info('Set users response mode to: %s', context.user_data['response_mode'])

    await context.bot.send_message(
        chat_id = update.effective_chat.id, text = f'Set response mode to {response_mode}')

async def score_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''Sends user provided text to scoring function, sends
    result back to user.'''

    time_fragment_received = time.time()

    # Get the logger
    function_logger = logging.getLogger('telegram_bot.score_text')

    # Get the message text
    text = update.message.text
    function_logger.info('Got text fragment from user')

    # Check the user's chosen response mode, setting default if
    # it hasn't been set yet
    if 'response_mode' not in context.user_data.keys():
        context.user_data['response_mode'] = 'default'

    response_mode = context.user_data['response_mode']
    function_logger.debug('User has requested %s response mode', response_mode)

    # Send the text to be scored
    submission = api_funcs.submit_text(suspect_text = text, response_mode = response_mode)
    result_id = await submission

    # Get the result, when ready
    result = api_funcs.retrieve_result(result_id = result_id)
    reply = await result

    function_logger.info('Result ID: %s', result_id)

    await context.bot.send_message(
        chat_id = update.effective_chat.id, text = reply)

    time_reply_sent = time.time()

    with open(config.FRAGMENT_TURNAROUND_DATA, 'a+', encoding = 'utf-8') as f:
        f.write(f'{time_fragment_received},{time_reply_sent}\n')

if __name__ == '__main__':

    # Start the logger
    logger = helper_funcs.start_logger(
        logfile_name = 'telegram_bot.log',
        logger_name = 'telegram_bot'
    )

    # Build app and add handlers
    application = ApplicationBuilder().token(os.environ['TELEGRAM_TOKEN']).build()
    start_handler = CommandHandler('start', start)
    response_mode_handler = CommandHandler('response_mode', set_response_mode)
    score_text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), score_text)

    application.add_handler(start_handler)
    application.add_handler(response_mode_handler)
    application.add_handler(score_text_handler)

    # Run the bot
    application.run_polling()
