# Malone

## News

**2024-07-08**: llm_detector is officially part of the Backdrop Build V5 cohort under the tentative name 'malone' starting today. Check out the backdrop [build page](https://backdropbuild.com/builds/v5/cadmus) for updates.

**2024-07-30**: Malone is live in Beta on Telegram, give it a try [here](https://t.me/the_malone_bot). Note: some Firefox users have reported issues with the botlink, you can also find malone by messaging '*/start*' to @the_malone_bot anywhere you use Telegram.

![malone](https://github.com/gperdrizet/llm_detector/blob/main/telegram_bot/assets/malone_A.jpg?raw=true)

Malone is a synthetic text detection service available on [Telegram Messenger](https://telegram.org/), written in Python using [HuggingFace](https://huggingface.co), [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://github.com/dmlc/xgboost), [Luigi](https://github.com/spotify/luigi) and [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot), supported by [Flask](https://flask.palletsprojects.com/en/3.0.x), [Celery](https://docs.celeryq.dev/en/stable/index.html), [Redis](https://redis.io/) & [Docker](https://www.docker.com/) and served via [Gunicorn](https://gunicorn.org/) and [Nginx](https://nginx.org/). Malone uses an in-house trained gradient boosting classifier to estimate the probability that a given text was generated by an LLM. It uses a set of engineered features derived from the input text, for more details see the [feature engineering notebooks](https://github.com/gperdrizet/llm_detector/tree/main/classifier/notebooks).

## Table of Contents

1. Features
2. Where to find malone
3. Usage
4. Performance
5. Demonstration/experimentation notebooks
6. About the author
7. Disclaimer

## 1. Features

- **Easily accessible** - use it anywhere you can access Telegram: iOS or Android apps and any web browser.
- **Simple interface** - no frills, just send the bot text and it will send back the probability that the text was machine generated.
- **Useful and accurate** - provides a probability that text is synthetic, allowing users to make their own decisions when evaluating content. Maximum likelihood classification accuracy ~90% on held-out test data.
- **Model agnostic** - malone is not trained to detect the output of a specific LLM, instead, it uses a gradient boosting classifier and a set of numerical features derived from/calibrated on a large corpus of human and synthetic text samples from multiple LLMs.
- **No logs** - no user data or message contents are ever persisted to disk.
- **Open source codebase** - malone is an open source project. Clone it, fork it, extend it, modify it, host it yourself and use it the way you want to use it.
- **Free**

## 2. Where to find malone

Malone is publicly available on Telegram. You can find malone on the [Telegram bot page](https://t.me/the_malone_bot), or just message @the_malone_bot with '/*start*' to start using it.

There are also plans in the works to offer the bare API to interested parties. If that's you, see section 6 below.

## 3. Usage

To use malone you will need a Telegram account. Telegram is free to use and available as an app for iOS and Android. There is also a web version for desktop use.

Once you have a Telegram account, malone is simple to use. Send the bot any 'suspect' text and it will reply with the probability that the text in question was written by a human or generated by an LLM. For smartphone use, a good trick is long press on 'suspect' text and then share it to malone on Telegram via the context menu. Malone is never more that 2 taps away!

![telegram app screenshot](https://github.com/gperdrizet/llm_detector/blob/main/telegram_bot/assets/telegram_screenshot.jpg?raw=true)

Malone can run in two response modes: 'default' and 'verbose'. Default mode returns the probability associated with the most likely class as a percent (e.g. 75% chance a human wrote this). Verbose mode gives a little more detail about the feature values and prediction metrics. Set the mode by messaging '*/set_mode verbose*' or '*/set_mode default*'.

For best results, submitted text must be between 50 and 500 words.

## 4. Performance

Malone is ~90% accurate with a binary log loss of ~0.25 on hold-out test data depending on the model and feature engineering hyperparameters and the specific train/test split (see example confusion matrix below). The miss-classified examples are more or less evenly split between false negatives and false positives.

![XGBoost confusion matrix](https://github.com/gperdrizet/llm_detector/blob/main/classifier/notebooks/figures/XGBoost_confusion_matrix.png?raw=true)

For more details on the classifier training and performance see: [XGBoost experimentation](https://github.com/gperdrizet/llm_detector/blob/main/classifier/notebooks/04.2-XGBoost_classifier_experimentation.ipynb) and [XGBoost finalized](https://github.com/gperdrizet/llm_detector/blob/main/classifier/notebooks/04.2-XGBoost_classifier_finalized.ipynb).

## 5. Demonstration/experimentation notebooks

Most of the testing and benchmarking during the design phase of the project was trialed in Jupyter notebooks before refactoring into modules. These notebooks are the best way to understand the approach and the engineered features used to train the classifier.

1. [Human and synthetic text training data](https://github.com/gperdrizet/llm_detector/blob/main/classifier/notebooks/01-hans_2024_data.ipynb)
2. [Perplexity ratio score](https://github.com/gperdrizet/llm_detector/blob/main/classifier/notebooks/02.2-perplexity_ratio_score_finalized.ipynb)
3. [TF-IDF score](https://github.com/gperdrizet/llm_detector/blob/main/classifier/notebooks/03.2-TF-IDF_finalized.ipynb)
4. [XGBoost classifier](https://github.com/gperdrizet/llm_detector/blob/main/classifier/notebooks/04.2-XGBoost_classifier_finalized.ipynb)

## 6. About the author

My name is Dr. George Perdrizet, I am a biochemistry & molecular biology PhD seeking a career step from academia to professional data science and/or machine learning engineering. This project was conceived from the scientific literature and built solo over the course of a few weeks - I strongly believe that I have a ton to offer the right organization. If you or anyone you know is interested in an ex-researcher from University of Chicago turned builder and data scientist, please reach out, I'd love to learn from and contribute to your project.

- **Email**: <hire.me@perdrizet.org>
- **LinkedIn**: linkedin.com/gperdrizet

## 7. Disclaimer

Malone is an experimental research project meant for educational, informational and entertainment purposes only. Any predictions made are inherently probabilistic in nature and subject to stochastic errors. Text classifications, no matter how high or low the reported probability, should never be interpreted as proof of authorship or the lack thereof in regard to any text submitted for analysis. Decisions about the source or value of any text are made by the user who considers all factors relevant to themselves and their purpose and takes full responsibility for their own judgment any and actions they may take as a result.
