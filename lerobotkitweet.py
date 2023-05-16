#!/usr/bin/env python3
# Imports du module standard Python
import argparse
import datetime
import locale
import logging
import os
import random
import re
import sys

# Imports de modules externes
import requests
import json
import yaml
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper.article import ArticleException
import tweepy
import feedparser
from difflib import SequenceMatcher
import numpy as np

# Téléchargements de données pour nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Variables for log
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(f'{current_dir}/application.yml', 'r') as stream:
    config = yaml.safe_load(stream)
log_dir = config['logging']['dir_name']
log_name = config['logging']['file_name']
script_name = os.path.basename(__file__)

# Variables globals
google_trend_rss = 'https://trends.google.com/trends/trendingsearches/daily/rss?geo=FR&hl=fr'

# Création du répertoire de log s'il n'existe pas
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configuration du logging
logging.basicConfig(
    filename=f"{log_dir}/{log_name}",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logging.info("{:=^90}".format(""))


def article_is_published(article_to_compare: str, tolerance: int = 50, comparison_type: str = 'title') -> bool:
    tolerance = float(tolerance) / 100
    logging.info(f"   Function article_is_published : lancement")
    logging.info(f"   Function article_is_published : paramètre  article_to_compare")
    logging.info(f"   Function article_is_published : paramètre  tolerance {tolerance}")
    logging.info(f"   Function article_is_published : paramètre  comparison_type {comparison_type}")
    logging.info(f"   Function article_is_published : paramètre  {comparison_type}: {article_to_compare}")
    with open(f'{current_dir}/articles.json', 'r') as f:
        data = json.load(f)

    lemmatizer = WordNetLemmatizer()
    tokens_to_compare = word_tokenize(article_to_compare)
    lemmas_to_compare = [lemmatizer.lemmatize(token) for token in tokens_to_compare]

    for article in data['articles']:
        text_published = article[comparison_type]
        tokens_published = word_tokenize(text_published)
        lemmas_published = [lemmatizer.lemmatize(token) for token in tokens_published]
        ratio = SequenceMatcher(None, ' '.join(tokens_published), ' '.join(lemmas_to_compare)).ratio()
        logging.debug(f"Function article_is_published : ratio: {ratio}")
        if ratio >= tolerance:
            logging.warning(f"Function article_is_published : return True")
            return True
    logging.info(f"   Function article_is_published : return False")
    return False


def get_articles(google_news, excluded_terms=None):
    logging.info(f"   Function get_articles         : lancement")
    logging.info(f"   Function get_articles         : paramètre google_news")
    if excluded_terms is None:
        excluded_terms = []
    if len(google_news) > 0:
        for article in google_news[1:200]:
            article_title = article["title"]
            logging.info(f"   Function get_articles         : appel fonction is_safe_search article_title")
            if not is_safe_search(article_title):
                print('Title not safe')
                logging.warning(f"   Function get_articles         : article {article_title} a été rejeté")
                continue
            logging.info(f"   Function get_articles         : appel fonction article_is_published")
            if article_is_published(article_title, 80, 'title'):
                logging.info(f"   Function get_articles         : article {article_title} a déjà été publié.")
                continue
            article_url = article["link"]
            if "http" not in article_url:
                logging.warning(f"   Function get_articles         : URL {article_url} incorrecte")
                continue
            logging.info(f"   Function get_articles         : appel fonction is_safe_search article_url")
            if not is_safe_search(article_url):
                logging.warning(f"   Function get_articles         : URL {article_url} a été rejeté")
                continue
            logging.info(f"   Function get_articles         : titre article: {article_title}")
            logging.info(f"   Function get_articles         : date parution article : {article['date']}")
            logging.info(f"   Function get_articles         : URL article : {article_url}")
            try:
                article = Article(article_url)
                article.download()
                article.parse()
                article_content = article.text
                logging.info(f"   Function get_articles         : longueur article : {len(article_content)}")
                article_date = article.publish_date
                formated_content = unidecode(article_content.replace(" ", "-").lower())
            except ArticleException as ae:
                logging.error(f" Function get_articles         : Erreur de traitement de l'article : {article_url}. ArticleException : {ae}")
                continue
            except Exception as e:
                logging.error(f" Function get_articles         : Erreur de traitement de l'article : {article_url}. Erreur : {e}")
                continue
            logging.info(f"   Function get_articles         : appel fonction is_safe_search formated_content")
            if not is_safe_search(formated_content):
                logging.info(f"   Function get_articles         : le contenu de l'article a été rejeté (contenu)")
                continue
            return article_title, article_url, article_date, article_content
        logging.error(f" Function get_articles         : plus aucun résultat n'a été trouvé pour la recherche marmi {len(google_news)} articles")
        push_last_log_to_web()
        sys.exit(1)
    else:
        logging.warning(f"   Function get_articles         : aucun résultat n'a été trouvé pour la recherche")
        push_last_log_to_web()
        sys.exit(1)


def get_clean_tweet(tweet):
    # Retire les espaces en début et en fin de tweet
    tweet = tweet.strip()
    # Remplace les doubles espaces par des espaces simples
    tweet = ' '.join(tweet.split())
    # Retire les doublons de "#" en ne laissant qu'une seule occurrence
    tweet = remove_duplicate_hashtags(tweet)
    # Supprime les hashtags qui ne consistent qu'en une seule lettre
    tweet = ' '.join(word for word in tweet.split() if not (word.startswith("#") and len(word) == 2))
    # Retourne le tweet nettoyé
    return tweet


def get_google_news(subject: str, lang: str = 'fr'):
    logging.info(f"   Function get_google_news      : lancement")
    logging.info(f"   Function get_google_news      : paramètre subject : {subject}")
    googlenews = GoogleNews(lang=lang, region="com", period='1d')
    try:
        googlenews.search(subject)
        if not googlenews.result():
            logging.error(f"  Function get_google_news      : aucun résultat n'a été trouvé pour la recherche.")
            sys.exit(1)
    except Exception as e:
        logging.error(f" Function get_google_news      : erreur s'est produite lors de l'appel à la méthode search() : {e}")
        sys.exit()
    try:
        googlenews = googlenews.result()
    except AttributeError as e:
        logging.error(f" Function get_google_news      : une erreur s'est produite lors de l'appel à la méthode result() : {e}")
        sys.exit()
    return googlenews


def get_google_trends(url):
    logging.info(f"   Function get_google_news      : lancement")
    logging.info(f"   Function get_google_news      : paramètre url : {url}")
    # Charger le flux RSS
    feed = feedparser.parse(url)
    # Initialiser la liste des résultats
    results = []
    # Parcourir tous les items du flux RSS
    for item in feed.entries:
        # Récupérer les informations nécessaires pour chaque item
        title = item.title
        logging.info(f"   Function get_google_news      : titre article: {title}")
        approx_traffic = item.get("ht_approx_traffic", "")
        description = item.description
        link = item.link
        # Ajouter les informations à la liste des résultats
        results.append({
            "titre": title,
            "trafic_approximatif": approx_traffic,
            "description": description,
            "lien": link
        })
        return results[0]['titre']


def get_gpt_response(prompt: str, temperature: float = 0.8, lang: str = 'fr'):
    try:
        logging.info(f"   Function get_gpt_response     : lancement")
        logging.info(f"   Function get_gpt_response     : paramètre prompt")
        logging.info(f"   Function get_gpt_response     : paramètre temerature: {temperature}")
        OPENAI_API_KEY = config['chat-GPT']['openai_key']
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENAI_API_KEY,
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "sytem",
                    "content": config['chat-GPT']['message']['content'][lang],
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
            "max_tokens": config['chat-GPT']['max_tokens'],
        }
        logging.info(f"   Function get_gpt_response     : appel de l'API GPT")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = json.loads(response.content)
        logging.info(f"   Function get_gpt_response     : nombre de token utilisés : {response_data['usage']['total_tokens']}")
        return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f" Function get_gpt_response     : erreur lors de l'appel à l'API OpenAI : {response_data['error']['message']}")
        print(e)
        push_last_log_to_web()
        sys.exit(1)


def get_prompt(article_content: str = None, subject: str = None, lang: str = 'fr'):
    logging.info(f"   Function get_prompt           : lancement")
    logging.info(f"   Function get_prompt           : paramètre article_content")
    logging.info(f"   Function get_prompt           : paramètre subject : {subject}")
    logging.info(f"   Function get_prompt           : paramètre lang : {lang}")
    if subject == 'humeur_matin':
        prompt = config['chat-GPT']['prompts']['message_morning'][lang]
    elif subject == 'humeur_soir':
        prompt = config['chat-GPT']['prompts']['message_evening'][lang]
    elif subject == 'cinema':
        prompt = f"{config['chat-GPT']['prompts']['cinema'][lang]} {article_content}"
    elif subject == 'etienne_klein':
        prompt = f"{config['chat-GPT']['prompts']['etienne_klein'][lang]} {article_content}"
    elif subject == 'too_long':
        prompt = f"{config['chat-GPT']['prompts']['too_long'][lang]} {article_content}"
    elif subject == 'twitter_trends':
        prompt = f"{config['chat-GPT']['prompts']['twitter_trends'][lang]} {article_content}"
    else:
        prompt = f"{config['chat-GPT']['prompts']['resume_article'][lang]} {article_content}"
    return prompt


def get_subject():
    logging.info(f"   Function get_subject          : lancement")
    subjects = config['subjects']
    poids = [subjects[key]['weight'] for key in subjects]
    chosen_subject = random.choices(list(subjects.keys()), weights=poids, k=1)[0]
    logging.info(f"   Function get_subject          : sujet obtenu : {chosen_subject}")
    return subjects[chosen_subject]['name'], subjects[chosen_subject]['lang']


def get_twitter_trends():
    logging.info(f"   Function get_twitter_trends   : lancement")
    url = "https://www.twitter-trending.com/rss/feed?c=france&gmt_z=Europe/Paris&l=fr"
    feed = feedparser.parse(url)
    items = [[]]
    last_title = feed.entries[0].title
    for entry in feed.entries:
        title = entry.title
        content = entry.content[0].value
        content = content.replace('<p>', '').replace('</p>', '')
        items_list = content.split(') ')
        items_list[-1] = items_list[-1].replace('..[top50]', '').strip()
        items_list = [f"{item.rsplit(' ', 1)[0]}" for item in items_list]
        if title != last_title:
            last_title = title
            items.append(items_list[1:])
        else:
            items[-1] += items_list[1:]
    trends = [f"Trend_{i*30}" for i in range(len(items))]
    trend_arrays = [np.array(items[i]) for i in range(len(trends))]
    logging.info(f"   Function get_twitter_trends   : trend obtenu : {trend_arrays[0][0]}")
    return trend_arrays


def inspect_launch_args(subject):
    logging.info(f"   Function inspect_launch_args  : lancement")
    logging.info(f"   Function inspect_launch_args  : paramètre subject : {subject}")
    if os.getenv("SHELL") == "/bin/sh":
        logging.info(f"   Function inspect_launch_args  : script lancé par cron")
    else:
        logging.info(f"   Function inspect_launch_args  : script lancé manuellement")

def is_safe_search(search: str):
    excluded_terms = config['excluded_terms']
    logging.info(f"   Function is_safe_search       : lancement")
    logging.info(f"   Function is_safe_search       : paramètre search")
    excluded_terms_check = [unidecode(term.replace(" ", "-").lower()) for term in excluded_terms]
    search_safer = unidecode(search.replace(" ", "-").lower())
    for term in excluded_terms_check:
        if term in search_safer:
            logging.warning(f"Function is_safe_search       : terme exclu {term} trouvé dans la recherche")
            logging.warning(f"Function is_safe_search       : return False")
            return False
    logging.info(f"   Function is_safe_search       : return True")
    return True


def parse_arguments ():
    logging.info(f"   Function parse_arguments      : lancement")
    authorized_subjects = list(config['subjects_custom'].keys())
    parser = argparse.ArgumentParser(description='Tweet Generator')
    parser.add_argument('-s', '--subject', choices=authorized_subjects, help='Subject to tweet about', default='random')
    parser.add_argument('--lang', help='Language of the subject', default='fr')
    args = parser.parse_args()

    # Initialisation des variables
    hashtag = ""
    subject = args.subject
    lang = args.lang
    search_activated = True
    if subject == 'random':
        logging.info(f"   Function parse_arguments      : appel fonction get_subject")
        subject, lang = get_subject()
    elif subject == 'etienne_klein':
        lang = config['subjects_custom']['etienne_klein']['lang']
        subject = 'etienne klein'
    elif subject == 'cinema':
        locale.setlocale(locale.LC_ALL, 'fr_FR.utf8')
        today = datetime.date.today()
        days_ahead = (2 - today.weekday()) % 7
        next_wednesday = today + datetime.timedelta(days=days_ahead)
        next_cinema = next_wednesday.strftime('%d %B %Y')
        lang = config['subjects_custom']['cinema']['lang']
        subject = f"sorties cinema {next_cinema}"
    elif subject == 'google_trends':
        logging.info(f"   Function parse_arguments      : appel fonction get_google_trends")
        subject = get_google_trends(google_trend_rss)
        lang = config['subjects_custom']['google_trends']['lang']
    elif subject == 'twitter_trends':
        logging.info(f"   Function parse_arguments      : appel fonction get_twitter_trends")
        trends_array = get_twitter_trends()
        twitter_trend = trends_array[0][0]
        hashtag = twitter_trend
        if not hashtag.startswith("#"):
            hashtag = "#" + hashtag
        lang = config['subjects_custom']['twitter_trends']['lang']
        logging.info(f"   Function parse_arguments      : appel fonction get_prompt")
        prompt = get_prompt(hashtag, subject, lang)
        logging.info(f"   Function parse_arguments      : appel fonction get_gpt_response")
        subject = get_gpt_response(prompt, 0.1)
        search_activated = True
    else:
        search_activated = False
    logging.info(f"   Function parse_arguments      : argument subject : {subject}")
    logging.info(f"   Function parse_arguments      : argument hashtag : {hashtag}")
    logging.info(f"   Function parse_arguments      : argument lang : {lang}")
    logging.info(f"   Function parse_arguments      : argument search_activated : {search_activated}")
    return subject,hashtag,lang,search_activated


def publish_tweet(client, tweet_content: str, article_url: str = None):
    try:
        logging.info(f"   Function publish_tweet        : lancement")
        logging.info(f"   Function publish_tweet        : paramètre client {client}")
        logging.info(f"   Function publish_tweet        : paramètre tweet_content {tweet_content}")
        logging.info(f"   Function publish_tweet        : paramètre article_url {article_url}")
        if article_url is not None:
            tweet_content += f" {article_url}"
        tweet_content = tweet_content.replace('"', '')
        logging.info(f"   Function publish_tweet        : publication du Tweet")
        logging.info(f"   Function publish_tweet        : longueur du tweet : {len(tweet_content)}")
        response = client.create_tweet(text=tweet_content)
        return response
    except Exception as e:
        logging.error(f" Function publish_tweet        : erreur lors de la publication du Tweet {tweet_content}: {e}")
        push_last_log_to_web()
        return None


def push_article_json(article_title: str, article_url: str):
    logging.info(f"   Function push_article_json    : lancement")
    with open(f'{current_dir}/articles.json', 'r') as f:
        articles = json.load(f)
    new_article = {
        "title": article_title,
        "url": article_url,
    }
    articles["articles"].append(new_article)
    with open(f'{current_dir}/articles.json', 'w') as f:
        json.dump(articles, f, indent=4)

def push_last_log_to_web():
    logging.info(f"   Function push_last_log_to_web : lancement")
    with open(f"{log_dir}/{log_name}","r") as source_file:
        lines = source_file.readlines()[-200:]
        lines.reverse()
        with open('/var/www/html/lrto.txt',"w") as target_file:
            for line in lines:
                target_file.write(line)


def remove_duplicate_hashtags(text):
    words = text.split()
    unique_words = []
    seen_hashtags = set()
    for word in words:
        if word.startswith("#"):
            if word not in seen_hashtags:
                seen_hashtags.add(word)
                unique_words.append(word)
            continue
        unique_words.append(word)
    return ' '.join(unique_words)


def tweepy_client():
    logging.info(f"   Function tweepy_client        : lancement")
    return tweepy.Client(
        consumer_key=config['tweeter_keys']['consumer_key'],
        consumer_secret=config['tweeter_keys']['consumer_secret'],
        access_token=config['tweeter_keys']['access_token'],
        access_token_secret=config['tweeter_keys']['access_token_secret']
    )


def main():
    try:
        article_url = None
        logging.info(f"   Main                          : début programme {script_name}")
        logging.info(f"   Main                          : appel fonction parse_arguments")
        subject,hashtag,lang,search_activated = parse_arguments ()
        logging.info(f"   Main                          : appel fonction inspect_launch_args")
        inspect_launch_args(subject)
        # Vérification si le sujet est autorisé par la fonction is_safe_search()
        logging.info(f"   Main                          : appel fonction is_safe_search")
        if is_safe_search(subject):
            if search_activated:
                # Récupération des articles de Google News liés au sujet
                logging.info(f"   Main                          : appel fonction get_google_news")
                google_news = get_google_news(subject, lang)
                # Extraction de l'article pertinent
                logging.info(f"   Main                          : appel fonction get_articles")
                article_title, article_url, article_date, article_content = get_articles(google_news)
            else:
                article_content = None
        else:
            # Si le sujet n'est pas autorisé par is_safe_search(), on quitte le script
            push_last_log_to_web()
            sys.exit()
        # Ajout de l'article dans le fichier JSON des articles
        if subject not in ['humeur_soir', 'humeur_matin']:
            logging.info(f"   Main                          : appel fonction push_article_json")
            push_article_json(article_title, article_url)

        # Obtention de la phrase d'accroche à partir de l'article
        logging.info(f"   Main                          : appel fonction get_prompt")
        prompt = get_prompt(article_content, subject, lang)
        client = tweepy_client()

        # Génération de la réponse GPT-3 à partir de la phrase d'accroche
        logging.info(f"   Main                          : appel fonction get_gpt_response")
        tweet = f"{get_gpt_response(prompt, 0.8, lang)} {hashtag}"
        tweet = get_clean_tweet(tweet)
        # Vérification si le tweet n'est pas vide
        if tweet is not None:
            # Vérification si le tweet est trop long
            max_tweet_length = 250
            max_attempts = 3
            attempts = 0
            while len(tweet) > max_tweet_length and attempts < max_attempts:
                logging.warning(f"Main                          : impossible de tweeter : {tweet} (longueur : {len(tweet)})")
                logging.info(f"   Main                          : appel fonction get_prompt")
                prompt = get_prompt(tweet, 'too_long', lang)
                logging.warning(f"Main                          : longueur Tweet : {len(tweet)}")
                logging.warning(f"Main                          : le nouveau prompt est {prompt}")
                tweet = get_gpt_response(prompt, 0.8, lang)

                attempts += 1
            if len(tweet) > max_tweet_length:
                push_last_log_to_web()
                logging.warning(f"Main                          : impossible de tweeter : {tweet} (longueur : {len(tweet)})")
            else:
                # supprimer les espaces en double ainsi que les hashtags collés
                tweet = re.sub(r'#(\w+)', r' #\1', tweet)
                tweet = re.sub(r'\s+', ' ', tweet)
                logging.info(f"   Main                          : appel fonction publish_tweet")
                publish_tweet(client, tweet, article_url)
                push_last_log_to_web()
                logging.info(f"   Main                          : fin main() OK")
        else:
            logging.warning("Main                          : le tweet est vide !")
    except Exception as e:
        logging.error(f" Main                          : une erreur critique s'est produite : {e}")
        push_last_log_to_web()
        pass


if __name__ == "__main__":
    main()