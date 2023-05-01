#!/usr/bin/env python3
# Imports du module standard Python
import logging
import os
import random
import sys
from typing import List

# Imports de modules externes
import requests
import json
import yaml
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from fuzzywuzzy import fuzz
from GoogleNews import GoogleNews
from newspaper import Article
import pyshorteners
import tweepy
import feedparser

# Téléchargements de données pour nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Création du répertoire de log s'il n'existe pas
log_dir = "/var/log/leRobotKiTweet"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configuration du logging
logging.basicConfig(
    filename=f"{log_dir}/Twitos.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
published_articles = set()
script_name = os.path.basename(__file__)
logging.info("{:=^90}".format(" {} ".format(script_name)))

# Chargement du fichier de conf
logging.info('Chargement du fichier de configuration')
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(f'{current_dir}/application.yml', 'r') as stream:
    config = yaml.safe_load(stream)

# Récupération des titres déjà traités
logging.info('Récupération des titres déjà traités')

published_articles = set()
google_trend_rss = 'https://trends.google.com/trends/trendingsearches/daily/rss?geo=FR&hl=fr'


def article_is_published(articles_data_published: List[str], article_to_compare: str, tolerance: int = 50) -> bool:
    logging.info(f'Lancement de la fonction article_is_published')
    lemmatizer = WordNetLemmatizer()
    tokens_to_compare = word_tokenize(article_to_compare)
    lemmas_to_compare = [lemmatizer.lemmatize(token) for token in tokens_to_compare]
    for article_published in articles_data_published:
        tokens_published = word_tokenize(article_published)
        lemmas_published = [lemmatizer.lemmatize(token) for token in tokens_published]
        ratio = fuzz.ratio(' '.join(tokens_published), ' '.join(tokens_to_compare))
        if ratio >= tolerance:
            logging.info(f'Le ratio vaut : {ratio}, l\'article a déjà été publié')
            return True
        else:
            logging.info(f'Le ratio vaut : {ratio}, l\'article n\'a pas été publié, on le publie')
            return False


def check_subject(subject: str):
    logging.info(f'Lancement de la fonction check_subject avec le paramètre subject : {subject}')
    subjects_list = ['etienne_klein', 'humeur_matin', 'humeur_soir', 'google_trends']
    if subject not in subjects_list:
        print('subject not autorized')
        sys.exit(1)


def get_articles(google_news, excluded_terms=None):
    logging.info(f'Lancement de la fonction get_articles')
    s = pyshorteners.Shortener()
    if excluded_terms is None:
        excluded_terms = []
    if len(google_news) > 0:
        for article in google_news[1:50]:
            titre = article["title"]
            if not safe_search(titre, 'title'):
                print('Title not safe')
                logging.info(f'L\'article {titre} a été rejeté')
                continue
            published_articles = get_published_articles()
            published_articles = list(published_articles)
            if article_is_published(published_articles, titre):
                logging.info(f'L article {titre} a déjà été publié.')
                continue
            short_url = s.tinyurl.short(article["link"])
            article_url = article["link"]
            if not safe_search(article_url, 'url'):
                logging.warning(f'L\'url contient des mots indterdits : {article_url}')
                continue
            logging.info(f'Titre article: {titre}')
            logging.info(f"Date parution article : {article['date']}")
            logging.info(f"Url article {article_url}")
            article = Article(article_url)
            try:
                article.download()
            except:
                logging.error(f'Impossible de télécharger l\'article {article_url}')
                continue
            article.html
            article.parse()
            article_content = article.text
            article_date = article.publish_date
            formated_content = unidecode(article_content.replace(" ", "-").lower())
            if not safe_search(formated_content, 'article_content'):
                logging.info('Le contenu de l\'article contient des termes indterdits')
                continue
            return article_content, short_url
        return None
    else:
        logging.warning(f"Aucun résultat n'a été trouvé pour la recherche")
        return None


def get_google_news(subject: str, lang: str = 'fr'):
    logging.info(f'Lancement de la fonction get_google_news avec le paramètre subject : {subject}')
    googlenews = GoogleNews(lang=lang, region="com", period='1d')
    googlenews.search(subject)
    googlenews = googlenews.result()
    return googlenews


def get_google_trends(url):
    logging.info(f'Lancement de la fonction get_google_trends avec le paramètre {url}')
    # Charger le flux RSS
    feed = feedparser.parse(url)
    # Initialiser la liste des résultats
    results = []
    # Parcourir tous les items du flux RSS
    for item in feed.entries:
        # Récupérer les informations nécessaires pour chaque item
        title = item.title
        logging.info(f'Titre article: {title}')
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


def get_gpt_response(prompt: str, temperature: float = 0.8):
    try:
        logging.info(f'Lancement de la fonction get_gpt_response avec les paramètre prompt et temerature: {temperature}')
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
                    "content": config['chat-GPT']['message']['content'],
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
            "max_tokens": config['chat-GPT']['max_tokens'],
        }
        logging.info("Appel de l API GPT")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = json.loads(response.content)
        logging.info(f"Nombre de token utilisés : {response_data['usage']['total_tokens']}")
        return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Erreur lors de l'appel à l'API OpenAI : {e}")
        return None


def get_prompt(article_content: str = None, subject: str = None):
    logging.info(f'Lancement de la fonction get_subject avec les paramètres article_content et subject : {subject}')
    if subject == 'humeur_matin':
        prompt = config['chat-GPT']['prompts']['message_morning']
    elif subject == 'humeur_soir':
        prompt = config['chat-GPT']['prompts']['message_evening']
    elif subject == 'etienne_klein':
        prompt = f"{config['chat-GPT']['prompts']['etienne_klein']}{article_content}"
    elif subject == 'too_long':
        prompt = f"{config['chat-GPT']['prompts']['too_long']}{article_content}"
    else:
        prompt = f"{config['chat-GPT']['prompts']['resume_article']}{article_content}"
    return prompt


def get_published_articles():
    with open(f"{log_dir}/Twitos.log", "r") as f:
        for line in f:
            if "Titre article:" in line:
                log_title = line.split("Titre article: ")[1].strip()
                published_articles.add(log_title)
    return published_articles


def get_subject():
    logging.info(f'Lancement de la fonction get_subject')
    subjects = config['subjects']
    poids = [subjects[key]['weight'] for key in subjects]
    chosen_subject = random.choices(list(subjects.keys()), weights=poids, k=1)[0]
    logging.info(f"Sujet obtenu : {chosen_subject}")
    return subjects[chosen_subject]['name'], subjects[chosen_subject]['lang']


def launch_conditions(subject):
    logging.info(f'Lancement de la fonction launch_conditions avec le paramètre {subject}')
    if os.getenv("SHELL") == "/bin/sh":
        logging.info(f"Le script est lancé par cron avec les paramètre {subject}.")
    else:
        logging.info(f"Le script est lancé manuellement avec les paramètre {subject}.")


def publish_tweet(client, tweet_content: str, short_url: str = None):
    try:
        logging.info(f'Lancement de la fonction publish_tweet avec les paramètres client, tweet_content, short_url')
        if short_url is not None:
            tweet_content += f" {short_url}"
        tweet_c = tweet_content.replace('"', '')
        tweet_content = tweet_c
        lenght_tweet = len(tweet_content)
        logging.info(f"Publication du Tweet {tweet_content}")
        logging.info(f"Longueur du tweet : {lenght_tweet}")
        response = client.create_tweet(text=tweet_content)
        return response
    except Exception as e:
        logging.error(f"Erreur lors de la publication du Tweet {tweet_content}: {e}")
        return None


def safe_search(search: str, type_search: str = ''):
    excluded_terms = config['excluded_terms']
    logging.info(f'Lancement de la fonction safe_search avec le paramètre {type_search}')
    excluded_terms_check = [unidecode(term.replace(" ", "-").lower()) for term in excluded_terms]
    search_safer = unidecode(search.replace(" ", "-").lower())
    if any(term in search_safer for term in excluded_terms_check):
        return False
    else:
        return True


def tweepy_client():
    logging.info(f'Lancement de la fonction tweepy_client')
    logging.info('Set des varoables pour l API tweepy')
    return tweepy.Client(
        consumer_key = config['tweeter_keys']['consumer_key'],
        consumer_secret = config['tweeter_keys']['consumer_secret'],
        access_token = config['tweeter_keys']['access_token'],
        access_token_secret = config['tweeter_keys']['access_token_secret']
    )


def main():
    try:
        short_url = None
        if len(sys.argv) < 2:
            subject, lang = get_subject()
            search_activated = True
        else:
            subject = " ".join(sys.argv[1:])
            check_subject(subject)
            search_activated = False
            if subject == 'etienne_klein':
                lang = config['subjects_custum']['etienne_klein']['lang']
                sujbect = 'etienne klein'
                search_activated = True
            elif subject == 'google_trends':
                subject = get_google_trends(google_trend_rss)
                lang = config['subjects_custum']['google_trends']['lang']
                search_activated = True
        launch_conditions(subject)
        if safe_search(subject, 'subject'):
            if search_activated:
                google_news = get_google_news(subject, lang)
                article_content, short_url = get_articles(google_news)
            else:
                article_content = None
        else:
            sys.exit()
        prompt = get_prompt(article_content, subject)
        client = tweepy_client()
        tweet = get_gpt_response(prompt)
        max_tweet_length = 256
        max_attempts = 3
        attempts = 0
        while len(tweet) > max_tweet_length and attempts < max_attempts:
            logging.warning(f"Impossible de tweeter : {tweet} (longueur : {len(tweet)})")
            prompt = get_prompt(tweet, 'too_long')
            tweet = get_gpt_response(prompt)
            attempts += 1
        if len(tweet) > max_tweet_length:
            logging.warning(f"Impossible de tweeter : {tweet} (longueur : {len(tweet)})")
        else:
            publish_tweet(client, tweet, short_url)
    except Exception as e:
        logging.error(f"Une erreur critique s'est produite : {e}")
        pass


if __name__ == "__main__":
    main()
