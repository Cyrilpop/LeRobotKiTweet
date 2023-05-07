#!/usr/bin/env python3
# Imports du module standard Python
import locale
import logging
import os
import random
import sys
from datetime import datetime, timedelta
import datetime

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
import tweepy
import feedparser
from difflib import SequenceMatcher


current_dir = os.path.dirname(os.path.abspath(__file__))
# Chargement du fichier de conf
with open(f'{current_dir}/application.yml', 'r') as stream:
    config = yaml.safe_load(stream)

log_dir = config['logging']['dir_name']
log_name = config['logging']['file_name']


# Téléchargements de données pour nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

script_name = os.path.basename(__file__)

# Création du répertoire de log s'il n'existe pas
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configuration du logging
logging.basicConfig(
    filename=f"{log_dir}/{log_name}",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logging.info("{:=^90}".format(" {} ".format(script_name)))

google_trend_rss = 'https://trends.google.com/trends/trendingsearches/daily/rss?geo=FR&hl=fr'


def article_is_published(article_to_compare: str, tolerance: int = 50, comparison_type: str = 'title') -> bool:
    """
    Vérifie si un article a déjà été publié en comparant le texte fourni avec les articles publiés dans articles.json.
    :param articles_data_published: Chemin vers le fichier JSON contenant les données des articles déjà publiés.
    :param article_to_compare: Texte à comparer avec les articles déjà publiés.
    :param tolerance: Taux de ressemblance minimum pour considérer qu'un article a été publié.
    :param comparison_type: Type de comparaison à effectuer ('title', 'url' ou 'content').
    :return: True si l'article a été publié, False sinon.
    """
    tolerance = float(tolerance) / 100
    logging.info(f'Lancement de la fonction article_is_published pour {comparison_type}')
    with open(f'{current_dir}/articles.json', 'r') as f:
        data = json.load(f)

    lemmatizer = WordNetLemmatizer()
    tokens_to_compare = word_tokenize(article_to_compare)
    lemmas_to_compare = [lemmatizer.lemmatize(token) for token in tokens_to_compare]

    for article in data['articles']:
        text_published = article[comparison_type]
        tokens_published = word_tokenize(text_published)
        lemmas_published = [lemmatizer.lemmatize(token) for token in tokens_published]
        ratio = SequenceMatcher(None, ' '.join(tokens_published), ' '.join(tokens_to_compare)).ratio()
        if ratio >= tolerance:
            return True
    return False


def check_subject(subject: str):
    logging.info(f'Lancement de la fonction check_subject avec le paramètre subject : {subject}')
    subjects_list = ['etienne_klein', 'humeur_matin', 'humeur_soir', 'google_trends', 'cinema']
    if subject not in subjects_list:
        print('subject not authorized')
        sys.exit(1)


def get_articles(google_news, excluded_terms=None):
    logging.info(f'Lancement de la fonction get_articles')
    if excluded_terms is None:
        excluded_terms = []
    if len(google_news) > 0:
        for article in google_news[1:99]:
            article_title = article["title"]
            if not safe_search(article_title, 'title'):
                print('Title not safe')
                logging.info(f'L\'article {article_title} a été rejeté')
                continue
            if article_is_published(article_title, 50, 'title'):
                logging.info(f'L article {article_title} a déjà été publié.')
                continue
            article_url = article["link"]
            if not safe_search(article_url, 'url'):
                logging.warning(f'L\'url contient des mots indterdits : {article_url}')
                continue
            logging.info(f'Titre article: {article_title}')
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
            if article_is_published(article_content, 20, 'content'):
                logging.warning(f'L\'article {article_title} a déjà até publié')
                continue
            article_date = article.publish_date
            formated_content = unidecode(article_content.replace(" ", "-").lower())
            if not safe_search(formated_content, 'article_content'):
                logging.info('Le contenu de l\'article contient des termes indterdits')
                continue
            return article_title, article_url, article_url, article_date, article_content
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
        logging.info(
            f'Lancement de la fonction get_gpt_response avec les paramètre prompt et temerature: {temperature}')
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
    elif subject == 'cinema':
        prompt = f"{config['chat-GPT']['prompts']['cinema']}{article_content}"
    elif subject == 'etienne_klein':
        prompt = f"{config['chat-GPT']['prompts']['etienne_klein']}{article_content}"
    elif subject == 'too_long':
        prompt = f"{config['chat-GPT']['prompts']['too_long']}{article_content}"
    else:
        prompt = f"{config['chat-GPT']['prompts']['resume_article']}{article_content}"
    return prompt

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


def publish_tweet(client, tweet_content: str, article_url: str = None):
    try:
        logging.info(f'Lancement de la fonction publish_tweet avec les paramètres client, tweet_content, article_url')
        if article_url is not None:
            tweet_content += f" {article_url}"
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


def push_article_json(article_title: str, article_url: str, article_content: str):
    logging.info(f'Lancement de la fonction push_article_json')
    with open(f'{current_dir}/articles.json', 'r') as f:
        articles = json.load(f)
    new_article = {
        "title": article_title,
        "url": article_url,
        "content": article_content
    }
    articles["articles"].append(new_article)
    with open(f'{current_dir}/articles.json', 'w') as f:
        json.dump(articles, f, indent=4)


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
    logging.info('Set des variables pour l API tweepy')
    return tweepy.Client(
        consumer_key=config['tweeter_keys']['consumer_key'],
        consumer_secret=config['tweeter_keys']['consumer_secret'],
        access_token=config['tweeter_keys']['access_token'],
        access_token_secret=config['tweeter_keys']['access_token_secret']
    )


def main():
    try:
        article_url = None
        if len(sys.argv) < 2:
            subject, lang = get_subject()
            search_activated = True
        else:
            subject = " ".join(sys.argv[1:])
            check_subject(subject)
            search_activated = False
            if subject == 'etienne_klein':
                lang = config['subjects_custom']['etienne_klein']['lang']
                sujbect = 'etienne klein'
                search_activated = True
            elif subject == 'cinema':
                locale.setlocale(locale.LC_ALL, 'fr_FR.utf8')
                today = datetime.date.today()
                days_ahead = (2 - today.weekday()) % 7
                next_wednesday = today + datetime.timedelta(days=days_ahead)
                next_cinema = next_wednesday.strftime('%d %B %Y')

                lang = config['subjects_custom']['cinema']['lang']
                subject = f"sorties cinema {next_cinema}"
                search_activated = True
            elif subject == 'google_trends':
                subject = get_google_trends(google_trend_rss)
                lang = config['subjects_custom']['google_trends']['lang']
                search_activated = True
        launch_conditions(subject)
        if safe_search(subject, 'subject'):
            if search_activated:
                google_news = get_google_news(subject, lang)
                article_title, article_url, article_url, article_date, article_content = get_articles(google_news)
            else:
                article_content = None
        else:
            sys.exit()
        if subject not in ['humeur_soir', 'humeur_matin']:
            push_article_json(article_title, article_url, article_content)
        prompt = get_prompt(article_content, subject)
        client = tweepy_client()
        tweet = get_gpt_response(prompt)
        max_tweet_length = 255
        max_attempts = 3
        attempts = 0
        while len(tweet) > max_tweet_length and attempts < max_attempts:
            logging.warning(f"Impossible de tweeter : {tweet} (longueur : {len(tweet)})")
            prompt = get_prompt(tweet, 'too_long')
            logging.warning(f"Longueur Tweet : {len(tweet)}")
            logging.warning(f"Le nouveau prompt est {prompt}")
            tweet = get_gpt_response(prompt)
            attempts += 1
        if len(tweet) > max_tweet_length:
            logging.warning(f"Impossible de tweeter : {tweet} (longueur : {len(tweet)})")
        else:
            publish_tweet(client, tweet, article_url)
    except Exception as e:
        logging.error(f"Une erreur critique s'est produite : {e}")
        pass


if __name__ == "__main__":
    main()
