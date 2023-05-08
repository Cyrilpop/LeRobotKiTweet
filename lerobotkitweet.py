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
import numpy as np

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
    logging.info(f"Fonction article_is_published : lancement")
    logging.info(f"Fonction article_is_published : paramètre  article_to_compare")
    logging.info(f"Fonction article_is_published : paramètre  tolerance {tolerance}")
    logging.info(f"Fonction article_is_published : paramètre  comparison_type {comparison_type}")
    logging.info(f"Fonction article_is_published : paramètre  {comparison_type}: {article_to_compare}")
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
        logging.debug(f"Fonction article_is_published : ratio: {ratio}")
        if ratio >= tolerance:
            logging.info(f"Fonction article_is_published : return True")
            return True
    logging.info(f"Fonction article_is_published : return False")
    return False


def check_subject(subject: str):
    logging.info(f"Fonction check_subject : lancement")
    logging.info(f"Fonction check_subject : subject : {subject}")
    authorized_subjects = list(config['subjects_custom'].keys())
    if subject not in authorized_subjects:
        logging.info(f"Fonction check_subject : {subject} n\'est pas autorisé")
        print('sujet non autorisé')
        push_last_log_to_web()
        sys.exit(1)


def get_articles(google_news, excluded_terms=None):
    logging.info(f"Fonction get_articles : lancement")
    logging.info(f"Fonction get_articles : paramètre google_news")
    if excluded_terms is None:
        excluded_terms = []
    if len(google_news) > 0:
        for article in google_news[1:99]:
            article_title = article["title"]
            if not safe_search(article_title):
                print('Title not safe')
                logging.info(f"Fonction get_articles : article {article_title} a été rejeté")
                continue
            if article_is_published(article_title, 80, 'title'):
                logging.info(f"Fonction get_articles : article {article_title} a déjà été publié.")
                continue
            article_url = article["link"]
            if not safe_search(article_url):
                logging.warning(f"Fonction get_articles : URL {article_url} a été rejeté")
                continue
            logging.info(f"Fonction get_articles : titre article: {article_title}")
            logging.info(f"Fonction get_articles : date parution article : {article['date']}")
            logging.info(f"Fonction get_articles : URL article : {article_url}")
            article = Article(article_url)
            try:
                article.download()
            except:
                logging.error(f"Fonction get_articles : Impossible de télécharger {article_url}")
                continue
            article.html
            article.parse()
            article_content = article.text
            article_date = article.publish_date
            formated_content = unidecode(article_content.replace(" ", "-").lower())
            if not safe_search(formated_content):
                logging.info("Fonction get_articles : le contenu de l'article a été rejeté (contenu)")
                continue
            return article_title, article_url, article_date, article_content
        return None
    else:
        logging.warning(f"Fonction get_articles : aucun résultat n'a été trouvé pour la recherche")
        push_last_log_to_web()
        sys.exit(1)


def get_google_news(subject: str, lang: str = 'fr'):
    logging.info(f"Fonction get_google_news : lancement")
    logging.info(f"Fonction get_google_news : paramètre  subject : {subject}")
    googlenews = GoogleNews(lang=lang, region="com", period='1d')
    try:
        googlenews.search(subject)
        if not googlenews.result():
            logging.error("Fonction get_google_news : aucun résultat n'a été trouvé pour la recherche.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Fonction get_google_news : erreur s'est produite lors de l'appel à la méthode search() : {e}")
        sys.exit()
    try:
        googlenews = googlenews.result()
    except AttributeError as e:
        logging.error(f"Fonction get_google_news : une erreur s'est produite lors de l'appel à la méthode result() : {e}")
        sys.exit()
    return googlenews


def get_google_trends(url):
    logging.info(f"Fonction get_google_news : lancement")
    logging.info(f"Fonction get_google_news : paramètre url : {url}")
    # Charger le flux RSS
    feed = feedparser.parse(url)
    # Initialiser la liste des résultats
    results = []
    # Parcourir tous les items du flux RSS
    for item in feed.entries:
        # Récupérer les informations nécessaires pour chaque item
        title = item.title
        logging.info(f"Fonction get_google_news : titre article: {title}")
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
        logging.info(f"Fonction get_gpt_response : lancement")
        logging.info(f"Fonction get_gpt_response : paramètre prompt")
        logging.info(f"Fonction get_gpt_response : paramètre temerature: {temperature}")
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
        logging.info("Fonction get_gpt_response : appel de l'API GPT")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = json.loads(response.content)
        logging.info(f"Fonction get_gpt_response : nombre de token utilisés : {response_data['usage']['total_tokens']}")
        return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Fonction get_gpt_response : erreur lors de l'appel à l'API OpenAI : {response_data['error']['message']}")
        print(e)
        push_last_log_to_web()
        sys.exit(1)


def get_prompt(article_content: str = None, subject: str = None):
    logging.info(f"Fonction get_prompt : lancement")
    logging.info(f"Fonction get_prompt : paramètre article_content")
    logging.info(f"Fonction get_prompt : paramètre subject : {subject}")
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
    logging.info(f"Fonction get_subject : lancement")
    subjects = config['subjects']
    poids = [subjects[key]['weight'] for key in subjects]
    chosen_subject = random.choices(list(subjects.keys()), weights=poids, k=1)[0]
    logging.info(f"Fonction get_prompt : sujet obtenu : {chosen_subject}")
    return subjects[chosen_subject]['name'], subjects[chosen_subject]['lang']

def get_twitter_trends():
    logging.info(f"Fonction get_twitter_trends : lancement")
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
    logging.info(f"Fonction get_twitter_trends : trend obtenu : {trend_arrays[0][0]}")
    return trend_arrays

def launch_conditions(subject):
    logging.info(f"Fonction launch_conditions : lancement")
    logging.info(f"Fonction launch_conditions : paramètre subject : {subject}")
    if os.getenv("SHELL") == "/bin/sh":
        logging.info(f"Fonction launch_conditions : script lancé par cron")
    else:
        logging.info(f"Fonction launch_conditions : script lancé manuellement")


def publish_tweet(client, tweet_content: str, article_url: str = None):
    try:
        logging.info(f"Fonction publish_tweet : lancement")
        logging.info(f"Fonction publish_tweet : paramètre client {client}")
        logging.info(f"Fonction publish_tweet : paramètre tweet_content {tweet_content}")
        logging.info(f"Fonction publish_tweet : paramètre article_url {article_url}")
        if article_url is not None:
            tweet_content += f" {article_url}"
        tweet_content = tweet_content.replace('"', '')
        logging.info(f"Fonction publish_tweet : publication du Tweet")
        logging.info(f"Fonction publish_tweet : longueur du tweet : {len(tweet_content)}")
        response = client.create_tweet(text=tweet_content)
        return response
    except Exception as e:
        logging.error(f"Fonction publish_tweet : erreur lors de la publication du Tweet {tweet_content}: {e}")
        push_last_log_to_web()
        return None


def push_article_json(article_title: str, article_url: str):
    logging.info(f"Fonction push_article_json : lancement")
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
    logging.info(f"Fonction push_last_log_to_web : lancement")
    with open(f"{log_dir}/{log_name}","r") as source_file:
        lines = source_file.readlines()[-200:]
        lines.reverse()
        with open('/var/www/html/lrto.txt',"w") as target_file:
            for line in lines:
                target_file.write(line)

def safe_search(search: str):
    excluded_terms = config['excluded_terms']
    logging.info(f"Fonction safe_search : lancement")
    logging.info(f"Fonction safe_search : paramètre search")
    excluded_terms_check = [unidecode(term.replace(" ", "-").lower()) for term in excluded_terms]
    search_safer = unidecode(search.replace(" ", "-").lower())
    for term in excluded_terms_check:
        if term in search_safer:
            logging.info(f"Fonction safe_search : terme exclu {term} trouvé dans la recherche")
            logging.info(f"Fonction safe_search : return False")
            return False
    logging.info(f"Fonction safe_search : return True")
    return True


def tweepy_client():
    logging.info(f"Fonction safe_search : lancement")
    return tweepy.Client(
        consumer_key=config['tweeter_keys']['consumer_key'],
        consumer_secret=config['tweeter_keys']['consumer_secret'],
        access_token=config['tweeter_keys']['access_token'],
        access_token_secret=config['tweeter_keys']['access_token_secret']
    )


def main():
    try:
        # Initialisation des variables
        article_url = None
        hashtag = ""

        # Vérification des arguments passés au script
        if len(sys.argv) < 2:
            # Récupération d'un sujet aléatoire et de sa langue associée
            subject, lang = get_subject()
            search_activated = True
        else:
            # Utilisation du sujet spécifié en ligne de commande
            subject = " ".join(sys.argv[1:])
            check_subject(subject)
            search_activated = False
            if subject == 'etienne_klein':
                # Si le sujet est Etienne Klein, on utilise la langue spécifique et on active la recherche
                lang = config['subjects_custom']['etienne_klein']['lang']
                sujbect = 'etienne klein'
                search_activated = True
            elif subject == 'cinema':
                # Si le sujet est cinema, on construit le sujet à partir de la date de la prochaine sortie cinéma et on active la recherche
                locale.setlocale(locale.LC_ALL, 'fr_FR.utf8')
                today = datetime.date.today()
                days_ahead = (2 - today.weekday()) % 7
                next_wednesday = today + datetime.timedelta(days=days_ahead)
                next_cinema = next_wednesday.strftime('%d %B %Y')
                lang = config['subjects_custom']['cinema']['lang']
                subject = f"sorties cinema {next_cinema}"
                search_activated = True
            elif subject == 'google_trends':
                # Si le sujet est Google Trends, on utilise le sujet obtenu depuis le flux RSS de Google Trends et on active la recherche
                subject = get_google_trends(google_trend_rss)
                lang = config['subjects_custom']['google_trends']['lang']
                search_activated = True
            elif subject == 'twitter_trends':
                # Si le sujet est Twitter Trends, on utilise le premier sujet dans la liste des tendances Twitter et on active la recherche
                trends_array = get_twitter_trends()
                twitter_trend = trends_array[0][0]
                if "#" in twitter_trend:
                    hashtag = twitter_trend
                else:
                    hashtag = ""
                subject, lang, search_activated = twitter_trend.replace('#',''), config['subjects_custom']['twitter_trends']['lang'], True
        launch_conditions(subject)

        # Vérification si le sujet est autorisé par la fonction safe_search()
        if safe_search(subject):
            if search_activated:
                # Récupération des articles de Google News liés au sujet
                google_news = get_google_news(subject, lang)
                # Extraction de l'article pertinent
                article_title, article_url, article_date, article_content = get_articles(google_news)
            else:
                article_content = None
        else:
            # Si le sujet n'est pas autorisé par safe_search(), on quitte le script
            push_last_log_to_web()
            sys.exit()

        # Ajout de l'article dans le fichier JSON des articles
        if subject not in ['humeur_soir', 'humeur_matin']:
            push_article_json(article_title, article_url)

        # Obtention de la phrase d'accroche à partir de l'article
        prompt = get_prompt(article_content, subject)
        client = tweepy_client()

        # Génération de la réponse GPT-3 à partir de la phrase d'accroche
        tweet = f"{get_gpt_response(prompt)}{hashtag}"

        # Vérification si le tweet n'est pas vide
        if tweet is not None:
            # Vérification si le tweet est trop long
            max_tweet_length = 250
            max_attempts = 3
            attempts = 0
            while len(tweet) > max_tweet_length and attempts < max_attempts:
                logging.warning(f"Main : impossible de tweeter : {tweet} (longueur : {len(tweet)})")
                prompt = get_prompt(tweet, 'too_long')
                logging.warning(f"Main : longueur Tweet : {len(tweet)}")
                logging.warning(f"Main : le nouveau prompt est {prompt}")
                tweet = get_gpt_response(prompt)
                attempts += 1
            if len(tweet) > max_tweet_length:
                push_last_log_to_web()
                logging.warning(f"Main : impossible de tweeter : {tweet} (longueur : {len(tweet)})")
            else:
                publish_tweet(client, tweet, article_url)
                push_last_log_to_web()
        else:
            logging.warning("Main : le tweet est vide !")
    except Exception as e:
        push_last_log_to_web()
        logging.error(f"Main : une erreur critique s'est produite : {e}")
        pass

if __name__ == "__main__":
    main()