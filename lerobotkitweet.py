#!/usr/bin/env python3
import os
import sys
import time
import random
import logging
import requests
import json
from GoogleNews import GoogleNews
from newspaper import Article
import tweepy
import pyshorteners

# Création du répertoire de log s'il n'existe pas
log_dir = "/var/log/twittos"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configuration du logging
logging.basicConfig(filename=f"{log_dir}/Twitos.log",
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

def launch_conditions():
    if os.getenv('SHELL') == '/bin/sh':
        logging.info("Le script est lancé par cron.")
        nombre_aleatoire = random.randint(0, 8)
        duree_en_secondes = random.randint(45, 480)
        time.sleep(duree_en_secondes)
    else:
        logging.info("Le script est lancé manuellement.")

def tweepy_client():
    return tweepy.Client(
        consumer_key="",
        consumer_secret="",
        access_token="",
        access_token_secret=""
    )

def get_gpt_response(prompt: str):
    OPENAI_API_KEY = ""
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + OPENAI_API_KEY
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "sytem", "content": "Tu es expert en SEO Twiter. Tes réponses ne dépassent pas 260 cractères. Soient environs 30 mots",
                "role": "user", "content": prompt
            }
        ],
        "temperature": 0.7
    }
    logging.info('Appel de l API GPT')
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_data = json.loads(response.content)
    return response_data["choices"][0]["message"]["content"]

def publish_tweet(client, tweet_content: str, url_source: str = None):
    if url_source is not None:
        tweet_content += f" {url_source}"
    logging.info(f"Publication du Tweet {tweet_content}")
    print(tweet_content)
    response = client.create_tweet(text=tweet_content)

def main():
    launch_conditions()
    client = tweepy_client()

    if len(sys.argv) < 2:
        print("Usage: robotKiTweet.py [sujet]")
        sys.exit(1)

    sujet = ' '.join(sys.argv[1:])
    logging.info(f"Lancement du script avec en paramètres {sujet}")

    if sujet in ['humeur_matin', 'humeur_soir']:
        prompt = "Tu joues le role de LeRobotKiTweet, un petit robot intergalactique qui voyage dans notre monde. C'est {0}, {1} Le texte est pour twitter, il doit être {2}, ton message doit être unique et original, donner la bonne humeur, le smile et faire environ {3} mots. Il doit se terminer par 3 ou 4 hashtags.".format('le début de journée' if sujet == 'humeur_matin' else 'la fin de journée', 'et tu donnes ton humeur du jour.' if sujet == 'humeur_matin' else "tu racontes toutes les aventures que tu as faites durant celle-ci. Tu donnes ton point de vue, tes sentiments. Parfois tu as le droit d'être triste mais ton message doit être positif.", 'drole' if sujet == 'humeur_matin' else 'positif', '25' if sujet == 'humeur_matin' else '30')
        tweet_content = get_gpt_response(prompt)
        publish_tweet(client, tweet_content)
    else:
        s = pyshorteners.Shortener()
        googlenews = GoogleNews(lang='fr', region='FR')
        logging.info('Appel de l API GoogleNew')
        googlenews.search(sujet)
        result = googlenews.result()
        for idx, article in enumerate(result[1:2]):
            titre = article['title']
            url_source = s.tinyurl.short(article['link'])
            article = Article(url_source)
            article.download()
            article.html
            article.parse()
            contenu = article.text
            prompt = f"Résume le texte en input en 20 mots maximum et 250 caractères maximum également. Le texte de retour doit être écrit en francais et compter 25 mots MAXIMUM. Il doit finir par 3 ou 4 hashtag (sans tiret ni apostrophe) racoleurs, et donnant envie et vendeurs. Input : {contenu}"
            logging.info(f"Titre article: {titre}")
            tweet_content = get_gpt_response(prompt)
            logging.info(f"Publication du tweet: {tweet_content} {url_source}")
            publish_tweet(client, tweet_content, url_source)

if __name__ == "__main__":
    main()

